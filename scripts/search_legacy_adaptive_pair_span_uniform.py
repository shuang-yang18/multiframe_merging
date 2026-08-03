#!/usr/bin/env python3
"""Concurrent TUM300 pair/span search for adaptive multiframe FastVGGT.

Workers share one locked queue.  A 0.01 coarse neighborhood around the current
legacy setting is evaluated first.  The completed coarse ranking seeds a 0.001
local search that follows the current AUC@3 leader after every completed run.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import shutil
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any


PAIR_SEED = 0.986
SPAN_SEED = 0.948
LOWER_BOUND = 0.92
UPPER_BOUND = 0.999


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument(
        "--result-root",
        type=Path,
        default=root / "auc_eval_results" / "01" / "ours_adaptive_pair_span_uniform300_search",
    )
    parser.add_argument(
        "--max-fine-trials",
        type=int,
        default=0,
        help="Optional cap on fine candidates; 0 means no cap and is the default.",
    )
    parser.add_argument("--set-unbounded", action="store_true", help="Update an existing search state to remove its fine-search cap.")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    return parser.parse_args()


def clamp(value: float) -> float:
    return round(min(UPPER_BOUND, max(LOWER_BOUND, value)), 3)


def candidate_id(pair: float, span: float) -> str:
    return f"p{pair:.3f}_s{span:.3f}".replace(".", "")


def coarse_candidates() -> list[dict[str, Any]]:
    offsets = [
        (0.00, 0.00),
        (-0.01, 0.00), (0.01, 0.00), (0.00, -0.01), (0.00, 0.01),
        (-0.01, -0.01), (-0.01, 0.01), (0.01, -0.01), (0.01, 0.01),
        (-0.02, 0.00), (0.02, 0.00), (0.00, -0.02), (0.00, 0.02),
        (-0.02, -0.02), (0.02, 0.02), (-0.02, 0.02), (0.02, -0.02),
    ]
    seen: set[str] = set()
    candidates = []
    for pair_offset, span_offset in offsets:
        pair = clamp(PAIR_SEED + pair_offset)
        span = clamp(SPAN_SEED + span_offset)
        identifier = candidate_id(pair, span)
        if identifier in seen:
            continue
        seen.add(identifier)
        candidates.append({"id": identifier, "pair_threshold": pair, "span_threshold": span, "stage": "coarse", "status": "pending"})
    return candidates


@contextmanager
def locked_state(result_root: Path):
    result_root.mkdir(parents=True, exist_ok=True)
    lock_path = result_root / ".queue.lock"
    with lock_path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        state_path = result_root / "search_state.json"
        state = json.loads(state_path.read_text()) if state_path.is_file() else None
        yield state
        if state is not None:
            state_path.write_text(json.dumps(state, indent=2) + "\n")
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def initialize(result_root: Path, max_fine_trials: int) -> None:
    result_root.mkdir(parents=True, exist_ok=True)
    lock_path = result_root / ".queue.lock"
    with lock_path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        state_path = result_root / "search_state.json"
        if state_path.is_file():
            raise FileExistsError(f"Search state already exists: {state_path}")
        state = {
            "protocol": "adaptive_pair_span_uniform300_v1",
            "dataset": "tum_dynamic",
            "frame_selection_protocol": "uniform_full_sequence_v1",
            "pair_seed": PAIR_SEED,
            "span_seed": SPAN_SEED,
            "bounds": [LOWER_BOUND, UPPER_BOUND],
            "max_fine_trials": max_fine_trials,
            "stage": "coarse",
            "candidates": coarse_candidates(),
            "results": [],
        }
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    print(f"Initialized {len(coarse_candidates())} coarse candidates in {result_root}", flush=True)


def pending_or_running(state: dict[str, Any], stage: str) -> bool:
    return any(candidate["stage"] == stage and candidate["status"] in {"pending", "running"} for candidate in state["candidates"])


def result_key(result: dict[str, Any]) -> tuple[float, float]:
    return float(result["auc3"]), float(result.get("fps", 0.0))


def best_result(state: dict[str, Any]) -> dict[str, Any] | None:
    successful = [result for result in state["results"] if result.get("status") == "complete"]
    return max(successful, key=result_key) if successful else None


def append_candidate(state: dict[str, Any], pair: float, span: float, stage: str) -> bool:
    pair, span = clamp(pair), clamp(span)
    identifier = candidate_id(pair, span)
    if any(candidate["id"] == identifier for candidate in state["candidates"]):
        return False
    state["candidates"].append(
        {"id": identifier, "pair_threshold": pair, "span_threshold": span, "stage": stage, "status": "pending"}
    )
    return True


def add_fine_neighborhood(state: dict[str, Any], center_pair: float, center_span: float, radius: int) -> None:
    fine_count = sum(candidate["stage"] == "fine" for candidate in state["candidates"])
    limit = int(state["max_fine_trials"])
    for pair_step in range(-radius, radius + 1):
        for span_step in range(-radius, radius + 1):
            if limit > 0 and fine_count >= limit:
                return
            if append_candidate(state, center_pair + pair_step * 0.001, center_span + span_step * 0.001, "fine"):
                fine_count += 1


def add_fine_ring(state: dict[str, Any], center_pair: float, center_span: float, radius: int) -> int:
    fine_count = sum(candidate["stage"] == "fine" for candidate in state["candidates"])
    limit = int(state["max_fine_trials"])
    added = 0
    for pair_step in range(-radius, radius + 1):
        for span_step in range(-radius, radius + 1):
            if max(abs(pair_step), abs(span_step)) != radius:
                continue
            if limit > 0 and fine_count >= limit:
                return added
            if append_candidate(state, center_pair + pair_step * 0.001, center_span + span_step * 0.001, "fine"):
                fine_count += 1
                added += 1
    return added


def advance_stage(state: dict[str, Any]) -> None:
    if state["stage"] == "coarse" and not pending_or_running(state, "coarse"):
        best = best_result(state)
        if best is None:
            state["stage"] = "complete"
            return
        state["stage"] = "fine"
        state["fine_radius"] = 3
        add_fine_neighborhood(state, float(best["pair_threshold"]), float(best["span_threshold"]), radius=3)
    elif state["stage"] == "fine":
        best = best_result(state)
        if not pending_or_running(state, "fine"):
            if best is None:
                state["stage"] = "complete"
                return
            next_radius = int(state.get("fine_radius", 3)) + 1
            added = add_fine_ring(
                state,
                float(best["pair_threshold"]),
                float(best["span_threshold"]),
                next_radius,
            )
            state["fine_radius"] = next_radius
            if added == 0:
                state["stage"] = "complete"


def write_ranking(result_root: Path, state: dict[str, Any]) -> None:
    rows = sorted(
        (result for result in state["results"] if result.get("status") == "complete"),
        key=result_key,
        reverse=True,
    )
    if rows:
        with (result_root / "results_by_auc3.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        (result_root / "best_by_auc3.json").write_text(json.dumps(rows[0], indent=2) + "\n")


def claim_candidate(result_root: Path, worker_gpu: int) -> tuple[str, dict[str, Any] | None]:
    with locked_state(result_root) as state:
        if state is None:
            raise FileNotFoundError("Run with --initialize before launching workers")
        advance_stage(state)
        for candidate in state["candidates"]:
            if candidate["status"] == "pending":
                candidate["status"] = "running"
                candidate["gpu"] = worker_gpu
                candidate["started_at"] = time.time()
                return "run", dict(candidate)
        return ("done", None) if state["stage"] == "complete" else ("wait", None)


def read_metrics(summary_path: Path) -> dict[str, float]:
    payload = json.loads(summary_path.read_text())
    depth = payload["video_depth"]
    pose = payload["pose_auc"]
    return {
        "auc3": float(pose["AUC@3"]),
        "auc30": float(pose["AUC@30"]),
        "abs_rel": float(depth["Abs Rel"]),
        "delta125": float(depth["delta < 1.25"]),
        "fps": float(depth["fps"]),
    }


def archive_summaries(candidate_root: Path, result_root: Path, identifier: str) -> None:
    source = candidate_root / "tum_dynamic"
    destination = result_root / "summaries" / identifier
    destination.mkdir(parents=True, exist_ok=True)
    for name in (
        "_summary_complete_scale_shift.json",
        "_summary_scale_shift.json",
        "_summary_pose_auc.json",
        "_sequence_metrics_scale_shift.csv",
        "_sequence_pose_auc.csv",
    ):
        path = source / name
        if path.is_file():
            shutil.copy2(path, destination / name)
    sequence_dirs = source.iterdir() if source.is_dir() else []
    for sequence_dir in sequence_dirs:
        if not sequence_dir.is_dir():
            continue
        saved_sequence = destination / sequence_dir.name
        saved_sequence.mkdir(exist_ok=True)
        for name in ("_pose_auc.json", "_time.json", "_input_frames.json"):
            path = sequence_dir / name
            if path.is_file():
                shutil.copy2(path, saved_sequence / name)


def run_candidate(root: Path, result_root: Path, gpu: int, candidate: dict[str, Any]) -> dict[str, Any]:
    candidate_root = result_root / "_temporary" / candidate["id"]
    shutil.rmtree(candidate_root, ignore_errors=True)
    candidate_root.mkdir(parents=True, exist_ok=True)
    log_path = result_root / "logs" / f"{candidate['id']}_gpu{gpu}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "/data/mmc_syang/miniconda3/envs/fastvggt/bin/python",
        str(root / "inference" / "infer.py"),
        "--dataset", "tum_dynamic",
        "--dataset-root", "/data/mmc_syang/dataset/TUM-Dynamics",
        "--output-dir", str(candidate_root),
        "--max-frames-per-seq", "300",
        "--window-size", "0",
        "--checkpoint", str(root / "checkpoints" / "vggt_omega_1b_512.pt"),
        "--overwrite", "--eval", "--eval-align", "scale_shift",
        "--pose-eval-frames", "0", "--pose-eval-seed", "0",
        "--omega-accelerator", "none",
        "--enable-token-merging",
        "--token-merging-method", "frame_persistent_adaptive_spatial",
        "--token-merging-ratio", "0.9",
        "--token-merging-layer-ratios", "1-10:0.9,11-18:0.0,19-24:0.9",
        "--token-merging-start", "0",
        "--token-merging-frame-restore-layer", "24",
        "--token-merging-frame-alpha", "0.1",
        "--token-merging-frame-segment-threshold", "0.9",
        "--token-merging-frame-merge-threshold", "0.1",
        "--token-merging-frame-max-window", "20",
        "--token-merging-frame-pool-stride", "2",
        "--token-merging-frame-multi-max-group-size", "4",
        "--token-merging-frame-multi-pair-threshold", f"{candidate['pair_threshold']:.3f}",
        "--token-merging-frame-multi-span-threshold", f"{candidate['span_threshold']:.3f}",
        "--token-merging-frame-group-strategy", "local",
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "PYTHONPATH": str(root),
            "PYTHONNOUSERSITE": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "HF_HOME": str(root / ".cache" / "huggingface"),
            "TRANSFORMERS_CACHE": str(root / ".cache" / "huggingface" / "hub"),
        }
    )
    with log_path.open("w") as handle:
        subprocess.run(command, cwd=root, env=environment, stdout=handle, stderr=subprocess.STDOUT, check=True)
    summary_path = candidate_root / "tum_dynamic" / "_summary_complete_scale_shift.json"
    metrics = read_metrics(summary_path)
    archive_summaries(candidate_root, result_root, candidate["id"])
    shutil.rmtree(candidate_root, ignore_errors=True)
    return metrics


def finish_candidate(result_root: Path, candidate: dict[str, Any], metrics: dict[str, float] | None, error: str | None) -> None:
    with locked_state(result_root) as state:
        if state is None:
            raise RuntimeError("Search state disappeared")
        stored = next(item for item in state["candidates"] if item["id"] == candidate["id"])
        stored["status"] = "complete" if error is None else "failed"
        stored["finished_at"] = time.time()
        result: dict[str, Any] = {
            "candidate": candidate["id"],
            "stage": candidate["stage"],
            "pair_threshold": candidate["pair_threshold"],
            "span_threshold": candidate["span_threshold"],
            "gpu": candidate["gpu"],
            "status": stored["status"],
        }
        if metrics is not None:
            result.update(metrics)
        if error is not None:
            result["error"] = error
        state["results"].append(result)
        advance_stage(state)
        write_ranking(result_root, state)


def worker(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[1]
    result_root = args.result_root.resolve()
    while True:
        action, candidate = claim_candidate(result_root, args.gpu)
        if action == "done":
            print(f"GPU {args.gpu}: search complete", flush=True)
            return
        if action == "wait":
            time.sleep(args.poll_seconds)
            continue
        assert candidate is not None
        print(f"GPU {args.gpu}: starting {candidate['id']} ({candidate['stage']})", flush=True)
        try:
            metrics = run_candidate(root, result_root, args.gpu, candidate)
            finish_candidate(result_root, candidate, metrics, None)
            print(f"GPU {args.gpu}: {candidate['id']} AUC@3={metrics['auc3']:.6f}", flush=True)
        except Exception as error:  # Keep the queue moving after one failed candidate.
            shutil.rmtree(result_root / "_temporary" / candidate["id"], ignore_errors=True)
            finish_candidate(result_root, candidate, None, repr(error))
            print(f"GPU {args.gpu}: {candidate['id']} failed: {error!r}", flush=True)


def set_unbounded(result_root: Path) -> None:
    with locked_state(result_root) as state:
        if state is None:
            raise FileNotFoundError(f"No search state in {result_root}")
        state["max_fine_trials"] = 0
    print(f"Removed fine-search cap in {result_root}", flush=True)


def main() -> None:
    args = parse_args()
    modes = int(args.initialize) + int(args.worker) + int(args.set_unbounded)
    if modes != 1:
        raise ValueError("Pass exactly one of --initialize, --worker, or --set-unbounded")
    if args.max_fine_trials < 0:
        raise ValueError("--max-fine-trials must be non-negative")
    if args.initialize:
        initialize(args.result_root.resolve(), args.max_fine_trials)
    elif args.set_unbounded:
        set_unbounded(args.result_root.resolve())
    else:
        if args.gpu is None:
            raise ValueError("--gpu is required for --worker")
        worker(args)


if __name__ == "__main__":
    main()
