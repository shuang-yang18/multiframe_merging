#!/usr/bin/env python3
"""Unbounded 0.001 pair/span search for temporary per-global-block frame fusion.

All candidates use frame_temporary_adaptive_spatial with the 50% upper-rate
adaptation disabled. Workers share a file-locked queue and expand an untested
0.001 Chebyshev ring around the current TUM AUC@3 leader indefinitely.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any


PAIR_SEED = 0.960
SPAN_SEED = 0.950
LOWER_BOUND = 0.920
UPPER_BOUND = 0.999


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--reset-failed", action="store_true")
    parser.add_argument(
        "--result-root",
        type=Path,
        default=root / "auc_eval_results" / "01" / "temporary_frame_pair_span_search_p0960_s0950",
    )
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    return parser.parse_args()


def clamp(value: float) -> float:
    return round(min(UPPER_BOUND, max(LOWER_BOUND, value)), 3)


def candidate_id(pair: float, span: float) -> str:
    return f"p{pair:.3f}_s{span:.3f}".replace(".", "")


@contextmanager
def locked_state(result_root: Path):
    result_root.mkdir(parents=True, exist_ok=True)
    with (result_root / ".queue.lock").open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        path = result_root / "search_state.json"
        state = json.loads(path.read_text()) if path.is_file() else None
        yield state
        if state is not None:
            path.write_text(json.dumps(state, indent=2) + "\n")
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def add_candidate(state: dict[str, Any], pair: float, span: float, stage: str) -> bool:
    pair, span = clamp(pair), clamp(span)
    if span > pair:
        return False
    identifier = candidate_id(pair, span)
    if any(item["id"] == identifier for item in state["candidates"]):
        return False
    state["candidates"].append(
        {
            "id": identifier,
            "pair_threshold": pair,
            "span_threshold": span,
            "stage": stage,
            "status": "pending",
        }
    )
    return True


def best_result(state: dict[str, Any]) -> dict[str, Any] | None:
    completed = [item for item in state["results"] if item.get("status") == "complete"]
    return max(completed, key=lambda item: (float(item["auc3"]), float(item.get("fps", 0.0)))) if completed else None


def add_ring(state: dict[str, Any], pair: float, span: float, radius: int, stage: str) -> int:
    added = 0
    for pair_step in range(-radius, radius + 1):
        for span_step in range(-radius, radius + 1):
            if max(abs(pair_step), abs(span_step)) != radius:
                continue
            if add_candidate(state, pair + pair_step * 0.001, span + span_step * 0.001, stage):
                added += 1
    return added


def initialize(result_root: Path) -> None:
    result_root.mkdir(parents=True, exist_ok=True)
    state_path = result_root / "search_state.json"
    with (result_root / ".queue.lock").open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        if state_path.is_file():
            raise FileExistsError(f"Search state already exists: {state_path}")
        state = {
            "protocol": "temporary_frame_pair_span_auc3_v1",
            "dataset": "tum_dynamic",
            "frame_selection_protocol": "uniform_full_sequence_v1",
            "upper_adaptive": False,
            "pair_seed": PAIR_SEED,
            "span_seed": SPAN_SEED,
            "bounds": [LOWER_BOUND, UPPER_BOUND],
            "step": 0.001,
            "next_radius": 1,
            "candidates": [],
            "results": [],
        }
        add_candidate(state, PAIR_SEED, SPAN_SEED, "seed")
        add_ring(state, PAIR_SEED, SPAN_SEED, radius=1, stage="ring_1")
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    print(f"Initialized temporary search in {result_root}", flush=True)


def advance_queue(state: dict[str, Any]) -> None:
    if any(item["status"] in {"pending", "running"} for item in state["candidates"]):
        return
    best = best_result(state)
    if best is None:
        return
    radius = int(state["next_radius"]) + 1
    center_pair = float(best["pair_threshold"])
    center_span = float(best["span_threshold"])
    while radius <= int(round((UPPER_BOUND - LOWER_BOUND) / 0.001)):
        if add_ring(state, center_pair, center_span, radius, f"ring_{radius}"):
            state["next_radius"] = radius
            return
        radius += 1
    # The finite domain can eventually be exhausted. Keep workers alive so a
    # user can inspect or extend the bounds without relaunching sessions.
    state["next_radius"] = radius


def claim_candidate(result_root: Path, gpu: int) -> dict[str, Any] | None:
    with locked_state(result_root) as state:
        if state is None:
            raise FileNotFoundError("Run with --initialize before starting workers")
        advance_queue(state)
        for item in state["candidates"]:
            if item["status"] == "pending":
                item["status"] = "running"
                item["gpu"] = gpu
                item["started_at"] = time.time()
                return dict(item)
    return None


def archive_summaries(candidate_root: Path, result_root: Path, identifier: str) -> None:
    source = candidate_root / "tum_dynamic"
    target = result_root / "summaries" / identifier
    target.mkdir(parents=True, exist_ok=True)
    for name in (
        "_summary_complete_scale_shift.json",
        "_summary_scale_shift.json",
        "_summary_pose_auc.json",
        "_sequence_metrics_scale_shift.csv",
        "_sequence_pose_auc.csv",
    ):
        if (source / name).is_file():
            shutil.copy2(source / name, target / name)
    for sequence_dir in source.iterdir() if source.is_dir() else []:
        if sequence_dir.is_dir():
            saved = target / sequence_dir.name
            saved.mkdir(exist_ok=True)
            for name in ("_time.json", "_pose_auc.json", "_input_frames.json"):
                if (sequence_dir / name).is_file():
                    shutil.copy2(sequence_dir / name, saved / name)


def run_candidate(root: Path, result_root: Path, gpu: int, candidate: dict[str, Any]) -> dict[str, float]:
    candidate_root = result_root / "_temporary" / candidate["id"]
    shutil.rmtree(candidate_root, ignore_errors=True)
    candidate_root.mkdir(parents=True, exist_ok=True)
    log_path = result_root / "logs" / f"{candidate['id']}_gpu{gpu}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
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
        "--token-merging-method", "frame_temporary_adaptive_spatial",
        "--token-merging-layer-ratios", "1-10:0.9,11-18:0.0,19-24:0.9",
        "--token-merging-frame-alpha", "0.1",
        "--token-merging-frame-segment-threshold", "0.9",
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
    complete = json.loads((candidate_root / "tum_dynamic" / "_summary_complete_scale_shift.json").read_text())
    metrics = {
        "auc3": float(complete["pose_auc"]["AUC@3"]),
        "auc30": float(complete["pose_auc"]["AUC@30"]),
        "abs_rel": float(complete["video_depth"]["Abs Rel"]),
        "delta125": float(complete["video_depth"]["delta < 1.25"]),
        "fps": float(complete["video_depth"]["fps"]),
        "frame_merge_ratio": float(complete["video_depth"].get("frame_merge_merge_ratio_mean", 0.0)),
    }
    archive_summaries(candidate_root, result_root, candidate["id"])
    shutil.rmtree(candidate_root, ignore_errors=True)
    return metrics


def write_ranking(result_root: Path, state: dict[str, Any]) -> None:
    rows = sorted(
        (item for item in state["results"] if item.get("status") == "complete"),
        key=lambda item: (float(item["auc3"]), float(item.get("fps", 0.0))),
        reverse=True,
    )
    if not rows:
        return
    with (result_root / "results_by_auc3.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (result_root / "best_by_auc3.json").write_text(json.dumps(rows[0], indent=2) + "\n")


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
        advance_queue(state)
        write_ranking(result_root, state)


def worker(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[1]
    result_root = args.result_root.resolve()
    while True:
        candidate = claim_candidate(result_root, args.gpu)
        if candidate is None:
            time.sleep(args.poll_seconds)
            continue
        print(f"GPU {args.gpu}: starting {candidate['id']}", flush=True)
        try:
            metrics = run_candidate(root, result_root, args.gpu, candidate)
            finish_candidate(result_root, candidate, metrics, None)
            print(f"GPU {args.gpu}: {candidate['id']} AUC@3={metrics['auc3']:.6f}", flush=True)
        except Exception as error:
            shutil.rmtree(result_root / "_temporary" / candidate["id"], ignore_errors=True)
            finish_candidate(result_root, candidate, None, repr(error))
            print(f"GPU {args.gpu}: {candidate['id']} failed: {error!r}", flush=True)


def reset_failed(result_root: Path) -> None:
    with locked_state(result_root) as state:
        if state is None:
            raise FileNotFoundError(f"No search state in {result_root}")
        failed = {item["id"] for item in state["candidates"] if item["status"] == "failed"}
        for item in state["candidates"]:
            if item["id"] in failed:
                item["status"] = "pending"
                item.pop("gpu", None)
                item.pop("started_at", None)
                item.pop("finished_at", None)
        state["results"] = [item for item in state["results"] if item.get("candidate") not in failed]
        write_ranking(result_root, state)
    print(f"Reset {len(failed)} failed candidates in {result_root}", flush=True)


def main() -> None:
    args = parse_args()
    if int(args.initialize) + int(args.worker) + int(args.reset_failed) != 1:
        raise ValueError("Pass exactly one of --initialize, --worker, or --reset-failed")
    if args.initialize:
        initialize(args.result_root.resolve())
    elif args.reset_failed:
        reset_failed(args.result_root.resolve())
    elif args.gpu is None:
        raise ValueError("--gpu is required with --worker")
    else:
        worker(args)


if __name__ == "__main__":
    main()
