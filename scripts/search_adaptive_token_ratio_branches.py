#!/usr/bin/env python3
"""Tune adaptive-fusion token ratios for selected branch combinations.

The TUM sequence-uniform final-token/original-token ratio is the stopping
criterion.  Candidates are stored only in a temporary directory.  Once a
branch reaches the target range, this script keeps the matching TUM result,
runs its 7Scenes result once, and optionally retains one earlier TUM candidate
with a strictly higher AUC@3.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


TARGET_LOW = 0.34
TARGET_HIGH = 0.38
TARGET_CENTER = (TARGET_LOW + TARGET_HIGH) / 2.0
RATIO_KEY = "adaptive_fusion_final_token_over_initial_token_ratio_sequence_mean"


@dataclass(frozen=True)
class Parameters:
    group_similarity_threshold: float
    frame_token_similarity_threshold: float
    token_keep_ratio: float


@dataclass
class Candidate:
    parameters: Parameters
    ratio: float
    auc3: float
    root: Path


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help="Run one GPU worker.")
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument(
        "--result-root",
        type=Path,
        default=root / "auc_eval_results" / "1new_token_ratio_target034_038",
    )
    parser.add_argument("--max-candidates", type=int, default=24)
    return parser.parse_args()


def in_target(value: float) -> bool:
    return TARGET_LOW <= value <= TARGET_HIGH


def candidate_key(parameters: Parameters) -> tuple[float, float, float]:
    return (
        round(parameters.group_similarity_threshold, 7),
        round(parameters.frame_token_similarity_threshold, 7),
        round(parameters.token_keep_ratio, 7),
    )


def directional_values(value: float, direction: int, values: list[float]) -> list[float]:
    """Return only values that move the effective token count in `direction`."""
    if direction < 0:
        return [candidate for candidate in values if candidate < value]
    return [candidate for candidate in values if candidate > value]


def load_tum_metrics(run_root: Path, method: str) -> tuple[float, float]:
    path = run_root / method / "tum" / "summaries" / "_summary_complete_scale_shift.json"
    with path.open() as handle:
        payload = json.load(handle)
    depth = payload["video_depth"]
    ratio = depth.get(RATIO_KEY)
    auc3 = (payload.get("pose_auc") or {}).get("AUC@3")
    if not isinstance(ratio, (int, float)) or not isinstance(auc3, (int, float)):
        raise RuntimeError(f"Missing {RATIO_KEY} or AUC@3 in {path}")
    return float(ratio), float(auc3)


def run_branch(root: Path, runner: Path, gpu: int, run_root: Path, method: str, dataset: str, params: Parameters) -> None:
    env = os.environ.copy()
    env.update(
        {
            "ROOT": str(root),
            "RUN_ROOT": str(run_root),
            "ADAPTIVE_GROUP_SIMILARITY_THRESHOLD": str(params.group_similarity_threshold),
            "ADAPTIVE_FRAME_TOKEN_SIMILARITY_THRESHOLD": str(params.frame_token_similarity_threshold),
            "ADAPTIVE_TOKEN_KEEP_RATIO": str(params.token_keep_ratio),
        }
    )
    subprocess.run(["bash", str(runner), str(gpu), dataset, method], cwd=root, env=env, check=True)


def append_candidate(
    candidates: list[Candidate],
    seen: set[tuple[float, float, float]],
    root: Path,
    runner: Path,
    gpu: int,
    method: str,
    work_root: Path,
    parameters: Parameters,
    max_candidates: int,
) -> Candidate | None:
    if len(candidates) >= max_candidates or candidate_key(parameters) in seen:
        return None
    seen.add(candidate_key(parameters))
    run_root = work_root / f"candidate_{len(candidates) + 1:02d}"
    print(f"{method}: trying {parameters}", flush=True)
    run_branch(root, runner, gpu, run_root, method, "tum", parameters)
    ratio, auc3 = load_tum_metrics(run_root, method)
    candidate = Candidate(parameters=parameters, ratio=ratio, auc3=auc3, root=run_root)
    candidates.append(candidate)
    print(f"{method}: ratio={ratio:.6f}, AUC@3={auc3:.6f}", flush=True)
    return candidate


def closest(candidates: list[Candidate]) -> Candidate:
    return min(candidates, key=lambda candidate: abs(candidate.ratio - TARGET_CENTER))


def search_method(root: Path, runner: Path, gpu: int, result_root: Path, method: str, max_candidates: int) -> bool:
    work_root = result_root / "_temporary_candidates" / method
    shutil.rmtree(work_root, ignore_errors=True)
    work_root.mkdir(parents=True, exist_ok=True)

    candidates: list[Candidate] = []
    seen: set[tuple[float, float, float]] = set()
    initial = Parameters(0.998, 0.995, 0.4)
    found = append_candidate(candidates, seen, root, runner, gpu, method, work_root, initial, max_candidates)
    if found is None:
        return False

    # First tune the most direct ratio control.  The other two controls are
    # then used only when keep-ratio alone cannot reach the target interval.
    if not in_target(found.ratio):
        direction = -1 if found.ratio > TARGET_HIGH else 1
        keep_grid = (
            [0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05]
            if direction < 0
            else [0.45, 0.50, 0.60, 0.70, 0.80]
        )
        for keep_ratio in keep_grid:
            candidate = append_candidate(
                candidates,
                seen,
                root,
                runner,
                gpu,
                method,
                work_root,
                Parameters(initial.group_similarity_threshold, initial.frame_token_similarity_threshold, keep_ratio),
                max_candidates,
            )
            if candidate is not None and in_target(candidate.ratio):
                found = candidate
                break

    if not in_target(found.ratio):
        pivot = closest(candidates)
        direction = -1 if pivot.ratio > TARGET_HIGH else 1
        group_grid = directional_values(
            pivot.parameters.group_similarity_threshold,
            direction,
            [0.980, 0.985, 0.990, 0.992, 0.995, 0.996, 0.997, 0.9975, 0.9985, 0.999, 0.9995, 0.9998],
        )
        for group_threshold in group_grid:
            candidate = append_candidate(
                candidates,
                seen,
                root,
                runner,
                gpu,
                method,
                work_root,
                Parameters(group_threshold, pivot.parameters.frame_token_similarity_threshold, pivot.parameters.token_keep_ratio),
                max_candidates,
            )
            if candidate is not None and in_target(candidate.ratio):
                found = candidate
                break

    if not in_target(found.ratio):
        pivot = closest(candidates)
        direction = -1 if pivot.ratio > TARGET_HIGH else 1
        frame_grid = directional_values(
            pivot.parameters.frame_token_similarity_threshold,
            direction,
            [0.970, 0.975, 0.980, 0.985, 0.990, 0.992, 0.993, 0.994, 0.996, 0.997, 0.998, 0.999],
        )
        for frame_threshold in frame_grid:
            candidate = append_candidate(
                candidates,
                seen,
                root,
                runner,
                gpu,
                method,
                work_root,
                Parameters(pivot.parameters.group_similarity_threshold, frame_threshold, pivot.parameters.token_keep_ratio),
                max_candidates,
            )
            if candidate is not None and in_target(candidate.ratio):
                found = candidate
                break

    if not in_target(found.ratio):
        pivot = closest(candidates)
        direction = -1 if pivot.ratio > TARGET_HIGH else 1
        group_steps = directional_values(
            pivot.parameters.group_similarity_threshold,
            direction,
            [pivot.parameters.group_similarity_threshold - 0.001, pivot.parameters.group_similarity_threshold - 0.002,
             pivot.parameters.group_similarity_threshold + 0.001, pivot.parameters.group_similarity_threshold + 0.002],
        )
        frame_steps = directional_values(
            pivot.parameters.frame_token_similarity_threshold,
            direction,
            [pivot.parameters.frame_token_similarity_threshold - 0.001, pivot.parameters.frame_token_similarity_threshold - 0.002,
             pivot.parameters.frame_token_similarity_threshold + 0.001, pivot.parameters.frame_token_similarity_threshold + 0.002],
        )
        for group_threshold, frame_threshold in zip(group_steps, frame_steps):
            candidate = append_candidate(
                candidates,
                seen,
                root,
                runner,
                gpu,
                method,
                work_root,
                Parameters(group_threshold, frame_threshold, pivot.parameters.token_keep_ratio),
                max_candidates,
            )
            if candidate is not None and in_target(candidate.ratio):
                found = candidate
                break

    final = next((candidate for candidate in candidates if in_target(candidate.ratio)), None)
    if final is None:
        print(f"{method}: no candidate reached [{TARGET_LOW}, {TARGET_HIGH}]; discarding candidates", flush=True)
        shutil.rmtree(work_root, ignore_errors=True)
        return False

    final_root = result_root / "final" / method
    shutil.rmtree(final_root, ignore_errors=True)
    final_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(final.root / method / "tum", final_root / "tum")
    print(f"{method}: selected final {final.parameters}; running 7scenes once", flush=True)
    run_branch(root, runner, gpu, result_root / "final", method, "7scenes", final.parameters)

    higher_auc_candidates = [candidate for candidate in candidates if candidate.auc3 > final.auc3]
    if higher_auc_candidates:
        best_auc = max(higher_auc_candidates, key=lambda candidate: candidate.auc3)
        best_root = result_root / "best_auc3" / method
        shutil.rmtree(best_root, ignore_errors=True)
        best_root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(best_auc.root / method / "tum", best_root / "tum")
        print(f"{method}: retained higher-AUC@3 TUM candidate {best_auc.parameters}", flush=True)

    shutil.rmtree(work_root, ignore_errors=True)
    return True


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    runner = root / "scripts" / "run_adaptive_frame_token_branch.sh"
    result_root = args.result_root.resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    succeeded = 0
    for method in args.methods:
        try:
            succeeded += int(search_method(root, runner, args.gpu, result_root, method, args.max_candidates))
        except subprocess.CalledProcessError as error:
            print(f"{method}: runner failed with exit code {error.returncode}; continuing", file=sys.stderr, flush=True)
        except Exception as error:  # Keep other independent branch jobs moving.
            print(f"{method}: {error}", file=sys.stderr, flush=True)
    print(f"GPU {args.gpu}: completed {succeeded}/{len(args.methods)} branch groups", flush=True)


if __name__ == "__main__":
    main()
