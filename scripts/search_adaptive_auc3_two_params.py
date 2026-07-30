#!/usr/bin/env python3
"""Grid-search adaptive frame grouping and token retention by mean AUC@3.

Each candidate is evaluated on the fixed two-sequence TUM and 7Scenes-test
subsets. Dense reconstruction output is discarded by the branch runner; this
script retains only the copied dataset summaries and per-sequence pose/time
JSON files for every candidate.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


GROUP_THRESHOLDS = (0.995, 0.997, 0.998, 0.999)
TOKEN_KEEP_RATIOS = (0.4, 0.6, 0.8, 0.9)


@dataclass(frozen=True)
class Metrics:
    abs_rel: float
    delta125: float
    auc3: float
    auc30: float
    fps: float
    final_token_ratio: float | None


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument(
        "--result-root",
        type=Path,
        default=root / "auc_eval_results" / "1new_auc3_two_param_search",
    )
    return parser.parse_args()


def case_name(group_threshold: float, token_keep_ratio: float) -> str:
    group = f"{group_threshold:.3f}".replace(".", "")
    keep = f"{token_keep_ratio:.1f}".replace(".", "")
    return f"g{group}_keep{keep}"


def load_metrics(path: Path) -> Metrics:
    with path.open() as handle:
        payload = json.load(handle)
    depth = payload["video_depth"]
    pose = payload.get("pose_auc") or {}
    return Metrics(
        abs_rel=float(depth["Abs Rel"]),
        delta125=float(depth["delta < 1.25"]),
        auc3=float(pose["AUC@3"]),
        auc30=float(pose["AUC@30"]),
        fps=float(depth["fps"]),
        final_token_ratio=(
            float(depth["adaptive_fusion_final_token_over_initial_token_ratio_sequence_mean"])
            if isinstance(depth.get("adaptive_fusion_final_token_over_initial_token_ratio_sequence_mean"), (int, float))
            else None
        ),
    )


def run_dataset(
    *,
    root: Path,
    runner: Path,
    matrix: Path,
    gpu: int,
    method: str,
    dataset: str,
    candidate_root: Path,
    group_threshold: float,
    token_keep_ratio: float,
) -> Metrics:
    env = os.environ.copy()
    env.update(
        {
            "ROOT": str(root),
            "RUN_ROOT": str(candidate_root),
            "MATRIX_FILE": str(matrix),
            "ADAPTIVE_GROUP_SIMILARITY_THRESHOLD": str(group_threshold),
            "ADAPTIVE_GROUP_MAX_SIZE": "3",
            "ADAPTIVE_FRAME_TOKEN_SIMILARITY_THRESHOLD": "0.995",
            "ADAPTIVE_TOKEN_KEEP_RATIO": str(token_keep_ratio),
        }
    )
    subprocess.run(
        ["bash", str(runner), str(gpu), dataset, method],
        cwd=root,
        env=env,
        check=True,
    )
    dataset_name = "tum_dynamic" if dataset == "tum" else "7scenes"
    summary = candidate_root / method / dataset / "summaries" / "_summary_complete_scale_shift.json"
    if not summary.is_file():
        raise RuntimeError(f"Expected summary is missing: {summary}")
    metrics = load_metrics(summary)
    saved_summary_root = candidate_root / method / dataset / "summaries"
    destination = candidate_root.parent.parent / "summaries" / candidate_root.name / dataset
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(saved_summary_root, destination, dirs_exist_ok=True)
    return metrics


def row(method: str, group_threshold: float, token_keep_ratio: float, tum: Metrics, scenes: Metrics) -> dict[str, object]:
    return {
        "method": method,
        "group_similarity_threshold": group_threshold,
        "token_keep_ratio": token_keep_ratio,
        "mean_auc3": (tum.auc3 + scenes.auc3) / 2.0,
        "tum_abs_rel": tum.abs_rel,
        "tum_delta125": tum.delta125,
        "tum_auc3": tum.auc3,
        "tum_auc30": tum.auc30,
        "tum_fps": tum.fps,
        "tum_final_token_ratio": tum.final_token_ratio,
        "7scenes_abs_rel": scenes.abs_rel,
        "7scenes_delta125": scenes.delta125,
        "7scenes_auc3": scenes.auc3,
        "7scenes_auc30": scenes.auc30,
        "7scenes_fps": scenes.fps,
        "7scenes_final_token_ratio": scenes.final_token_ratio,
    }


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    rows.sort(key=lambda item: float(item["mean_auc3"]), reverse=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    runner = root / "scripts" / "run_adaptive_frame_token_branch.sh"
    matrix = root / "scripts" / "adaptive_frame_token_auc3_matrix.tsv"
    result_root = args.result_root.resolve() / args.method
    temporary_root = result_root / "_temporary"
    summary_root = result_root / "summaries"
    result_root.mkdir(parents=True, exist_ok=True)
    summary_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for group_threshold in GROUP_THRESHOLDS:
        for token_keep_ratio in TOKEN_KEEP_RATIOS:
            label = case_name(group_threshold, token_keep_ratio)
            candidate_root = temporary_root / label
            shutil.rmtree(candidate_root, ignore_errors=True)
            print(f"{args.method} {label}: starting", flush=True)
            try:
                tum = run_dataset(
                    root=root, runner=runner, matrix=matrix, gpu=args.gpu,
                    method=args.method, dataset="tum", candidate_root=candidate_root,
                    group_threshold=group_threshold, token_keep_ratio=token_keep_ratio,
                )
                scenes = run_dataset(
                    root=root, runner=runner, matrix=matrix, gpu=args.gpu,
                    method=args.method, dataset="7scenes", candidate_root=candidate_root,
                    group_threshold=group_threshold, token_keep_ratio=token_keep_ratio,
                )
                candidate = row(args.method, group_threshold, token_keep_ratio, tum, scenes)
                rows.append(candidate)
                write_rows(result_root / "results_by_mean_auc3.csv", rows)
                print(
                    f"{args.method} {label}: mean AUC@3={candidate['mean_auc3']:.6f}; "
                    f"TUM={tum.auc3:.6f}, 7Scenes={scenes.auc3:.6f}",
                    flush=True,
                )
            except subprocess.CalledProcessError as error:
                print(f"{args.method} {label}: runner failed ({error.returncode}); continuing", flush=True)
            except Exception as error:
                print(f"{args.method} {label}: {error}; continuing", flush=True)
            finally:
                shutil.rmtree(candidate_root, ignore_errors=True)

    if rows:
        write_rows(result_root / "results_by_mean_auc3.csv", rows)
        best = max(rows, key=lambda item: float(item["mean_auc3"]))
        with (result_root / "best_by_mean_auc3.json").open("w") as handle:
            json.dump(best, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
