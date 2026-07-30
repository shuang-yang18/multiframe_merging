#!/usr/bin/env python3
"""Increase a pairwise branch's frame retention while lowering token retention.

Only candidates whose frame-fused token ratio exceeds the original branch
baseline and whose final-token/original-token ratio stays in [0.34, 0.38]
are retained.  All other candidate summaries are removed after inspection.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


TARGET_LOW = 0.34
TARGET_HIGH = 0.38


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--method", default="pairwise_01")
    parser.add_argument("--baseline-frame-token-ratio", required=True, type=float)
    parser.add_argument("--group-threshold", required=True, type=float)
    parser.add_argument("--keep-ratios", nargs="+", required=True, type=float)
    parser.add_argument(
        "--result-root",
        type=Path,
        default=root / "auc_eval_results" / "1new_pairwise01_frame_up_token_down",
    )
    return parser.parse_args()


def case_name(group_threshold: float, keep_ratio: float) -> str:
    group = f"{group_threshold:.4f}".replace(".", "")
    keep = f"{keep_ratio:.2f}".replace(".", "")
    return f"g{group}_keep{keep}"


def metrics(path: Path) -> tuple[float, float, float, float, float]:
    with path.open() as handle:
        payload = json.load(handle)
    depth = payload["video_depth"]
    pose = payload.get("pose_auc") or {}
    return (
        float(depth["adaptive_fusion_frame_token_ratio_sequence_mean"]),
        float(depth["adaptive_fusion_final_token_over_initial_token_ratio_sequence_mean"]),
        float(pose["AUC@3"]),
        float(pose["AUC@30"]),
        float(depth["fps"]),
    )


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    runner = root / "scripts" / "run_adaptive_frame_token_branch.sh"
    result_root = args.result_root.resolve()
    temporary_root = result_root / "_temporary" / f"gpu{args.gpu}"
    qualified_root = result_root / "qualified"
    temporary_root.mkdir(parents=True, exist_ok=True)
    qualified_root.mkdir(parents=True, exist_ok=True)

    for keep_ratio in args.keep_ratios:
        label = case_name(args.group_threshold, keep_ratio)
        candidate_root = temporary_root / label
        shutil.rmtree(candidate_root, ignore_errors=True)
        env = os.environ.copy()
        env.update(
            {
                "ROOT": str(root),
                "RUN_ROOT": str(candidate_root),
                "ADAPTIVE_GROUP_SIMILARITY_THRESHOLD": str(args.group_threshold),
                "ADAPTIVE_GROUP_MAX_SIZE": "3",
                "ADAPTIVE_FRAME_TOKEN_SIMILARITY_THRESHOLD": "0.995",
                "ADAPTIVE_TOKEN_KEEP_RATIO": str(keep_ratio),
            }
        )
        print(f"{label}: start", flush=True)
        subprocess.run(["bash", str(runner), str(args.gpu), "tum", args.method], cwd=root, env=env, check=True)
        summary_path = candidate_root / args.method / "tum" / "summaries" / "_summary_complete_scale_shift.json"
        frame_ratio, final_ratio, auc3, auc30, fps = metrics(summary_path)
        qualifying = frame_ratio > args.baseline_frame_token_ratio and TARGET_LOW <= final_ratio <= TARGET_HIGH
        print(
            f"{label}: frame_ratio={frame_ratio:.6f}, final_ratio={final_ratio:.6f}, "
            f"AUC@3={auc3:.6f}, AUC@30={auc30:.6f}, FPS={fps:.6f}, qualifying={qualifying}",
            flush=True,
        )
        if qualifying:
            destination = qualified_root / label
            shutil.rmtree(destination, ignore_errors=True)
            shutil.copytree(candidate_root / args.method / "tum", destination / "tum")
        shutil.rmtree(candidate_root, ignore_errors=True)


if __name__ == "__main__":
    main()
