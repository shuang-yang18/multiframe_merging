#!/usr/bin/env python3
"""Collect the TUM300 anchor-hybrid restore-layer comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


RUNS = (
    ("VGGT-Omega baseline (revalidated)", "tum300_omega_baseline_revalidate_20260723"),
    ("Frame-anchor hybrid, restore=20", "tum300_frame_anchor_hybrid_k0_restore20_anchor4_uniform_20260723"),
    ("Frame-anchor hybrid, restore=24 (final restore)", "tum300_frame_anchor_hybrid_k0_restore24_anchor4_uniform_20260723"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, float | int | str | None]] = []
    for method, directory in RUNS:
        path = args.run_root / directory / "tum_dynamic" / "_summary_complete_scale_shift.json"
        with path.open() as handle:
            summary = json.load(handle)
        depth = summary["video_depth"]
        pose = summary["pose_auc"]
        speed = summary["speed"]
        rows.append(
            {
                "method": method,
                "abs_rel": depth["Abs Rel"],
                "delta_125": depth["delta < 1.25"],
                "auc_3": pose["AUC@3"],
                "auc_30": pose["AUC@30"],
                "fps": speed["fps"],
                "frame_merge_active_frames_mean": speed.get("frame_merge_active_frames_mean"),
                "frame_merge_merge_ratio_mean": speed.get("frame_merge_merge_ratio_mean"),
                "frame_merge_anchor_count": speed.get("frame_merge_anchor_count"),
            }
        )

    baseline_fps = float(rows[0]["fps"])
    for row in rows:
        row["speedup"] = float(row["fps"]) / baseline_fps

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    markdown_path = args.output.with_suffix(".md")
    lines = [
        "| Method | Abs Rel | delta<1.25 | AUC@3 | AUC@30 | FPS | Speedup | Active frames | Frame merge | Anchors |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        active_frames = row["frame_merge_active_frames_mean"]
        merge_ratio = row["frame_merge_merge_ratio_mean"]
        anchor_count = row["frame_merge_anchor_count"]
        lines.append(
            "| {method} | {abs_rel:.6f} | {delta_125:.6f} | {auc_3:.6f} | {auc_30:.6f} | {fps:.6f} | "
            "{speedup:.4f} | {active_frames} | {merge_ratio} | {anchor_count} |".format(
                **row,
                active_frames="-" if active_frames is None else f"{float(active_frames):.2f}",
                merge_ratio="-" if merge_ratio is None else f"{float(merge_ratio):.2%}",
                anchor_count="-" if anchor_count is None else str(anchor_count),
            )
        )
    markdown_path.write_text("\n".join(lines) + "\n")
    print(markdown_path)


if __name__ == "__main__":
    main()
