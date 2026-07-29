#!/usr/bin/env python3
"""Collect named inter-frame-attention skip experiment results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for case_dir in sorted(path for path in args.root.iterdir() if path.is_dir()):
        summary_path = case_dir / "summaries" / "_summary_complete_scale_shift.json"
        if not summary_path.is_file():
            continue
        with summary_path.open() as handle:
            summary = json.load(handle)
        depth = summary["video_depth"]
        pose = summary["pose_auc"]
        speed = summary["speed"]
        rows.append(
            {
                "case": case_dir.name,
                "skipped_inter_frame_blocks_0based": depth.get("skip_inter_frame_attention_blocks", ""),
                "Abs Rel": depth["Abs Rel"],
                "delta < 1.25": depth["delta < 1.25"],
                "AUC@3": pose["AUC@3"],
                "AUC@30": pose["AUC@30"],
                "fps": speed["fps"],
            }
        )
    fields = ["case", "skipped_inter_frame_blocks_0based", "Abs Rel", "delta < 1.25", "AUC@3", "AUC@30", "fps"]
    for name, ordered_rows in {
        "results_all.csv": rows,
        "results_by_auc3.csv": sorted(rows, key=lambda row: float(row["AUC@3"]), reverse=True),
    }.items():
        with (args.root / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(ordered_rows)


if __name__ == "__main__":
    main()
