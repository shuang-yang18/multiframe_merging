#!/usr/bin/env python3
"""Collect one-sequence baseline and single inter-frame-block skip results."""

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
        block = None if case_dir.name == "baseline" else int(case_dir.name.removeprefix("skip_block_"))
        rows.append(
            {
                "case": case_dir.name,
                "skipped_inter_frame_block_0based": "" if block is None else block,
                "Abs Rel": depth["Abs Rel"],
                "delta < 1.25": depth["delta < 1.25"],
                "AUC@3": pose["AUC@3"],
                "AUC@30": pose["AUC@30"],
                "fps": speed["fps"],
            }
        )

    fields = ["case", "skipped_inter_frame_block_0based", "Abs Rel", "delta < 1.25", "AUC@3", "AUC@30", "fps"]
    with (args.root / "results_all.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (args.root / "results_by_auc3.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: float(row["AUC@3"]), reverse=True))


if __name__ == "__main__":
    main()
