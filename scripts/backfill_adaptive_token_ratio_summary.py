#!/usr/bin/env python3
"""Add the sequence-uniform adaptive token ratio to existing summaries."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


RATIO_KEY = "adaptive_fusion_token_over_pre_frame_token_ratio_mean"
SUMMARY_MEAN_KEY = "adaptive_fusion_final_token_over_initial_token_ratio_sequence_mean"
SUMMARY_COUNT_KEY = "adaptive_fusion_final_token_over_initial_token_ratio_sequence_count"
SUMMARY_FILENAMES = {"_summary_scale_shift.json", "_summary_complete_scale_shift.json"}


def sequence_ratios(summary_dir: Path) -> list[float]:
    values: list[float] = []
    for timing_path in sorted((summary_dir / "sequences").rglob("_time.json")):
        with timing_path.open() as handle:
            value = json.load(handle).get(RATIO_KEY)
        if isinstance(value, (int, float)) and math.isfinite(value):
            values.append(float(value))
    return values


def update_summary(path: Path, mean: float, count: int) -> None:
    with path.open() as handle:
        payload = json.load(handle)

    target = payload.get("video_depth") if isinstance(payload.get("video_depth"), dict) else payload
    target[SUMMARY_MEAN_KEY] = mean
    target[SUMMARY_COUNT_KEY] = count
    if isinstance(payload.get("speed"), dict):
        payload["speed"][SUMMARY_MEAN_KEY] = mean
        payload["speed"][SUMMARY_COUNT_KEY] = count

    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path, help="Experiment roots containing summaries directories")
    args = parser.parse_args()

    updated = 0
    skipped = 0
    for root in args.roots:
        for summary_path in sorted(root.rglob("_summary*_scale_shift.json")):
            if summary_path.name not in SUMMARY_FILENAMES:
                continue
            values = sequence_ratios(summary_path.parent)
            if not values:
                skipped += 1
                continue
            mean = sum(values) / len(values)
            update_summary(summary_path, mean, len(values))
            updated += 1
            print(f"updated {summary_path}: mean={mean:.9f}, sequences={len(values)}")
    print(f"updated={updated}, skipped_without_sequence_ratios={skipped}")


if __name__ == "__main__":
    main()
