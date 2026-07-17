#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: export_frame_merge_groups.py <dataset_output_dir> <output_prefix>", file=sys.stderr)
        return 2

    dataset_dir = Path(sys.argv[1])
    output_prefix = Path(sys.argv[2])
    rows: list[dict[str, object]] = []
    payload: dict[str, object] = {"dataset_dir": str(dataset_dir), "sequences": []}

    for time_path in sorted(dataset_dir.glob("*/_time.json")):
        sequence = time_path.parent.name
        with time_path.open() as handle:
            timing = json.load(handle)
        sequence_item: dict[str, object] = {
            "sequence": sequence,
            "frames": timing.get("frames"),
            "fps": timing.get("fps"),
            "events": [],
        }
        for event_idx, stat in enumerate(timing.get("frame_merge_stats", [])):
            batch_groups = stat.get("merge_groups", [])
            frame_to_active = stat.get("frame_to_active", [])
            event_item = {
                "event": event_idx,
                "block": stat.get("block"),
                "mode": stat.get("mode"),
                "strategy": stat.get("frame_group_strategy"),
                "original_frames": stat.get("original_frames"),
                "active_frames_mean": stat.get("active_frames_mean"),
                "merge_ratio_mean": stat.get("merge_ratio_mean"),
                "merge_groups": batch_groups,
                "frame_to_active": frame_to_active,
            }
            sequence_item["events"].append(event_item)
            for batch_idx, groups in enumerate(batch_groups):
                for group_idx, group in enumerate(groups):
                    rows.append(
                        {
                            "sequence": sequence,
                            "event": event_idx,
                            "block": stat.get("block"),
                            "mode": stat.get("mode"),
                            "strategy": stat.get("frame_group_strategy"),
                            "batch": batch_idx,
                            "group": group_idx,
                            "size": len(group),
                            "frames": " ".join(str(frame_idx) for frame_idx in group),
                        }
                    )
        payload["sequences"].append(sequence_item)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")
    with json_path.open("w") as handle:
        json.dump(payload, handle, indent=2)
    with csv_path.open("w", newline="") as handle:
        fieldnames = ["sequence", "event", "block", "mode", "strategy", "batch", "group", "size", "frames"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"groups={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
