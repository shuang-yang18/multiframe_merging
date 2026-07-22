#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path


RUN_ROOT = Path("outputs/bonn_tum300_accel_20260721")
ROWS = [
    ("bonn", "vggt-Ω", RUN_ROOT / "bonn300_omega" / "bonn" / "_summary_complete_scale_shift.json"),
    (
        "bonn",
        "vggt-Ω+fastvggt(r=0.9)",
        RUN_ROOT / "bonn300_fastvggt_r090" / "bonn" / "_summary_complete_scale_shift.json",
    ),
    (
        "bonn",
        "vggt-Ω+sparse-vggt",
        RUN_ROOT / "bonn300_sparse_vggt" / "bonn" / "_summary_complete_scale_shift.json",
    ),
    ("bonn", "vggt-Ω+DA-vggt", RUN_ROOT / "bonn300_da_vggt" / "bonn" / "_summary_complete_scale_shift.json"),
    (
        "tum_dynamic",
        "vggt-Ω",
        RUN_ROOT / "tum300_omega" / "tum_dynamic" / "_summary_complete_scale_shift.json",
    ),
    (
        "tum_dynamic",
        "vggt-Ω+fastvggt(r=0.9)",
        RUN_ROOT / "tum300_fastvggt_r090" / "tum_dynamic" / "_summary_complete_scale_shift.json",
    ),
    (
        "tum_dynamic",
        "vggt-Ω+sparse-vggt",
        RUN_ROOT / "tum300_sparse_vggt" / "tum_dynamic" / "_summary_complete_scale_shift.json",
    ),
    (
        "tum_dynamic",
        "vggt-Ω+DA-vggt",
        RUN_ROOT / "tum300_da_vggt" / "tum_dynamic" / "_summary_complete_scale_shift.json",
    ),
]


def resolve_summary(path: Path) -> Path:
    if path.is_file():
        return path
    dataset = path.parent.name
    alt = path.parent / f"{dataset}-complete-scale_shift.json"
    if alt.is_file():
        return alt
    return path


def pick(payload: dict) -> dict:
    depth = payload.get("video_depth") or {}
    pose = payload.get("pose_auc") or {}
    speed = payload.get("speed") or {}
    return {
        "abs_rel": depth.get("Abs Rel"),
        "delta<1.25": depth.get("delta < 1.25"),
        "auc@3": pose.get("AUC@3"),
        "auc@30": pose.get("AUC@30"),
        "fps": speed.get("fps") or depth.get("fps"),
    }


def main() -> None:
    rows = []
    missing = []
    for dataset, method, path in ROWS:
        path = resolve_summary(path)
        if not path.is_file():
            missing.append(str(path))
            continue
        with path.open() as handle:
            metrics = pick(json.load(handle))
        rows.append({"dataset": dataset, "method": method, **metrics, "summary_json": str(path)})

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = RUN_ROOT / "key_metrics.csv"
    md_path = RUN_ROOT / "key_metrics.md"
    fields = ["dataset", "method", "abs_rel", "delta<1.25", "auc@3", "auc@30", "fps", "summary_json"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    with md_path.open("w") as handle:
        handle.write("| dataset | method | abs rel | δ<1.25 | auc@3 | auc@30 | fps |\n")
        handle.write("|---|---|---:|---:|---:|---:|---:|\n")
        for row in rows:
            values = [row.get(key) for key in ("abs_rel", "delta<1.25", "auc@3", "auc@30", "fps")]
            formatted = ["" if value is None else f"{float(value):.6f}" for value in values]
            handle.write(f"| {row['dataset']} | {row['method']} | " + " | ".join(formatted) + " |\n")
        if missing:
            handle.write("\nMissing summaries:\n")
            for path in missing:
                handle.write(f"- `{path}`\n")

    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    if missing:
        print("missing:")
        for path in missing:
            print(path)


if __name__ == "__main__":
    main()
