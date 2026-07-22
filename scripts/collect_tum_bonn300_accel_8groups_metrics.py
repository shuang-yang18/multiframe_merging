#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROWS = [
    ("Bonn300", "VGGT-Omega baseline", "bonn300_baseline", "bonn"),
    ("Bonn300", "VGGT-Omega + FastVGGT r=0.9", "bonn300_fastvggt_r090", "bonn"),
    ("Bonn300", "VGGT-Omega + DA-VGGT", "bonn300_da_vggt", "bonn"),
    ("Bonn300", "VGGT-Omega + Sparse-VGGT", "bonn300_sparse_vggt", "bonn"),
    ("TUM300", "VGGT-Omega baseline", "tum300_baseline", "tum_dynamic"),
    ("TUM300", "VGGT-Omega + FastVGGT r=0.9", "tum300_fastvggt_r090", "tum_dynamic"),
    ("TUM300", "VGGT-Omega + DA-VGGT", "tum300_da_vggt", "tum_dynamic"),
    ("TUM300", "VGGT-Omega + Sparse-VGGT", "tum300_sparse_vggt", "tum_dynamic"),
]


def resolve_summary(run_root: Path, output_name: str, dataset: str) -> Path:
    primary = run_root / output_name / dataset / "_summary_complete_scale_shift.json"
    if primary.is_file():
        return primary
    alt = run_root / output_name / dataset / f"{dataset}-complete-scale_shift.json"
    if alt.is_file():
        return alt
    return primary


def pick(payload: dict) -> dict:
    depth = payload.get("video_depth") or {}
    pose = payload.get("pose_auc") or {}
    speed = payload.get("speed") or {}
    return {
        "Abs Rel": depth.get("Abs Rel"),
        "delta<1.25": depth.get("delta < 1.25"),
        "AUC@3": pose.get("AUC@3"),
        "AUC@30": pose.get("AUC@30"),
        "FPS": speed.get("fps") or depth.get("fps"),
        "pose_eval_frames": pose.get("pose_eval_frames"),
        "pose_eval_seed": pose.get("pose_eval_seed"),
    }


def fmt(value) -> str:
    return "" if value is None else f"{float(value):.6f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default="outputs/tum_bonn300_accel_8groups_20260722")
    args = parser.parse_args()
    run_root = Path(args.run_root)

    rows = []
    missing = []
    baseline_fps: dict[str, float] = {}
    for dataset_name, method, output_name, dataset_dir in ROWS:
        path = resolve_summary(run_root, output_name, dataset_dir)
        if not path.is_file():
            missing.append(str(path))
            continue
        metrics = pick(json.loads(path.read_text()))
        fps = metrics.get("FPS")
        if method == "VGGT-Omega baseline" and fps is not None:
            baseline_fps[dataset_name] = float(fps)
        rows.append(
            {
                "Dataset": dataset_name,
                "Method": method,
                **metrics,
                "summary_json": str(path),
            }
        )

    for row in rows:
        base = baseline_fps.get(row["Dataset"])
        fps = row.get("FPS")
        row["Speedup"] = (float(fps) / base) if base and fps is not None else None

    fields = [
        "Dataset",
        "Method",
        "Abs Rel",
        "delta<1.25",
        "AUC@3",
        "AUC@30",
        "FPS",
        "Speedup",
        "pose_eval_frames",
        "pose_eval_seed",
        "summary_json",
    ]
    run_root.mkdir(parents=True, exist_ok=True)
    csv_path = run_root / "key_metrics.csv"
    md_path = run_root / "key_metrics.md"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    with md_path.open("w") as handle:
        handle.write("| Dataset | Method | Abs Rel | delta<1.25 | AUC@3 | AUC@30 | FPS | Speedup | Pose Frames | Seed |\n")
        handle.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| {row['Dataset']} | {row['Method']} | {fmt(row['Abs Rel'])} | "
                f"{fmt(row['delta<1.25'])} | {fmt(row['AUC@3'])} | {fmt(row['AUC@30'])} | "
                f"{fmt(row['FPS'])} | {fmt(row['Speedup'])} | "
                f"{row.get('pose_eval_frames') or ''} | {row.get('pose_eval_seed') or ''} |\n"
            )
        if missing:
            handle.write("\nMissing summaries:\n")
            for path in missing:
                handle.write(f"- `{path}`\n")

    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    if missing:
        print("missing summaries:")
        for path in missing:
            print(path)


if __name__ == "__main__":
    main()
