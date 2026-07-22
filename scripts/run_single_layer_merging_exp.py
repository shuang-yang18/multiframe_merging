#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get("PYTHON", "/data/mmc_syang/miniconda3/envs/fastvggt/bin/python")
GPU = os.environ.get("GPU", "5")
EXP_DIR = Path(os.environ.get("EXP_DIR", ROOT / "layerwise_results" / "exp"))
DATASET = os.environ.get("DATASET", "tum_dynamic")
CHECKPOINT = os.environ.get("CHECKPOINT", "checkpoints/vggt_omega_1b_512.pt")
MAX_FRAMES = os.environ.get("MAX_FRAMES", "300")


def load_summary(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def result_from_summary(layer: int, output_dir: Path, summary_path: Path, elapsed: float, returncode: int) -> dict:
    result = {
        "layer": layer,
        "config": f"layer{layer:02d}_r090",
        "schedule": f"{layer}-{layer}:0.9",
        "pair": 0.986,
        "span": 0.955,
        "base_r": 0.0,
        "layer_r": 0.9,
        "returncode": returncode,
        "elapsed_sec": round(elapsed, 3),
        "summary_json": str(summary_path) if summary_path.exists() else "",
    }
    if returncode != 0 or not summary_path.exists():
        return result
    summary = load_summary(summary_path)
    depth = summary.get("video_depth", summary)
    pose = summary.get("pose_auc") or {}
    speed = summary.get("speed") or summary
    result.update(
        {
            "abs_rel": depth.get("Abs Rel"),
            "delta_1.25": depth.get("delta < 1.25"),
            "auc@3": pose.get("AUC@3"),
            "auc@30": pose.get("AUC@30"),
            "fps": speed.get("fps") or depth.get("fps"),
            "frame_merge_ratio": speed.get("frame_merge_merge_ratio_mean")
            or depth.get("frame_merge_merge_ratio_mean"),
            "active_frames": speed.get("frame_merge_active_frames_mean")
            or depth.get("frame_merge_active_frames_mean"),
            "token_after_over_frame_merged": speed.get(
                "token_merging_active_over_frame_merged_token_ratio_mean"
            ),
            "token_after_over_frame_original": speed.get(
                "token_merging_active_over_frame_original_token_ratio_mean"
            ),
        }
    )
    return result


def sort_rows(rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: (
            row.get("auc@3") is not None,
            float(row.get("auc@3") or -1.0),
            float(row.get("auc@30") or -1.0),
            float(row.get("fps") or -1.0),
        ),
        reverse=True,
    )


def write_results(rows: list[dict]) -> None:
    rows = sort_rows(rows)
    fields = [
        "rank_by_auc3",
        "layer",
        "config",
        "schedule",
        "pair",
        "span",
        "base_r",
        "layer_r",
        "abs_rel",
        "delta_1.25",
        "auc@3",
        "auc@30",
        "fps",
        "frame_merge_ratio",
        "active_frames",
        "token_after_over_frame_merged",
        "token_after_over_frame_original",
        "returncode",
        "elapsed_sec",
        "summary_json",
    ]
    csv_path = EXP_DIR / "results_by_auc3.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(rows, 1):
            writer.writerow({field: (rank if field == "rank_by_auc3" else row.get(field, "")) for field in fields})

    md_path = EXP_DIR / "summary_by_auc3.md"
    with md_path.open("w") as handle:
        handle.write("# Single-layer token merging ablation\n\n")
        handle.write("Fixed params: `pair=0.986`, `span=0.955`, `max_group=4`, `restore=24`, `base_r=0.0`.\n\n")
        handle.write("| Rank | Layer | AUC@3 | AUC@30 | FPS | Abs Rel | Token/Original | Summary |\n")
        handle.write("|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for rank, row in enumerate(rows, 1):
            handle.write(
                f"| {rank} | {row.get('layer', '')} | "
                f"{float(row.get('auc@3') or 0):.6f} | "
                f"{float(row.get('auc@30') or 0):.6f} | "
                f"{float(row.get('fps') or 0):.6f} | "
                f"{float(row.get('abs_rel') or 0):.6f} | "
                f"{float(row.get('token_after_over_frame_original') or 0) * 100:.2f}% | "
                f"`{Path(str(row.get('summary_json', ''))).name}` |\n"
            )


def run_layer(layer: int) -> dict:
    name = f"tum300_single_layer{layer:02d}_r090_p0986_s0955"
    output_dir = EXP_DIR / "_tmp" / name
    log_path = EXP_DIR / "logs" / f"{name}.log"
    summary_keep = EXP_DIR / "summaries" / f"{name}_summary.json"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    summary_keep.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(output_dir, ignore_errors=True)

    cmd = [
        PYTHON,
        "inference/infer.py",
        "--dataset",
        DATASET,
        "--output-dir",
        str(output_dir),
        "--max-frames-per-seq",
        MAX_FRAMES,
        "--window-size",
        "0",
        "--checkpoint",
        CHECKPOINT,
        "--overwrite",
        "--eval",
        "--enable-token-merging",
        "--token-merging-method",
        "frame_persistent_spatial",
        "--token-merging-ratio",
        "0.0",
        "--token-merging-layer-ratios",
        f"{layer}-{layer}:0.9",
        "--token-merging-start",
        "0",
        "--token-merging-frame-pool-stride",
        "2",
        "--token-merging-frame-segment-threshold",
        "0.9",
        "--token-merging-frame-merge-threshold",
        "0.1",
        "--token-merging-frame-alpha",
        "0.1",
        "--token-merging-frame-max-window",
        "20",
        "--token-merging-frame-restore-layer",
        "24",
        "--token-merging-frame-multi-max-group-size",
        "4",
        "--token-merging-frame-multi-pair-threshold",
        "0.986",
        "--token-merging-frame-multi-span-threshold",
        "0.955",
    ]
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": GPU,
            "PYTHONPATH": str(ROOT),
            "PYTHONNOUSERSITE": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "HF_HOME": env.get("HF_HOME", str(ROOT / ".cache" / "huggingface")),
            "TRANSFORMERS_CACHE": env.get("TRANSFORMERS_CACHE", str(ROOT / ".cache" / "huggingface" / "hub")),
        }
    )
    start = time.time()
    with log_path.open("w") as log_file:
        log_file.write("cmd=" + " ".join(cmd) + "\n")
        log_file.flush()
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - start

    summary_dir = output_dir / DATASET
    summary_path = summary_dir / "_summary_complete_scale_shift.json"
    if not summary_path.exists():
        summary_path = summary_dir / "_summary_scale_shift.json"
    if summary_path.exists():
        shutil.copy2(summary_path, summary_keep)
    shutil.rmtree(output_dir, ignore_errors=True)
    if proc.returncode == 0:
        log_path.unlink(missing_ok=True)
    return result_from_summary(layer, output_dir, summary_keep, elapsed, proc.returncode)


def main() -> int:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for layer in range(1, 25):
        print(f"start layer {layer}", flush=True)
        row = run_layer(layer)
        rows.append(row)
        write_results(rows)
        print(
            f"done layer {layer} rc={row.get('returncode')} "
            f"auc3={row.get('auc@3')} auc30={row.get('auc@30')} fps={row.get('fps')}",
            flush=True,
        )
    write_results(rows)
    print(f"wrote {EXP_DIR / 'results_by_auc3.csv'}", flush=True)
    print(f"wrote {EXP_DIR / 'summary_by_auc3.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
