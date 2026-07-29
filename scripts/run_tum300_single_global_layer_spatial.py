#!/usr/bin/env python3
"""Run TUM300 baseline plus one-global-layer spatial token-merging ablations.

Each accelerated case applies FastVGGT-style spatial merging only in one
global inter-frame attention block. Register-only blocks and frame merging are
never enabled. Temporary reconstruction artifacts are removed after their
evaluation summaries have been copied to the experiment directory.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get("PYTHON", sys.executable)
GPU = os.environ.get("GPU", "7")
EXP_DIR = Path(os.environ.get("EXP_DIR", ROOT / "new_results" / "2" / "tum300_single_global_layer_spatial_r090"))
CHECKPOINT = os.environ.get("CHECKPOINT", "checkpoints/vggt_omega_1b_512.pt")
MAX_FRAMES = os.environ.get("MAX_FRAMES", "300")
MERGE_RATIO = float(os.environ.get("MERGE_RATIO", "0.9"))

# Aggregator blocks 2, 6, 9, 14 and 20 are register-only inter-frame layers.
GLOBAL_BLOCKS = (0, 1, 3, 4, 5, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18, 19, 21, 22, 23)


def summary_path(output_dir: Path) -> Path:
    dataset_dir = output_dir / "tum_dynamic"
    for name in ("_summary_complete_scale_shift.json", "_summary_scale_shift.json"):
        path = dataset_dir / name
        if path.is_file():
            return path
    return dataset_dir / "_summary_complete_scale_shift.json"


def metric_row(name: str, block: int | None, output_dir: Path, elapsed: float, returncode: int) -> dict[str, object]:
    path = summary_path(output_dir)
    row: dict[str, object] = {
        "method": name,
        "global_block": "baseline" if block is None else block,
        "schedule": "none" if block is None else f"global_block_{block}:r={MERGE_RATIO:g}",
        "merge_ratio": 0.0 if block is None else MERGE_RATIO,
        "returncode": returncode,
        "elapsed_sec": round(elapsed, 3),
        "summary_json": str(path) if path.exists() else "",
    }
    if returncode or not path.exists():
        return row
    summary = json.loads(path.read_text())
    depth = summary.get("video_depth", summary)
    pose = summary.get("pose_auc") or {}
    speed = summary.get("speed") or summary
    row.update(
        {
            "abs_rel": depth.get("Abs Rel"),
            "delta_1.25": depth.get("delta < 1.25"),
            "auc@3": pose.get("AUC@3"),
            "auc@30": pose.get("AUC@30"),
            "fps": speed.get("fps") or depth.get("fps"),
            "token_after_ratio": speed.get("token_merging_active_over_frame_original_token_ratio_mean"),
        }
    )
    return row


def write_tables(rows: list[dict[str, object]]) -> None:
    def ranking_key(row: dict[str, object]) -> tuple[bool, float, float, float]:
        return (
            row.get("auc@3") is not None,
            float(row.get("auc@3") or -1.0),
            float(row.get("auc@30") or -1.0),
            float(row.get("fps") or -1.0),
        )

    ranked = sorted(rows, key=ranking_key, reverse=True)
    fields = [
        "rank_by_auc3", "method", "global_block", "schedule", "merge_ratio",
        "abs_rel", "delta_1.25", "auc@3", "auc@30", "fps", "token_after_ratio",
        "returncode", "elapsed_sec", "summary_json",
    ]
    with (EXP_DIR / "results_by_auc3.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(ranked, 1):
            writer.writerow({key: (rank if key == "rank_by_auc3" else row.get(key, "")) for key in fields})

    with (EXP_DIR / "summary_by_auc3.md").open("w") as handle:
        handle.write("# TUM300 single-global-layer spatial token-merging ablation\n\n")
        handle.write(f"Baseline: no token merging. Each other row: spatial merging at exactly one global block with `r={MERGE_RATIO:g}`.\n\n")
        handle.write("| Rank | Method | Global block | AUC@3 | AUC@30 | FPS | Abs Rel | Token / original |\n")
        handle.write("|---:|---|---:|---:|---:|---:|---:|---:|\n")
        for rank, row in enumerate(ranked, 1):
            ratio = row.get("token_after_ratio")
            ratio_text = "-" if ratio is None else f"{float(ratio) * 100:.2f}%"
            handle.write(
                f"| {rank} | {row.get('method')} | {row.get('global_block')} | "
                f"{float(row.get('auc@3') or 0):.6f} | {float(row.get('auc@30') or 0):.6f} | "
                f"{float(row.get('fps') or 0):.6f} | {float(row.get('abs_rel') or 0):.6f} | {ratio_text} |\n"
            )


def run_case(name: str, block: int | None) -> dict[str, object]:
    output_dir = EXP_DIR / "_temporary" / name
    log_path = EXP_DIR / "logs" / f"{name}.log"
    keep_path = EXP_DIR / "summaries" / f"{name}.json"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    keep_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(output_dir, ignore_errors=True)

    command = [
        PYTHON, "inference/infer.py", "--dataset", "tum_dynamic", "--output-dir", str(output_dir),
        "--max-frames-per-seq", MAX_FRAMES, "--window-size", "0", "--checkpoint", CHECKPOINT,
        "--overwrite", "--eval",
    ]
    if block is not None:
        command.extend([
            "--enable-token-merging", "--token-merging-method", "spatial",
            "--token-merging-ratio", str(MERGE_RATIO), "--token-merging-start", "0",
            "--token-merging-layer-ratios", f"{block + 1}-{block + 1}:{MERGE_RATIO:g}",
        ])

    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": GPU,
        "PYTHONPATH": str(ROOT),
        "PYTHONNOUSERSITE": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "HF_HOME": env.get("HF_HOME", str(ROOT / ".cache" / "huggingface")),
        "TRANSFORMERS_CACHE": env.get("TRANSFORMERS_CACHE", str(ROOT / ".cache" / "huggingface" / "hub")),
    })
    start = time.time()
    with log_path.open("w") as log_file:
        log_file.write("cmd=" + " ".join(command) + "\n")
        log_file.flush()
        proc = subprocess.run(command, cwd=ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - start

    source = summary_path(output_dir)
    if source.exists():
        shutil.copy2(source, keep_path)
    dataset_dir = output_dir / "tum_dynamic"
    per_sequence_dir = EXP_DIR / "per_sequence" / name
    per_sequence_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("_sequence_metrics_scale_shift.csv", "_summary_pose_auc.json"):
        artifact = dataset_dir / filename
        if artifact.exists():
            shutil.copy2(artifact, per_sequence_dir / filename)
    row = metric_row(name, block, output_dir, elapsed, proc.returncode)
    if source.exists():
        row["summary_json"] = str(keep_path)
    shutil.rmtree(output_dir, ignore_errors=True)
    return row


def main() -> int:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    cases = [("tum300_baseline", None)] + [(f"tum300_spatial_global_block{block:02d}_r{int(MERGE_RATIO * 100):02d}", block) for block in GLOBAL_BLOCKS]
    for index, (name, block) in enumerate(cases, 1):
        print(f"[{index}/{len(cases)}] start {name}", flush=True)
        row = run_case(name, block)
        rows.append(row)
        write_tables(rows)
        print(f"[{index}/{len(cases)}] done {name} rc={row['returncode']} auc3={row.get('auc@3')} fps={row.get('fps')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
