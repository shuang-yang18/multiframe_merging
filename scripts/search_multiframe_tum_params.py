#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    name: str
    pair: float
    span: float
    pool_stride: int = 2
    restore_layer: int = 24
    max_window: int = 20
    segment: float = 0.9
    alpha: float = 0.1
    merge: float = 0.1
    max_group: int = 4


CONFIGS = [
    Config("p0980_s0950", 0.980, 0.950),
    Config("p0985_s0950", 0.985, 0.950),
    Config("p0985_s0960", 0.985, 0.960),
    Config("p0990_s0960", 0.990, 0.960),
    Config("p0990_s0970", 0.990, 0.970),
    Config("p0990_s0980", 0.990, 0.980),
    Config("p0992_s0970", 0.992, 0.970),
    Config("p0992_s0980", 0.992, 0.980),
    Config("p0995_s0980", 0.995, 0.980),
    Config("p0995_s0990", 0.995, 0.990),
    Config("p0998_s0990", 0.998, 0.990),
    Config("p0998_s0995", 0.998, 0.995),
]


def parse_gpus() -> list[str]:
    return [item.strip() for item in os.environ.get("GPUS", "4,5,6,7").split(",") if item.strip()]


def run_config(root: Path, python: str, checkpoint: str, search_dir: Path, gpu: str, cfg: Config) -> dict[str, object]:
    output_dir = search_dir / cfg.name
    log_path = search_dir / f"{cfg.name}.log"
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu,
            "PYTHONPATH": str(root),
            "PYTHONNOUSERSITE": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "HF_HOME": env.get("HF_HOME", str(root / ".cache" / "huggingface")),
            "TRANSFORMERS_CACHE": env.get("TRANSFORMERS_CACHE", str(root / ".cache" / "huggingface" / "hub")),
        }
    )
    cmd = [
        python,
        "inference/infer.py",
        "--dataset",
        "tum_dynamic",
        "--output-dir",
        str(output_dir),
        "--max-frames-per-seq",
        os.environ.get("MAX_FRAMES", "300"),
        "--window-size",
        "0",
        "--checkpoint",
        checkpoint,
        "--overwrite",
        "--eval",
        "--enable-token-merging",
        "--token-merging-method",
        "frame_persistent_spatial",
        "--token-merging-ratio",
        os.environ.get("TOKEN_MERGING_RATIO", "0.9"),
        "--token-merging-start",
        "0",
        "--token-merging-frame-pool-stride",
        str(cfg.pool_stride),
        "--token-merging-frame-segment-threshold",
        str(cfg.segment),
        "--token-merging-frame-merge-threshold",
        str(cfg.merge),
        "--token-merging-frame-alpha",
        str(cfg.alpha),
        "--token-merging-frame-max-window",
        str(cfg.max_window),
        "--token-merging-frame-restore-layer",
        str(cfg.restore_layer),
        "--token-merging-frame-multi-max-group-size",
        str(cfg.max_group),
        "--token-merging-frame-multi-pair-threshold",
        str(cfg.pair),
        "--token-merging-frame-multi-span-threshold",
        str(cfg.span),
    ]
    started = time.time()
    with log_path.open("w") as log_file:
        log_file.write(f"gpu={gpu} config={cfg}\n")
        log_file.flush()
        proc = subprocess.run(cmd, cwd=root, env=env, stdout=log_file, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - started
    result: dict[str, object] = {
        "name": cfg.name,
        "gpu": gpu,
        "returncode": proc.returncode,
        "elapsed_sec": round(elapsed, 3),
        "pair": cfg.pair,
        "span": cfg.span,
        "merge": cfg.merge,
        "max_group": cfg.max_group,
        "pool_stride": cfg.pool_stride,
        "restore_layer": cfg.restore_layer,
        "output_dir": str(output_dir),
        "log": str(log_path),
    }
    summary_path = output_dir / "tum_dynamic" / "_summary_complete_scale_shift.json"
    if proc.returncode == 0 and summary_path.exists():
        with summary_path.open() as handle:
            summary = json.load(handle)
        video_depth = summary.get("video_depth", {})
        pose_auc = summary.get("pose_auc", {})
        result.update(
            {
                "abs_rel": video_depth.get("Abs Rel"),
                "delta_1.25": video_depth.get("delta < 1.25"),
                "auc@3": pose_auc.get("AUC@3"),
                "auc@30": pose_auc.get("AUC@30"),
                "fps": video_depth.get("fps"),
                "frame_merge_ratio": video_depth.get("frame_merge_merge_ratio_mean"),
                "active_frames": video_depth.get("frame_merge_active_frames_mean"),
            }
        )
    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "name",
        "pair",
        "span",
        "merge",
        "max_group",
        "pool_stride",
        "restore_layer",
        "abs_rel",
        "delta_1.25",
        "auc@3",
        "auc@30",
        "fps",
        "frame_merge_ratio",
        "active_frames",
        "gpu",
        "returncode",
        "elapsed_sec",
        "output_dir",
        "log",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    python = os.environ.get("PYTHON", "python")
    checkpoint = os.environ.get("CHECKPOINT", "checkpoints/vggt_omega_1b_512.pt")
    gpus = parse_gpus()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    search_dir = root / "outputs" / f"tum_multiframe_param_search_{timestamp}"
    search_dir.mkdir(parents=True, exist_ok=True)

    configs = CONFIGS[: int(os.environ.get("MAX_CONFIGS", str(len(CONFIGS))))]
    print(f"search_dir={search_dir}")
    print(f"gpus={','.join(gpus)} configs={len(configs)}")

    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {}
        for idx, cfg in enumerate(configs):
            gpu = gpus[idx % len(gpus)]
            futures[executor.submit(run_config, root, python, checkpoint, search_dir, gpu, cfg)] = cfg
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"done {result['name']} gpu={result['gpu']} rc={result['returncode']} "
                f"auc3={result.get('auc@3')} auc30={result.get('auc@30')} fps={result.get('fps')}"
            )

    results.sort(key=lambda row: str(row["name"]))
    passed = [
        row
        for row in results
        if (row.get("auc@3") is not None and row.get("auc@30") is not None and row.get("fps") is not None)
        and float(row["auc@3"]) > 36.0
        and float(row["auc@30"]) > 84.0
    ]
    passed.sort(key=lambda row: float(row["fps"]), reverse=True)

    write_csv(search_dir / "results_all.csv", results)
    write_csv(search_dir / "results_passed_auc3_gt36_auc30_gt84_by_fps.csv", passed)
    with (search_dir / "summary.md").open("w") as handle:
        handle.write("# TUM multiframe parameter search\n\n")
        handle.write("Criteria: AUC@3 > 36 and AUC@30 > 84. Passed rows are sorted by FPS.\n\n")
        handle.write(f"- configs: {len(configs)}\n")
        handle.write(f"- passed: {len(passed)}\n")
        handle.write(f"- gpus: {', '.join(gpus)}\n\n")
        for row in passed:
            handle.write(
                f"- `{row['name']}` fps={float(row['fps']):.6f}, "
                f"auc@3={float(row['auc@3']):.6f}, auc@30={float(row['auc@30']):.6f}, "
                f"abs_rel={float(row['abs_rel']):.6f}\n"
            )
    print(f"wrote {search_dir / 'results_all.csv'}")
    print(f"wrote {search_dir / 'results_passed_auc3_gt36_auc30_gt84_by_fps.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
