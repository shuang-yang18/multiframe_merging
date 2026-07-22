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
    pair: float
    span: float

    @property
    def name(self) -> str:
        pair_tag = f"{self.pair:.3f}".replace(".", "")
        span_tag = f"{self.span:.3f}".replace(".", "")
        return f"p{pair_tag}_s{span_tag}"


LEVELS = [0.90, 0.92, 0.94, 0.96, 0.97, 0.98, 0.985, 0.99, 0.995, 0.998]
CONFIGS = [Config(pair, span) for pair in LEVELS for span in LEVELS if span <= pair]


def parse_gpus() -> list[str]:
    return [item.strip() for item in os.environ.get("GPUS", "6,7").split(",") if item.strip()]


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
        "0.9",
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
        "3",
        "--token-merging-frame-multi-pair-threshold",
        str(cfg.pair),
        "--token-merging-frame-multi-span-threshold",
        str(cfg.span),
    ]
    started = time.time()
    with log_path.open("w") as log_file:
        log_file.write(f"gpu={gpu} pair={cfg.pair} span={cfg.span}\n")
        log_file.flush()
        proc = subprocess.run(cmd, cwd=root, env=env, stdout=log_file, stderr=subprocess.STDOUT, text=True)

    result: dict[str, object] = {
        "name": cfg.name,
        "pair": cfg.pair,
        "span": cfg.span,
        "max_group": 3,
        "token_merging_ratio": 0.9,
        "restore_layer": 24,
        "gpu": gpu,
        "returncode": proc.returncode,
        "elapsed_sec": round(time.time() - started, 3),
        "output_dir": str(output_dir),
        "log": str(log_path),
    }

    summary_path = output_dir / "tum_dynamic" / "_summary_complete_scale_shift.json"
    if proc.returncode == 0 and summary_path.exists():
        with summary_path.open() as handle:
            summary = json.load(handle)
        video_depth = summary.get("video_depth", {})
        pose_auc = summary.get("pose_auc", {})
        speed = summary.get("speed", {})
        result.update(
            {
                "abs_rel": video_depth.get("Abs Rel"),
                "delta_1.25": video_depth.get("delta < 1.25"),
                "auc@3": pose_auc.get("AUC@3"),
                "auc@30": pose_auc.get("AUC@30"),
                "fps": video_depth.get("fps") or speed.get("fps"),
                "frame_merge_ratio": speed.get("frame_merge_merge_ratio_mean"),
                "active_frames": speed.get("frame_merge_active_frames_mean"),
                "token_after_over_frame_merged": speed.get(
                    "token_merging_active_over_frame_merged_token_ratio_mean"
                ),
                "token_after_over_frame_original": speed.get(
                    "token_merging_active_over_frame_original_token_ratio_mean"
                ),
            }
        )
    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "name",
        "pair",
        "span",
        "max_group",
        "token_merging_ratio",
        "restore_layer",
        "abs_rel",
        "delta_1.25",
        "auc@3",
        "auc@30",
        "fps",
        "frame_merge_ratio",
        "active_frames",
        "token_after_over_frame_merged",
        "token_after_over_frame_original",
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
    configs = CONFIGS[: int(os.environ.get("MAX_CONFIGS", str(len(CONFIGS))))]
    search_dir = root / "outputs" / f"tum_multiframe_max3_pair_span_search_{time.strftime('%Y%m%d_%H%M%S')}"
    search_dir.mkdir(parents=True, exist_ok=True)

    print(f"search_dir={search_dir}", flush=True)
    print(f"gpus={','.join(gpus)} configs={len(configs)}", flush=True)
    print("fixed: max_group=3 r=0.9 restore=24 merge_threshold=0.1", flush=True)

    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {
            executor.submit(run_config, root, python, checkpoint, search_dir, gpus[idx % len(gpus)], cfg): cfg
            for idx, cfg in enumerate(configs)
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"done {result['name']} gpu={result['gpu']} rc={result['returncode']} "
                f"auc3={result.get('auc@3')} auc30={result.get('auc@30')} fps={result.get('fps')} "
                f"frame_merge={result.get('frame_merge_ratio')}",
                flush=True,
            )

    results.sort(key=lambda row: (float(row["pair"]), float(row["span"])))
    passed = [
        row
        for row in results
        if row.get("auc@3") is not None
        and row.get("auc@30") is not None
        and row.get("fps") is not None
        and float(row["auc@3"]) > 35.5
        and float(row["auc@30"]) > 84.0
    ]
    passed.sort(key=lambda row: float(row["fps"]), reverse=True)

    write_csv(search_dir / "results_all.csv", results)
    write_csv(search_dir / "results_passed_auc3_gt35p5_auc30_gt84_by_fps.csv", passed)
    with (search_dir / "summary.md").open("w") as handle:
        handle.write("# TUM max-group-3 pair/span search\n\n")
        handle.write("Fixed: `max_group=3`, `r=0.9`, `restore_layer=24`.\n\n")
        handle.write("Criteria: `AUC@3 > 35.5` and `AUC@30 > 84`. Passed rows sorted by FPS.\n\n")
        handle.write(f"- configs: {len(configs)}\n")
        handle.write(f"- passed: {len(passed)}\n")
        handle.write(f"- gpus: {', '.join(gpus)}\n\n")
        for row in passed:
            handle.write(
                f"- `{row['name']}` fps={float(row['fps']):.6f}, "
                f"auc@3={float(row['auc@3']):.6f}, auc@30={float(row['auc@30']):.6f}, "
                f"frame_merge={float(row['frame_merge_ratio']) * 100:.2f}%\n"
            )

    print(f"wrote {search_dir / 'results_all.csv'}", flush=True)
    print(f"wrote {search_dir / 'results_passed_auc3_gt35p5_auc30_gt84_by_fps.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
