#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path


@dataclass(frozen=True)
class Config:
    pair: float
    span: float
    ratio: float

    @property
    def name(self) -> str:
        pair_tag = f"{self.pair:.4f}".rstrip("0").rstrip(".").replace(".", "")
        span_tag = f"{self.span:.4f}".rstrip("0").rstrip(".").replace(".", "")
        ratio_tag = f"{self.ratio:.3f}".rstrip("0").rstrip(".").replace(".", "")
        return f"p{pair_tag}_s{span_tag}_r{ratio_tag}"


def parse_float_list(env_name: str, default: str) -> list[float]:
    raw = os.environ.get(env_name, default)
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError(f"{env_name} produced an empty value list")
    return values


def parse_float_range(prefix: str, default_min: str, default_max: str, default_step: str) -> list[float]:
    explicit = os.environ.get(f"{prefix}_VALUES")
    if explicit:
        return parse_float_list(f"{prefix}_VALUES", explicit)

    start = Decimal(os.environ.get(f"{prefix}_MIN", default_min))
    end = Decimal(os.environ.get(f"{prefix}_MAX", default_max))
    step = Decimal(os.environ.get(f"{prefix}_STEP", default_step))
    if step <= 0:
        raise ValueError(f"{prefix}_STEP must be positive")
    values: list[float] = []
    value = start
    while value <= end:
        values.append(float(value))
        value += step
    if not values:
        raise ValueError(f"{prefix} range produced an empty value list")
    return values


def parse_gpus() -> list[str]:
    gpus = [item.strip() for item in os.environ.get("GPUS", "6,7").split(",") if item.strip()]
    if not gpus:
        raise ValueError("GPUS must contain at least one GPU id")
    return gpus


def make_configs() -> list[Config]:
    pair_values = parse_float_range("PAIR", "0.976", "0.996", "0.001")
    span_values = parse_float_range("SPAN", "0.938", "0.958", "0.001")
    ratio_values = parse_float_range("R", "0.50", "0.90", "0.01")
    configs = [
        Config(pair=pair, span=span, ratio=ratio)
        for pair in pair_values
        for span in span_values
        for ratio in ratio_values
        if span <= pair
    ]
    if not configs:
        raise ValueError("No configs after filtering span <= pair")
    if os.environ.get("CENTERED_ORDER", "1") != "0":
        pair_center = float(os.environ.get("PAIR_CENTER", "0.986"))
        span_center = float(os.environ.get("SPAN_CENTER", "0.948"))
        ratio_center = float(os.environ.get("R_CENTER", "0.9"))
        configs.sort(
            key=lambda cfg: (
                abs(cfg.pair - pair_center) + abs(cfg.span - span_center) + abs(cfg.ratio - ratio_center),
                abs(cfg.pair - pair_center),
                abs(cfg.span - span_center),
                abs(cfg.ratio - ratio_center),
            )
        )
    return configs


def config_key(cfg: Config) -> tuple[int, int, int]:
    return (round(cfg.pair * 1000), round(cfg.span * 1000), round(cfg.ratio * 100))


def config_from_key(key: tuple[int, int, int]) -> Config:
    return Config(pair=key[0] / 1000.0, span=key[1] / 1000.0, ratio=key[2] / 100.0)


def neighbor_configs(cfg: Config, valid_keys: set[tuple[int, int, int]], seen: set[tuple[int, int, int]]) -> list[Config]:
    pair0, span0, ratio0 = config_key(cfg)
    keys: list[tuple[int, int, int]] = []
    for dp in (-1, 0, 1):
        for ds in (-1, 0, 1):
            for dr in (-1, 0, 1):
                if dp == 0 and ds == 0 and dr == 0:
                    continue
                key = (pair0 + dp, span0 + ds, ratio0 + dr)
                if key in valid_keys and key not in seen:
                    keys.append(key)
    keys.sort(key=lambda key: (abs(key[0] - pair0) + abs(key[1] - span0) + abs(key[2] - ratio0), key))
    return [config_from_key(key) for key in keys]


def next_global_config(configs: list[Config], cursor: int, seen: set[tuple[int, int, int]]) -> tuple[Config | None, int]:
    while cursor < len(configs):
        cfg = configs[cursor]
        cursor += 1
        if config_key(cfg) not in seen:
            return cfg, cursor
    return None, cursor


def load_existing_results(search_dir: Path) -> list[dict[str, object]]:
    csv_path = search_dir / "results_all_by_auc3.csv"
    if not csv_path.exists():
        return []
    rows: list[dict[str, object]] = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            row: dict[str, object] = dict(raw)
            for key in (
                "pair",
                "span",
                "r",
                "abs_rel",
                "delta_1.25",
                "auc@3",
                "auc@30",
                "fps",
                "frame_merge_ratio",
                "active_frames",
                "token_after_over_frame_merged",
                "token_after_over_frame_original",
                "elapsed_sec",
            ):
                if row.get(key) not in {"", None}:
                    row[key] = float(row[key])
            for key in ("max_group", "restore_layer", "returncode"):
                if row.get(key) not in {"", None}:
                    row[key] = int(float(row[key]))
            rows.append(row)
    return rows


def run_config(root: Path, python: str, checkpoint: str, search_dir: Path, gpu: str, cfg: Config) -> dict[str, object]:
    output_dir = search_dir / cfg.name
    log_path = search_dir / f"{cfg.name}.log"
    summaries_dir = search_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
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
    layered_schedule = f"1-10:{cfg.ratio:g},11-18:0.0,19-24:{cfg.ratio:g}"
    cmd = [
        python,
        "inference/infer.py",
        "--dataset",
        os.environ.get("DATASET", "tum_dynamic"),
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
        str(cfg.ratio),
        "--token-merging-layer-ratios",
        layered_schedule,
        "--token-merging-start",
        os.environ.get("TOKEN_MERGING_START", "0"),
        "--token-merging-frame-pool-stride",
        os.environ.get("POOL_STRIDE", "2"),
        "--token-merging-frame-segment-threshold",
        os.environ.get("SEGMENT_THRESHOLD", "0.9"),
        "--token-merging-frame-merge-threshold",
        os.environ.get("MERGE_THRESHOLD", "0.1"),
        "--token-merging-frame-alpha",
        os.environ.get("FRAME_ALPHA", "0.1"),
        "--token-merging-frame-max-window",
        os.environ.get("MAX_WINDOW", "20"),
        "--token-merging-frame-restore-layer",
        os.environ.get("RESTORE_LAYER", "24"),
        "--token-merging-frame-multi-max-group-size",
        os.environ.get("MAX_GROUP_SIZE", "4"),
        "--token-merging-frame-multi-pair-threshold",
        str(cfg.pair),
        "--token-merging-frame-multi-span-threshold",
        str(cfg.span),
    ]
    started = time.time()
    with log_path.open("w") as log_file:
        log_file.write(f"gpu={gpu} config={cfg}\n")
        log_file.write("cmd=" + " ".join(cmd) + "\n")
        log_file.flush()
        proc = subprocess.run(cmd, cwd=root, env=env, stdout=log_file, stderr=subprocess.STDOUT, text=True)

    result: dict[str, object] = {
        "name": cfg.name,
        "pair": cfg.pair,
        "span": cfg.span,
        "r": cfg.ratio,
        "token_merging_layer_ratios": layered_schedule,
        "max_group": int(os.environ.get("MAX_GROUP_SIZE", "4")),
        "restore_layer": int(os.environ.get("RESTORE_LAYER", "24")),
        "gpu": gpu,
        "returncode": proc.returncode,
        "elapsed_sec": round(time.time() - started, 3),
        "output_dir": str(output_dir),
        "log": str(log_path),
        "summary_json": "",
    }
    dataset = os.environ.get("DATASET", "tum_dynamic")
    summary_dir = output_dir / dataset
    summary_path = summary_dir / "_summary_complete_scale_shift.json"
    if not summary_path.exists():
        summary_path = summary_dir / "_summary_scale_shift.json"
    if proc.returncode == 0 and summary_path.exists():
        kept_summary = summaries_dir / f"{cfg.name}_summary.json"
        shutil.copy2(summary_path, kept_summary)
        result["summary_json"] = str(kept_summary)
        with summary_path.open() as handle:
            summary = json.load(handle)
        video_depth = summary.get("video_depth", {})
        pose_auc = summary.get("pose_auc") or {}
        speed = summary.get("speed") or {}
        result.update(
            {
                "abs_rel": video_depth.get("Abs Rel"),
                "delta_1.25": video_depth.get("delta < 1.25"),
                "auc@3": pose_auc.get("AUC@3"),
                "auc@30": pose_auc.get("AUC@30"),
                "fps": speed.get("fps") or video_depth.get("fps"),
                "frame_merge_ratio": speed.get("frame_merge_merge_ratio_mean")
                or video_depth.get("frame_merge_merge_ratio_mean"),
                "active_frames": speed.get("frame_merge_active_frames_mean")
                or video_depth.get("frame_merge_active_frames_mean"),
                "token_after_over_frame_merged": speed.get(
                    "token_merging_active_over_frame_merged_token_ratio_mean"
                ),
                "token_after_over_frame_original": speed.get(
                    "token_merging_active_over_frame_original_token_ratio_mean"
                ),
            }
        )
        per_sequence_dir = search_dir / "per_sequence" / cfg.name
        per_sequence_dir.mkdir(parents=True, exist_ok=True)
        for filename in ("_sequence_metrics_scale_shift.csv", "_summary_pose_auc.json"):
            artifact = summary_dir / filename
            if artifact.exists():
                shutil.copy2(artifact, per_sequence_dir / filename)
    if os.environ.get("CLEAN_OUTPUTS", "1") != "0":
        shutil.rmtree(output_dir, ignore_errors=True)
    return result


def sort_by_auc3(rows: list[dict[str, object]]) -> list[dict[str, object]]:
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


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "rank_by_auc3",
        "name",
        "pair",
        "span",
        "r",
        "token_merging_layer_ratios",
        "max_group",
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
        "summary_json",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(rows, start=1):
            enriched = {"rank_by_auc3": idx, **row}
            writer.writerow({key: enriched.get(key, "") for key in fieldnames})


def write_summary(path: Path, rows: list[dict[str, object]], configs: list[Config], gpus: list[str]) -> None:
    with path.open("w") as handle:
        handle.write("# TUM pair/span/r search sorted by AUC@3\n\n")
        handle.write("Fixed settings: `frame_persistent_spatial`, `k0`, `max_group=4`, `restore_layer=24`, `300` frames.\n\n")
        handle.write("Layer schedule: 0-based blocks `10-17` use `r=0`; all other global blocks use candidate `r`. "
                     "CLI schedule per row: `1-10:r,11-18:0.0,19-24:r`.\n\n")
        handle.write(f"- finished configs: {len(rows)} / {len(configs)}\n")
        handle.write(f"- gpus: {', '.join(gpus)}\n")
        handle.write(f"- adaptive search: {os.environ.get('ADAPTIVE_SEARCH', '1') != '0'}\n")
        handle.write(f"- pair values: {', '.join(str(v) for v in sorted({cfg.pair for cfg in configs}))}\n")
        handle.write(f"- span values: {', '.join(str(v) for v in sorted({cfg.span for cfg in configs}))}\n")
        handle.write(f"- r values: {', '.join(str(v) for v in sorted({cfg.ratio for cfg in configs}))}\n\n")
        handle.write("| Rank | Config | AUC@3 | AUC@30 | FPS | Abs Rel | Frame Merge | Token/Original |\n")
        handle.write("|---:|---|---:|---:|---:|---:|---:|---:|\n")
        for idx, row in enumerate(rows, start=1):
            handle.write(
                f"| {idx} | `{row['name']}` | "
                f"{float(row.get('auc@3') or 0):.6f} | "
                f"{float(row.get('auc@30') or 0):.6f} | "
                f"{float(row.get('fps') or 0):.6f} | "
                f"{float(row.get('abs_rel') or 0):.6f} | "
                f"{float(row.get('frame_merge_ratio') or 0) * 100:.2f}% | "
                f"{float(row.get('token_after_over_frame_original') or 0) * 100:.2f}% |\n"
            )


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    python = os.environ.get("PYTHON", "python")
    checkpoint = os.environ.get("CHECKPOINT", "checkpoints/vggt_omega_1b_512.pt")
    gpus = parse_gpus()
    configs = make_configs()
    max_configs = int(os.environ.get("MAX_CONFIGS", str(len(configs))))
    configs = configs[:max_configs]
    time_budget_hours = float(os.environ.get("TIME_BUDGET_HOURS", "0"))
    deadline = float("inf") if time_budget_hours <= 0 else time.time() + time_budget_hours * 3600
    search_dir_env = os.environ.get("SEARCH_DIR", "")
    search_dir = Path(search_dir_env) if search_dir_env else root / "new_results" / f"tum_pair_span_r_search_{time.strftime('%Y%m%d_%H%M%S')}"
    if not search_dir.is_absolute():
        search_dir = root / search_dir
    search_dir.mkdir(parents=True, exist_ok=True)

    print(f"search_dir={search_dir}", flush=True)
    print(
        f"gpus={','.join(gpus)} configs={len(configs)} "
        f"time_budget_hours={time_budget_hours or 'unlimited'} "
        "layered_schedule=1-10:r,11-18:0.0,19-24:r",
        flush=True,
    )

    valid_keys = {config_key(cfg) for cfg in configs}
    seen: set[tuple[int, int, int]] = set()
    frontier: deque[Config] = deque()
    global_cursor = 0
    best_auc3 = float("-inf")
    adaptive = os.environ.get("ADAPTIVE_SEARCH", "1") != "0"
    results: list[dict[str, object]] = load_existing_results(search_dir) if os.environ.get("RESUME_SEARCH", "1") != "0" else []
    for row in results:
        if row.get("returncode") == 0 and row.get("pair") is not None and row.get("span") is not None and row.get("r") is not None:
            cfg = Config(pair=float(row["pair"]), span=float(row["span"]), ratio=float(row["r"]))
            seen.add(config_key(cfg))
            if row.get("auc@3") is not None and float(row["auc@3"]) > best_auc3:
                best_auc3 = float(row["auc@3"])
    if adaptive and results and best_auc3 > float("-inf"):
        best_row = sort_by_auc3(results)[0]
        best_cfg = Config(pair=float(best_row["pair"]), span=float(best_row["span"]), ratio=float(best_row["r"]))
        for cfg in reversed(neighbor_configs(best_cfg, valid_keys, seen)):
            frontier.appendleft(cfg)
        print(f"resumed {len(results)} rows, seen={len(seen)}, best={best_cfg.name} auc3={best_auc3}", flush=True)
    elif results:
        print(f"resumed {len(results)} rows, seen={len(seen)}", flush=True)

    def take_next_config() -> Config | None:
        nonlocal global_cursor
        while adaptive and frontier:
            cfg = frontier.popleft()
            key = config_key(cfg)
            if key not in seen and key in valid_keys:
                seen.add(key)
                return cfg
        cfg, global_cursor = next_global_config(configs, global_cursor, seen)
        if cfg is not None:
            seen.add(config_key(cfg))
        return cfg

    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        pending = {}
        for gpu in gpus:
            if time.time() >= deadline:
                break
            cfg = take_next_config()
            if cfg is None:
                break
            pending[executor.submit(run_config, root, python, checkpoint, search_dir, gpu, cfg)] = (gpu, cfg)

        while pending:
            for future in as_completed(pending):
                gpu, _cfg = pending.pop(future)
                break
            result = future.result()
            results.append(result)
            auc3 = result.get("auc@3")
            if adaptive and auc3 is not None and float(auc3) > best_auc3:
                best_auc3 = float(auc3)
                best_cfg = Config(pair=float(result["pair"]), span=float(result["span"]), ratio=float(result["r"]))
                neighbors = neighbor_configs(best_cfg, valid_keys, seen)
                for cfg in reversed(neighbors):
                    frontier.appendleft(cfg)
                print(f"new best {best_cfg.name} auc3={best_auc3}; queued_neighbors={len(neighbors)}", flush=True)
            sorted_rows = sort_by_auc3(results)
            write_csv(search_dir / "results_all_by_auc3.csv", sorted_rows)
            write_summary(search_dir / "summary_by_auc3.md", sorted_rows, configs, gpus)
            print(
                f"done {result['name']} gpu={result['gpu']} rc={result['returncode']} "
                f"auc3={result.get('auc@3')} auc30={result.get('auc@30')} fps={result.get('fps')}",
                flush=True,
            )
            if time.time() < deadline:
                cfg = take_next_config()
                if cfg is None:
                    continue
                pending[executor.submit(run_config, root, python, checkpoint, search_dir, gpu, cfg)] = (gpu, cfg)
            else:
                print("time budget reached; no new configs will be scheduled", flush=True)

    sorted_rows = sort_by_auc3(results)
    write_csv(search_dir / "results_all_by_auc3.csv", sorted_rows)
    write_summary(search_dir / "summary_by_auc3.md", sorted_rows, configs, gpus)
    print(f"wrote {search_dir / 'results_all_by_auc3.csv'}", flush=True)
    print(f"wrote {search_dir / 'summary_by_auc3.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
