"""Summarize Sintel video-depth speed and depth metrics for experiment outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from vggt_omega.evaluation import preprocess_sintel_depth


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="/data/mmc_syang/dataset/Sintel/training")
    parser.add_argument("--output-csv", default="outputs/video_depth_uniform_gpu2_summary_scale_shift.csv")
    parser.add_argument("--align", choices=["metric", "scale", "scale_shift"], default="scale_shift")
    parser.add_argument("runs", nargs="+", help="Entries formatted as label=prediction_dir")
    return parser.parse_args()


def align_scale_shift(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if pred.size > 200_000:
        sample_idx = np.linspace(0, pred.size - 1, 200_000, dtype=np.int64)
        fit_pred, fit_gt = pred[sample_idx], gt[sample_idx]
    else:
        fit_pred, fit_gt = pred, gt
    design = np.column_stack([fit_pred, np.ones_like(fit_pred)])
    params = np.array([np.median(fit_gt) / max(np.median(fit_pred), 1e-8), 0.0])
    for _ in range(20):
        residual = design @ params - fit_gt
        weights = 1.0 / np.maximum(np.abs(residual), 1e-4)
        sw = weights.sum()
        sx = np.dot(weights, fit_pred)
        sy = np.dot(weights, fit_gt)
        sxx = np.dot(weights, fit_pred * fit_pred)
        mean_x = sx / sw
        mean_y = sy / sw
        centered_x = fit_pred - mean_x
        variance_x = np.dot(weights, centered_x * centered_x)
        tolerance = np.finfo(np.float64).eps * max(abs(sxx), 1.0)
        if variance_x <= tolerance:
            weighted_design = design * np.sqrt(weights)[:, None]
            weighted_gt = fit_gt * np.sqrt(weights)
            params = np.linalg.lstsq(weighted_design, weighted_gt, rcond=None)[0]
        else:
            scale = np.dot(weights, centered_x * (fit_gt - mean_y)) / variance_x
            params = np.array([scale, mean_y - scale * mean_x])
    return params[0] * pred + params[1]


def depth_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    align: str,
    max_depth: float = 70.0,
) -> dict[str, float | int]:
    mask = np.isfinite(prediction) & np.isfinite(ground_truth) & (ground_truth > 0) & (ground_truth < max_depth)
    pred = prediction[mask].astype(np.float64)
    gt = ground_truth[mask].astype(np.float64)
    if pred.size == 0:
        raise ValueError("No valid depth pixels available for evaluation.")
    if align == "scale":
        pred = pred * (np.median(gt) / max(np.median(pred), 1e-8))
    elif align == "scale_shift":
        pred = align_scale_shift(pred, gt)
    elif align != "metric":
        raise ValueError(f"Unknown depth alignment mode: {align}")
    pred = np.clip(pred, 1e-5, max_depth)
    ratio = np.maximum(pred / gt, gt / pred)
    return {
        "abs_rel": float(np.mean(np.abs(pred - gt) / gt)),
        "sq_rel": float(np.mean((pred - gt) ** 2 / gt)),
        "rmse": float(np.sqrt(np.mean((pred - gt) ** 2))),
        "log_rmse": float(np.sqrt(np.mean((np.log(pred) - np.log(gt)) ** 2))),
        "delta_1": float(np.mean(ratio < 1.25)),
        "delta_2": float(np.mean(ratio < 1.25**2)),
        "delta_3": float(np.mean(ratio < 1.25**3)),
        "valid_pixels": int(pred.size),
    }


def summarize_run(
    label: str,
    pred_dir: Path,
    dataset_root: Path,
    align: str,
    ground_truth_cache: dict[tuple[str, int, int], np.ndarray],
) -> dict[str, float | int | str]:
    pred_root = pred_dir / "sintel"
    totals = {key: 0.0 for key in ("abs_rel", "sq_rel", "rmse", "log_rmse", "delta_1", "delta_2", "delta_3")}
    valid_pixels = 0
    frames = 0
    elapsed = 0.0
    peak_allocated = 0.0
    peak_reserved = 0.0
    register_patch_sample_tokens = None
    register_patch_sample_ratio = None
    register_patch_sample_mode = None

    for seq_dir in sorted(path for path in pred_root.iterdir() if path.is_dir()):
        seq = seq_dir.name
        with (seq_dir / "_time.json").open() as handle:
            timing = json.load(handle)
        width, height = timing["resolution"]
        frames += timing["frames"]
        elapsed += timing["time"]
        peak_allocated = max(peak_allocated, timing.get("peak_memory_allocated_gb", 0.0))
        peak_reserved = max(peak_reserved, timing.get("peak_memory_reserved_gb", 0.0))
        register_patch_sample_tokens = timing.get("register_patch_sample_tokens", register_patch_sample_tokens)
        register_patch_sample_ratio = timing.get("register_patch_sample_ratio", register_patch_sample_ratio)
        register_patch_sample_mode = timing.get("register_patch_sample_mode", register_patch_sample_mode)

        pred_paths = sorted(seq_dir.glob("frame_*.npy"))
        image_paths = sorted((dataset_root / "final" / seq).glob("*.png"))
        depth_paths = sorted((dataset_root / "depth" / seq).glob("*.dpt"))
        if not (len(pred_paths) == len(image_paths) == len(depth_paths)):
            raise ValueError(
                f"{label}/{seq}: found {len(pred_paths)} predictions, "
                f"{len(image_paths)} images, and {len(depth_paths)} depths"
            )
        predictions = np.stack([np.load(path) for path in pred_paths])
        cache_key = (seq, width, height)
        if cache_key not in ground_truth_cache:
            ground_truth_cache[cache_key] = np.stack(
                [
                    preprocess_sintel_depth(image_path, depth_path, (width, height))
                    for image_path, depth_path in zip(image_paths, depth_paths)
                ]
            )
        ground_truth = ground_truth_cache[cache_key]
        metrics = depth_metrics(predictions, ground_truth, align)
        for key in totals:
            totals[key] += metrics[key] * metrics["valid_pixels"]
        valid_pixels += metrics["valid_pixels"]

    row = {key: totals[key] / valid_pixels for key in totals}
    row.update(
        {
            "label": label,
            "register_patch_sample_tokens": register_patch_sample_tokens,
            "register_patch_sample_ratio": register_patch_sample_ratio,
            "register_patch_sample_mode": register_patch_sample_mode,
            "frames": frames,
            "time": elapsed,
            "fps": frames / elapsed,
            "seconds_per_frame": elapsed / frames,
            "peak_memory_allocated_gb": peak_allocated,
            "peak_memory_reserved_gb": peak_reserved,
            "valid_pixels": valid_pixels,
        }
    )
    return row


def main() -> None:
    args = parse_args()
    rows = []
    ground_truth_cache = {}
    for item in args.runs:
        label, pred_dir = item.split("=", 1)
        rows.append(summarize_run(label, Path(pred_dir), Path(args.dataset_root), args.align, ground_truth_cache))

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "register_patch_sample_tokens",
        "register_patch_sample_ratio",
        "register_patch_sample_mode",
        "frames",
        "time",
        "fps",
        "seconds_per_frame",
        "peak_memory_allocated_gb",
        "peak_memory_reserved_gb",
        "abs_rel",
        "sq_rel",
        "rmse",
        "log_rmse",
        "delta_1",
        "delta_2",
        "delta_3",
        "valid_pixels",
    ]
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(output_csv)
    for row in rows:
        print(
            f"{row['label']}: AbsRel={row['abs_rel']:.4f}, RMSE={row['rmse']:.4f}, "
            f"d1={row['delta_1']:.4f}, fps={row['fps']:.2f}, "
            f"peak={row['peak_memory_allocated_gb']:.3f} GB"
        )


if __name__ == "__main__":
    main()
