"""Evaluate VGGT-Omega video-depth predictions on supported benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from PIL import Image

from vggt_omega.evaluation import preprocess_sintel_depth, read_bonn_depth, write_csv

try:
    import cv2
except ImportError:  # pragma: no cover - runtime fallback for minimal environments.
    cv2 = None

if __package__:
    from .bonn_association import association_paths
    from .infer import DEFAULT_DATASET_ROOTS, load_manifest_selected_images, sequence_images, sequence_names, sequence_poses
    from .pose_auc import evaluate_pose_auc, summarize_pose_auc, write_json
else:
    from bonn_association import association_paths
    from infer import DEFAULT_DATASET_ROOTS, load_manifest_selected_images, sequence_images, sequence_names, sequence_poses
    from pose_auc import evaluate_pose_auc, summarize_pose_auc, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["sintel", "bonn", "7scenes", "tum_dynamic", "nrgbd", "all"], default="all")
    parser.add_argument("--dataset-root")
    parser.add_argument("--bonn-rgb-dir", default="rgb_110", help="Bonn RGB subdirectory; use 'rgb' for full-length sequences.")
    parser.add_argument("--bonn-depth-dir", default="depth_110", help="Bonn depth subdirectory; use 'depth' for full-length sequences.")
    parser.add_argument(
        "--bonn-split",
        choices=["test", "all"],
        default="test",
        help="Bonn split to use. Defaults to the five-sequence benchmark test split.",
    )
    parser.add_argument("--pred-dir", default="outputs/video_depth")
    parser.add_argument("--output-dir", default="outputs/video_depth")
    parser.add_argument("--align", choices=["metric", "scale", "scale_shift"], default="scale_shift")
    parser.add_argument("--max-depth", type=float, default=None)
    parser.add_argument(
        "--pose-eval-frames",
        type=int,
        default=0,
        help=(
            "Number of frames to sample only for pose AUC. Default 0 evaluates all inferred frames; "
            "for paper-style 10-view VGGT evaluation, run inference with 10 input frames instead of "
            "subsampling a longer prediction afterward."
        ),
    )
    parser.add_argument("--pose-eval-seed", type=int, default=0, help="Seed for deterministic pose-AUC frame sampling.")
    parser.add_argument("--all-scenes", action="store_true", default=True, help="Evaluate every scene available in the dataset root.")
    parser.add_argument("--sequences", nargs="*")
    parser.add_argument(
        "--seven-scenes-split",
        choices=["test", "train", "all"],
        default="test",
        help="7Scenes split to use when --dataset 7scenes. Defaults to the official test split.",
    )
    parser.add_argument(
        "--max-frames-per-seq",
        type=int,
        default=0,
        help="Retained for command compatibility; evaluation always uses the frames recorded by inference.",
    )
    return parser.parse_args()


def read_png_depth(filename: str | Path, scale: float) -> np.ndarray:
    raw = np.asarray(Image.open(filename))
    depth = raw.astype(np.float32) / scale
    depth[raw == 0] = -1.0
    return depth


DEFAULT_MAX_DEPTHS = {
    "sintel": 70.0,
    "bonn": 70.0,
    "tum_dynamic": 70.0,
    "7scenes": 10.0,
    "nrgbd": 10.0,
}


def resolve_max_depth(dataset: str, max_depth: float | None) -> float:
    return DEFAULT_MAX_DEPTHS[dataset] if max_depth is None else max_depth


def _timestamp(path: str | Path) -> float:
    return float(Path(path).stem)


def _associate_tum_depths(dataset_root: str | Path, seq: str, image_paths: list[str]) -> list[str]:
    depth_txt = Path(dataset_root) / seq / "depth.txt"
    if not depth_txt.is_file():
        raise FileNotFoundError(f"No TUM-Dynamics depth.txt found for sequence {seq}")
    entries = []
    with depth_txt.open() as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            stamp, rel_path = line.split()[:2]
            entries.append((float(stamp), str(Path(dataset_root) / seq / rel_path)))
    if not entries:
        raise FileNotFoundError(f"No TUM-Dynamics depth entries found in {depth_txt}")
    depth_times = np.asarray([item[0] for item in entries])
    depth_paths = [item[1] for item in entries]
    matched = []
    for image_path in image_paths:
        idx = int(np.argmin(np.abs(depth_times - _timestamp(image_path))))
        matched.append(depth_paths[idx])
    return matched


def _associate_bonn_depths(dataset_root: str | Path, seq: str, depth_dir: str, image_paths: list[str]) -> list[str]:
    paths = sorted((Path(dataset_root) / seq / depth_dir).glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No Bonn ground truth depths found below {Path(dataset_root) / seq / depth_dir}")
    depth_times = np.asarray([_timestamp(path) for path in paths])
    depth_paths = [str(path) for path in paths]
    matched = []
    for image_path in image_paths:
        idx = int(np.argmin(np.abs(depth_times - _timestamp(image_path))))
        matched.append(depth_paths[idx])
    return matched


def sequence_depths(
    dataset: str,
    dataset_root: str | Path,
    seq: str,
    bonn_depth_dir: str = "depth_110",
    bonn_rgb_dir: str = "rgb_110",
    image_paths: list[str] | None = None,
) -> list[str]:
    if dataset == "sintel":
        paths = sorted((Path(dataset_root) / "depth" / seq).glob("*.dpt"))
    elif dataset == "7scenes":
        paths = sorted((Path(dataset_root) / seq).glob("*.depth.png"))
    elif dataset == "tum_dynamic":
        images = image_paths or sequence_images(dataset, dataset_root, seq, bonn_rgb_dir)
        return _associate_tum_depths(dataset_root, seq, images)
    elif dataset == "nrgbd":
        paths = sorted(
            (Path(dataset_root) / seq / "depth").glob("depth*.png"),
            key=lambda path: int(path.stem.removeprefix("depth")),
        )
    else:
        if image_paths is not None:
            return _associate_bonn_depths(dataset_root, seq, bonn_depth_dir, image_paths)
        paths = sorted((Path(dataset_root) / seq / bonn_depth_dir).glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No {dataset} ground truth depths found for sequence {seq}")
    return [str(path) for path in paths]


def preprocess_depth(
    dataset: str, image_filename: str, depth_filename: str, resolution: tuple[int, int]
) -> np.ndarray:
    if dataset == "sintel":
        return preprocess_sintel_depth(image_filename, depth_filename, resolution)
    if dataset == "7scenes":
        raw = np.asarray(Image.open(depth_filename))
        depth = raw.astype(np.float32) / 1000.0
        depth[(raw == 0) | (raw == 65535)] = -1.0
        return depth
    if dataset == "tum_dynamic":
        return read_png_depth(depth_filename, 5000.0)
    if dataset == "nrgbd":
        return read_png_depth(depth_filename, 1000.0)
    return read_bonn_depth(depth_filename).astype(np.float32)


def resize_prediction_to_gt(prediction: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    if prediction.shape == ground_truth.shape:
        return prediction.astype(np.float32, copy=False)
    height, width = ground_truth.shape
    if cv2 is not None:
        return cv2.resize(prediction.astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC)
    image = Image.fromarray(prediction.astype(np.float32), mode="F")
    return np.asarray(image.resize((width, height), Image.Resampling.BICUBIC), dtype=np.float32)


def align_depth(pred: np.ndarray, gt: np.ndarray, mode: str) -> np.ndarray:
    if mode == "metric":
        return pred
    if mode == "scale":
        scale = np.median(gt) / max(np.median(pred), 1e-8)
        return pred * scale

    # Robust affine alignment for relative VGGT depth. Fit on a bounded, evenly
    # spaced sample so long videos do not let a single sequence dominate runtime.
    pred64 = pred.astype(np.float64, copy=False)
    gt64 = gt.astype(np.float64, copy=False)
    if pred64.size > 200_000:
        sample_idx = np.linspace(0, pred64.size - 1, 200_000, dtype=np.int64)
        fit_pred, fit_gt = pred64[sample_idx], gt64[sample_idx]
    else:
        fit_pred, fit_gt = pred64, gt64
    design = np.column_stack([fit_pred, np.ones_like(fit_pred)])
    params = np.array([np.median(fit_gt) / max(np.median(fit_pred), 1e-8), 0.0])
    for _ in range(20):
        residual = design @ params - fit_gt
        weights = 1.0 / np.maximum(np.abs(residual), 1e-4)
        weight_sum = weights.sum()
        if weight_sum <= 0:
            break
        mean_pred = np.dot(weights, fit_pred) / weight_sum
        mean_gt = np.dot(weights, fit_gt) / weight_sum
        centered_pred = fit_pred - mean_pred
        variance = np.dot(weights, centered_pred * centered_pred)
        if variance <= np.finfo(np.float64).eps:
            weighted_design = design * np.sqrt(weights)[:, None]
            weighted_gt = fit_gt * np.sqrt(weights)
            params = np.linalg.lstsq(weighted_design, weighted_gt, rcond=None)[0]
        else:
            scale = np.dot(weights, centered_pred * (fit_gt - mean_gt)) / variance
            params = np.array([scale, mean_gt - scale * mean_pred])
    return (params[0] * pred64 + params[1]).astype(np.float32)


def depth_metrics(prediction: np.ndarray, ground_truth: np.ndarray, mode: str, max_depth: float) -> dict:
    mask = (
        np.isfinite(prediction)
        & np.isfinite(ground_truth)
        & (ground_truth > 0)
        & (ground_truth < max_depth)
    )
    pred = prediction[mask].astype(np.float32)
    gt = ground_truth[mask].astype(np.float32)
    if pred.size == 0:
        raise ValueError("No valid depth pixels available for evaluation.")
    pred = np.clip(align_depth(pred, gt, mode), 1e-5, max_depth)
    ratio = np.maximum(pred / gt, gt / pred)
    return {
        "depth_eval_protocol": "vggt_omega_robust_scale_shift",
        "Abs Rel": float(np.mean(np.abs(pred - gt) / gt)),
        "Sq Rel": float(np.mean((pred - gt) ** 2 / gt)),
        "RMSE": float(np.sqrt(np.mean((pred - gt) ** 2))),
        "Log RMSE": float(np.sqrt(np.mean((np.log(pred) - np.log(gt)) ** 2))),
        "delta < 1.": float(np.mean(ratio < 1.0)),
        "delta < 1.25": float(np.mean(ratio < 1.25)),
        "delta < 1.25^2": float(np.mean(ratio < 1.25**2)),
        "delta < 1.25^3": float(np.mean(ratio < 1.25**3)),
        "valid_pixels": int(pred.size),
    }


def evaluate_dataset(args: argparse.Namespace, dataset: str) -> dict:
    dataset_root = args.dataset_root or DEFAULT_DATASET_ROOTS[dataset]
    sequences = sequence_names(
        dataset,
        dataset_root,
        args.sequences,
        args.all_scenes,
        args.seven_scenes_split,
        args.bonn_rgb_dir,
        args.bonn_split,
    )
    pred_root = Path(args.pred_dir) / dataset
    output_root = Path(args.output_dir) / dataset
    output_root.mkdir(parents=True, exist_ok=True)
    sequence_rows = []
    # The input value is already a sequence-level mean. Aggregate sequences
    # uniformly so long sequences or high-resolution frames do not dominate it.
    final_token_over_initial_token_ratios = []
    frame_token_ratios = []

    for seq in sequences:
        bonn_manifest = None
        if dataset == "bonn":
            association_path = pred_root / seq / "_bonn_associations.json"
            if not association_path.is_file():
                raise ValueError(
                    f"{dataset}/{seq}: missing timestamp association manifest {association_path}; "
                    "legacy Bonn predictions cannot be evaluated with the corrected protocol."
                )
            with association_path.open() as handle:
                bonn_manifest = json.load(handle)
            images, gt_paths = association_paths(
                dataset_root,
                seq,
                bonn_manifest["associations"],
                rgb_dir=bonn_manifest["rgb_dir"],
                depth_dir=bonn_manifest["depth_dir"],
            )
        else:
            source_images = sequence_images(dataset, dataset_root, seq, args.bonn_rgb_dir)
            images, selected_indices = load_manifest_selected_images(pred_root / seq, source_images)
            all_gt_paths = sequence_depths(
                dataset,
                dataset_root,
                seq,
                args.bonn_depth_dir,
                args.bonn_rgb_dir,
                image_paths=images if dataset == "tum_dynamic" else None,
            )
            if dataset == "tum_dynamic":
                gt_paths = all_gt_paths
            else:
                if len(all_gt_paths) != len(source_images):
                    raise ValueError(
                        f"{dataset}/{seq}: {len(all_gt_paths)} depth frames do not match "
                        f"{len(source_images)} RGB frames."
                    )
                gt_paths = [all_gt_paths[index] for index in selected_indices]
        pred_paths = sorted((pred_root / seq).glob("frame_*.npy"))
        if not (len(pred_paths) == len(gt_paths) == len(images)):
            raise ValueError(
                f"{dataset}/{seq}: found {len(pred_paths)} predictions, "
                f"{len(gt_paths)} ground truth depths, and {len(images)} images"
            )
        with (pred_root / seq / "_time.json").open() as handle:
            timing = json.load(handle)
        final_token_ratio = timing.get("adaptive_fusion_token_over_pre_frame_token_ratio_mean")
        if isinstance(final_token_ratio, (int, float)) and np.isfinite(final_token_ratio):
            final_token_over_initial_token_ratios.append(float(final_token_ratio))
        frame_token_ratio = timing.get("adaptive_fusion_frame_token_ratio_mean")
        if isinstance(frame_token_ratio, (int, float)) and np.isfinite(frame_token_ratio):
            frame_token_ratios.append(float(frame_token_ratio))
        width, height = timing["resolution"]
        gt_frames = [preprocess_depth(dataset, image, depth, (width, height)) for image, depth in zip(images, gt_paths)]
        pred = np.stack([resize_prediction_to_gt(np.load(path), gt) for path, gt in zip(pred_paths, gt_frames)])
        gt = np.stack(gt_frames)
        row = {"sequence": seq, **depth_metrics(pred, gt, args.align, resolve_max_depth(dataset, args.max_depth))}
        row["frames"] = timing["frames"]
        row["time"] = timing["time"]
        row["fps"] = timing.get("fps", timing["frames"] / timing["time"])
        row["seconds_per_frame"] = timing.get("seconds_per_frame", timing["time"] / timing["frames"])
        row["peak_memory_allocated_gb"] = timing.get("peak_memory_allocated_gb")
        row["peak_memory_reserved_gb"] = timing.get("peak_memory_reserved_gb")
        row["inter_frame_attention"] = timing.get("inter_frame_attention")
        row["register_patch_sample_tokens"] = timing.get("register_patch_sample_tokens")
        row["register_patch_sample_ratio"] = timing.get("register_patch_sample_ratio")
        row["register_patch_sample_mode"] = timing.get("register_patch_sample_mode")
        row["register_patch_merge_sources"] = timing.get("register_patch_merge_sources")
        row["register_patch_merge_protect_first_frame"] = timing.get("register_patch_merge_protect_first_frame")
        row["enable_token_merging"] = timing.get("enable_token_merging")
        row["token_merging_method"] = timing.get("token_merging_method")
        row["token_merging_start"] = timing.get("token_merging_start")
        row["token_merging_ratio"] = timing.get("token_merging_ratio")
        row["token_merging_flashvid_alpha"] = timing.get("token_merging_flashvid_alpha")
        row["token_merging_flashvid_expansion"] = timing.get("token_merging_flashvid_expansion")
        row["token_merging_flashvid_pool_stride"] = timing.get("token_merging_flashvid_pool_stride")
        row["frame_merge_anchor_count"] = timing.get("frame_merge_anchor_count")
        row["frame_merge_anchor_selection"] = timing.get("frame_merge_anchor_selection")
        row["frame_merge_events"] = timing.get("frame_merge_events")
        row["frame_merge_active_frames_mean"] = timing.get("frame_merge_active_frames_mean")
        row["frame_merge_retention_ratio_mean"] = timing.get("frame_merge_retention_ratio_mean")
        row["frame_merge_merge_ratio_mean"] = timing.get("frame_merge_merge_ratio_mean")
        row["frame_merge_raw_merge_ratio_mean"] = timing.get("frame_merge_raw_merge_ratio_mean")
        row["frame_merge_adaptive_policy"] = timing.get("frame_merge_adaptive_policy")
        row["frame_merge_selected_pair_threshold_mean"] = timing.get("frame_merge_selected_pair_threshold_mean")
        row["frame_merge_selected_span_threshold_mean"] = timing.get("frame_merge_selected_span_threshold_mean")
        row["frame_fusion_cuda_ms_mean"] = timing.get("frame_fusion_cuda_ms_mean")
        row["frame_fusion_cuda_ms_total"] = timing.get("frame_fusion_cuda_ms_total")
        row["frame_fusion_host_wall_ms_mean"] = timing.get("frame_fusion_host_wall_ms_mean")
        row["frame_fusion_host_wall_ms_total"] = timing.get("frame_fusion_host_wall_ms_total")
        row["token_merging_active_over_frame_merged_token_ratio_mean"] = timing.get(
            "token_merging_active_over_frame_merged_token_ratio_mean"
        )
        row["token_merging_active_over_frame_original_token_ratio_mean"] = timing.get(
            "token_merging_active_over_frame_original_token_ratio_mean"
        )
        row["token_merging_full_attention_token_ratio_mean"] = timing.get(
            "token_merging_full_attention_token_ratio_mean"
        )
        sequence_rows.append(row)
        print(f"{dataset}/{seq}: Abs Rel={row['Abs Rel']:.4f}, RMSE={row['RMSE']:.4f}, fps={row['fps']:.2f}")

    write_csv(output_root / f"_sequence_metrics_{args.align}.csv", sequence_rows)
    weights = np.asarray([row["valid_pixels"] for row in sequence_rows], dtype=np.float64)
    summary = {
        key: float(np.average([row[key] for row in sequence_rows], weights=weights))
        for key in sequence_rows[0]
        if key
        not in {
            "sequence",
            "frames",
            "valid_pixels",
            "time",
            "fps",
            "seconds_per_frame",
            "peak_memory_allocated_gb",
            "peak_memory_reserved_gb",
            "inter_frame_attention",
            "register_patch_sample_tokens",
            "register_patch_sample_ratio",
            "register_patch_sample_mode",
            "register_patch_merge_sources",
            "register_patch_merge_protect_first_frame",
            "enable_token_merging",
            "token_merging_method",
            "token_merging_start",
            "token_merging_ratio",
            "token_merging_flashvid_alpha",
            "token_merging_flashvid_expansion",
            "token_merging_flashvid_pool_stride",
            "frame_merge_anchor_count",
            "frame_merge_anchor_selection",
            "frame_merge_events",
            "frame_fusion_cuda_ms_mean",
            "frame_fusion_cuda_ms_total",
            "frame_fusion_host_wall_ms_mean",
            "frame_fusion_host_wall_ms_total",
            "depth_eval_protocol",
        }
        and all(isinstance(row.get(key), (int, float)) and row.get(key) is not None for row in sequence_rows)
    }
    summary["depth_eval_protocol"] = sequence_rows[0]["depth_eval_protocol"]
    anchor_counts = {row["frame_merge_anchor_count"] for row in sequence_rows if row["frame_merge_anchor_count"] is not None}
    if len(anchor_counts) == 1:
        summary["frame_merge_anchor_count"] = anchor_counts.pop()
    anchor_selections = {
        row["frame_merge_anchor_selection"]
        for row in sequence_rows
        if row["frame_merge_anchor_selection"] is not None
    }
    if len(anchor_selections) == 1:
        summary["frame_merge_anchor_selection"] = anchor_selections.pop()
    total_frames = int(sum(row["frames"] for row in sequence_rows))
    total_time = float(sum(row["time"] for row in sequence_rows))
    frame_fusion_events = sum(int(row.get("frame_merge_events") or 0) for row in sequence_rows)
    for name in ("frame_fusion_cuda_ms", "frame_fusion_host_wall_ms"):
        totals = [row.get(f"{name}_total") for row in sequence_rows]
        if frame_fusion_events and all(isinstance(value, (int, float)) for value in totals):
            total = float(sum(totals))
            summary[f"{name}_total"] = total
            summary[f"{name}_mean"] = total / frame_fusion_events
    summary["frames"] = total_frames
    summary["time"] = total_time
    summary["fps"] = float(total_frames / total_time) if total_time > 0 else 0.0
    summary["seconds_per_frame"] = total_time / total_frames if total_frames > 0 else None
    allocated_peaks = [row["peak_memory_allocated_gb"] for row in sequence_rows if row["peak_memory_allocated_gb"] is not None]
    reserved_peaks = [row["peak_memory_reserved_gb"] for row in sequence_rows if row["peak_memory_reserved_gb"] is not None]
    summary["peak_memory_allocated_gb"] = float(max(allocated_peaks)) if allocated_peaks else None
    summary["peak_memory_reserved_gb"] = float(max(reserved_peaks)) if reserved_peaks else None
    attention_modes = sorted({row["inter_frame_attention"] for row in sequence_rows if row["inter_frame_attention"]})
    if attention_modes:
        summary["inter_frame_attention"] = ",".join(attention_modes)
    register_sample_tokens = sorted({
        str(row["register_patch_sample_tokens"]) for row in sequence_rows if row["register_patch_sample_tokens"] is not None
    })
    if register_sample_tokens:
        summary["register_patch_sample_tokens"] = ",".join(register_sample_tokens)
    register_sample_ratios = sorted({
        str(row["register_patch_sample_ratio"]) for row in sequence_rows if row["register_patch_sample_ratio"] is not None
    })
    if register_sample_ratios:
        summary["register_patch_sample_ratio"] = ",".join(register_sample_ratios)
    register_sample_modes = sorted({
        row["register_patch_sample_mode"] for row in sequence_rows if row["register_patch_sample_mode"]
    })
    if register_sample_modes:
        summary["register_patch_sample_mode"] = ",".join(register_sample_modes)
    register_merge_sources = sorted({str(row["register_patch_merge_sources"]) for row in sequence_rows})
    summary["register_patch_merge_sources"] = ",".join(register_merge_sources)
    register_merge_protect_first = sorted({
        str(row["register_patch_merge_protect_first_frame"]) for row in sequence_rows
    })
    summary["register_patch_merge_protect_first_frame"] = ",".join(register_merge_protect_first)
    token_merging_values = sorted({str(row["enable_token_merging"]) for row in sequence_rows})
    summary["enable_token_merging"] = ",".join(token_merging_values)
    token_merging_methods = sorted({str(row["token_merging_method"]) for row in sequence_rows if row["token_merging_method"]})
    if token_merging_methods:
        summary["token_merging_method"] = ",".join(token_merging_methods)
    token_merging_ratios = sorted({str(row["token_merging_ratio"]) for row in sequence_rows if row["token_merging_ratio"] is not None})
    if token_merging_ratios:
        summary["token_merging_ratio"] = ",".join(token_merging_ratios)
    token_merging_alphas = sorted({
        str(row["token_merging_flashvid_alpha"])
        for row in sequence_rows
        if row["token_merging_flashvid_alpha"] is not None
    })
    if token_merging_alphas:
        summary["token_merging_flashvid_alpha"] = ",".join(token_merging_alphas)
    token_merging_expansions = sorted({
        str(row["token_merging_flashvid_expansion"])
        for row in sequence_rows
        if row["token_merging_flashvid_expansion"] is not None
    })
    if token_merging_expansions:
        summary["token_merging_flashvid_expansion"] = ",".join(token_merging_expansions)
    token_merging_pool_strides = sorted({
        str(row["token_merging_flashvid_pool_stride"])
        for row in sequence_rows
        if row["token_merging_flashvid_pool_stride"] is not None
    })
    if token_merging_pool_strides:
        summary["token_merging_flashvid_pool_stride"] = ",".join(token_merging_pool_strides)
    summary["valid_pixels"] = int(weights.sum())
    summary["adaptive_fusion_final_token_over_initial_token_ratio_sequence_mean"] = (
        float(np.mean(final_token_over_initial_token_ratios))
        if final_token_over_initial_token_ratios
        else None
    )
    summary["adaptive_fusion_final_token_over_initial_token_ratio_sequence_count"] = len(
        final_token_over_initial_token_ratios
    )
    summary["adaptive_fusion_frame_token_ratio_sequence_mean"] = (
        float(np.mean(frame_token_ratios)) if frame_token_ratios else None
    )
    summary["adaptive_fusion_frame_token_ratio_sequence_count"] = len(frame_token_ratios)
    write_csv(output_root / f"_summary_{args.align}.csv", [summary])
    write_json(output_root / f"_summary_{args.align}.json", summary)
    pose_rows = []
    pred_root = Path(args.pred_dir) / dataset
    for seq in sequences:
        pose_path = pred_root / seq / "_pose_auc.json"
        pred_pose_path = pred_root / seq / "pred_poses.npy"
        if dataset == "bonn":
            association_path = pred_root / seq / "_bonn_associations.json"
            with association_path.open() as handle:
                bonn_manifest = json.load(handle)
            images, _ = association_paths(
                dataset_root,
                seq,
                bonn_manifest["associations"],
                rgb_dir=bonn_manifest["rgb_dir"],
                depth_dir=bonn_manifest["depth_dir"],
            )
            pose_indices = [item["pose_index"] for item in bonn_manifest["associations"]]
        else:
            source_images = sequence_images(dataset, dataset_root, seq, args.bonn_rgb_dir)
            images, _ = load_manifest_selected_images(pred_root / seq, source_images)
            pose_indices = None
        gt_poses = sequence_poses(dataset, dataset_root, seq, len(images), images, pose_indices=pose_indices)
        if pred_pose_path.is_file() and gt_poses is not None:
            pred_poses = np.load(pred_pose_path)
            row = evaluate_pose_auc(
                pred_poses,
                gt_poses,
                num_frames=args.pose_eval_frames,
                seed=args.pose_eval_seed,
                sequence=seq,
            )
            write_json(pose_path, row)
            row["sequence"] = seq
            pose_rows.append(row)
        elif pose_path.is_file():
            with pose_path.open() as handle:
                row = json.load(handle)
            row["sequence"] = seq
            pose_rows.append(row)
    pose_summary = summarize_pose_auc(pose_rows) if pose_rows else None
    if pose_summary is not None:
        write_json(output_root / "_summary_pose_auc.json", pose_summary)
    complete = {"video_depth": summary, "pose_auc": pose_summary, "speed": {
        "frames": summary.get("frames"),
        "time": summary.get("time"),
        "fps": summary.get("fps"),
        "seconds_per_frame": summary.get("seconds_per_frame"),
        "peak_memory_allocated_gb": summary.get("peak_memory_allocated_gb"),
        "peak_memory_reserved_gb": summary.get("peak_memory_reserved_gb"),
        "frame_merge_active_frames_mean": summary.get("frame_merge_active_frames_mean"),
        "frame_merge_retention_ratio_mean": summary.get("frame_merge_retention_ratio_mean"),
        "frame_merge_merge_ratio_mean": summary.get("frame_merge_merge_ratio_mean"),
        "frame_merge_raw_merge_ratio_mean": summary.get("frame_merge_raw_merge_ratio_mean"),
        "frame_merge_adaptive_policy": summary.get("frame_merge_adaptive_policy"),
        "frame_merge_selected_pair_threshold_mean": summary.get("frame_merge_selected_pair_threshold_mean"),
        "frame_merge_selected_span_threshold_mean": summary.get("frame_merge_selected_span_threshold_mean"),
        "frame_merge_anchor_count": summary.get("frame_merge_anchor_count"),
        "frame_merge_anchor_selection": summary.get("frame_merge_anchor_selection"),
        "frame_fusion_cuda_ms_mean": summary.get("frame_fusion_cuda_ms_mean"),
        "frame_fusion_cuda_ms_total": summary.get("frame_fusion_cuda_ms_total"),
        "frame_fusion_host_wall_ms_mean": summary.get("frame_fusion_host_wall_ms_mean"),
        "frame_fusion_host_wall_ms_total": summary.get("frame_fusion_host_wall_ms_total"),
        "token_merging_active_over_frame_merged_token_ratio_mean": summary.get(
            "token_merging_active_over_frame_merged_token_ratio_mean"
        ),
        "token_merging_active_over_frame_original_token_ratio_mean": summary.get(
            "token_merging_active_over_frame_original_token_ratio_mean"
        ),
        "token_merging_full_attention_token_ratio_mean": summary.get(
            "token_merging_full_attention_token_ratio_mean"
        ),
        "adaptive_fusion_final_token_over_initial_token_ratio_sequence_mean": summary.get(
            "adaptive_fusion_final_token_over_initial_token_ratio_sequence_mean"
        ),
        "adaptive_fusion_final_token_over_initial_token_ratio_sequence_count": summary.get(
            "adaptive_fusion_final_token_over_initial_token_ratio_sequence_count"
        ),
        "adaptive_fusion_frame_token_ratio_sequence_mean": summary.get(
            "adaptive_fusion_frame_token_ratio_sequence_mean"
        ),
        "adaptive_fusion_frame_token_ratio_sequence_count": summary.get(
            "adaptive_fusion_frame_token_ratio_sequence_count"
        ),
    }}
    write_json(output_root / f"_summary_complete_{args.align}.json", complete)
    print(f"{dataset} {args.align}: {summary}")
    return {"dataset": dataset, "scenes": len(sequence_rows), **summary}


def main() -> None:
    args = parse_args()
    if args.dataset == "all" and args.dataset_root:
        raise ValueError("--dataset-root cannot be used with --dataset all; use the default dataset roots.")
    datasets = ["sintel", "bonn", "7scenes", "tum_dynamic", "nrgbd"] if args.dataset == "all" else [args.dataset]
    summaries = [evaluate_dataset(args, dataset) for dataset in datasets]
    if len(summaries) > 1:
        summary_path = Path(args.output_dir) / f"video-depth-summary-{args.align}.csv"
        write_csv(summary_path, summaries)
        print(f"Combined summary written to {summary_path}")


if __name__ == "__main__":
    main()
