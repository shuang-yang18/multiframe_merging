"""Evaluate VGGT-Omega video-depth predictions on supported benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from PIL import Image

from vggt_omega.evaluation import read_bonn_depth, read_sintel_depth, write_csv

try:
    import cv2
except ImportError:  # pragma: no cover - runtime fallback for minimal environments.
    cv2 = None

if __package__:
    from .infer import DEFAULT_DATASET_ROOTS, sequence_images, sequence_names, sequence_poses
    from .pose_auc import evaluate_pose_auc, summarize_pose_auc, write_json
else:
    from infer import DEFAULT_DATASET_ROOTS, sequence_images, sequence_names, sequence_poses
    from pose_auc import evaluate_pose_auc, summarize_pose_auc, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["sintel", "bonn", "7scenes", "tum_dynamic", "all"], default="all")
    parser.add_argument("--dataset-root")
    parser.add_argument("--bonn-rgb-dir", default="rgb_110", help="Bonn RGB subdirectory; use 'rgb' for full-length sequences.")
    parser.add_argument("--bonn-depth-dir", default="depth_110", help="Bonn depth subdirectory; use 'depth' for full-length sequences.")
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
    parser.add_argument("--max-frames-per-seq", type=int, default=0)
    parser.add_argument(
        "--frame-sample-mode",
        choices=["first", "random"],
        default="first",
        help="How to choose frames when --max-frames-per-seq is set.",
    )
    parser.add_argument("--frame-sample-seed", type=int, default=0, help="Seed for random frame input sampling.")
    return parser.parse_args()


def _sequence_seed(seed: int, sequence: str | None) -> int:
    if not sequence:
        return seed
    digest = hashlib.sha1(sequence.encode("utf-8")).digest()
    return (seed + int.from_bytes(digest[:4], "little")) % (2**32)


def limit_frames(
    paths: list[str],
    max_frames: int,
    sample_mode: str = "first",
    seed: int = 0,
    sequence: str | None = None,
) -> list[str]:
    if not max_frames or max_frames <= 0 or len(paths) <= max_frames:
        return paths
    if sample_mode == "random":
        rng = np.random.default_rng(_sequence_seed(seed, sequence))
        indices = sorted(rng.choice(len(paths), size=max_frames, replace=False).tolist())
        return [paths[index] for index in indices]
    return paths[:max_frames]


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


def sequence_depths(
    dataset: str,
    dataset_root: str | Path,
    seq: str,
    bonn_depth_dir: str = "depth_110",
    bonn_rgb_dir: str = "rgb_110",
) -> list[str]:
    if dataset == "sintel":
        paths = sorted((Path(dataset_root) / "depth" / seq).glob("*.dpt"))
    elif dataset == "7scenes":
        paths = sorted((Path(dataset_root) / seq).glob("*.depth.png"))
    elif dataset == "tum_dynamic":
        return _associate_tum_depths(dataset_root, seq, sequence_images(dataset, dataset_root, seq, bonn_rgb_dir))
    else:
        paths = sorted((Path(dataset_root) / seq / bonn_depth_dir).glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No {dataset} ground truth depths found for sequence {seq}")
    return [str(path) for path in paths]


def preprocess_depth(
    dataset: str, image_filename: str, depth_filename: str, resolution: tuple[int, int]
) -> np.ndarray:
    if dataset == "sintel":
        return read_sintel_depth(depth_filename).astype(np.float32)
    if dataset == "7scenes":
        raw = np.asarray(Image.open(depth_filename))
        depth = raw.astype(np.float32) / 1000.0
        depth[(raw == 0) | (raw == 65535)] = -1.0
        return depth
    if dataset == "tum_dynamic":
        return read_png_depth(depth_filename, 5000.0)
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
        scale = np.mean(gt) / max(np.mean(pred), 1e-8)
        for _ in range(10):
            residual = pred * scale - gt
            weights = 1.0 / np.maximum(np.abs(residual), 1e-8)
            denom = np.sum(weights * pred * pred)
            if denom <= 1e-12:
                break
            scale = max(float(np.sum(weights * pred * gt) / denom), 1e-3)
        return pred * scale

    scale = np.median(gt) / max(np.median(pred), 1e-8)
    shift = 0.0
    m_scale = v_scale = 0.0
    m_shift = v_shift = 0.0
    beta1 = 0.9
    beta2 = 0.999
    lr = 1e-4
    eps = 1e-8
    inv_n = 1.0 / max(pred.size, 1)
    for step in range(1, 1001):
        residual = pred * scale + shift - gt
        signed = np.sign(residual)
        grad_scale = float(np.sum(signed * pred) * inv_n)
        grad_shift = float(np.sum(signed) * inv_n)
        m_scale = beta1 * m_scale + (1.0 - beta1) * grad_scale
        v_scale = beta2 * v_scale + (1.0 - beta2) * grad_scale * grad_scale
        m_shift = beta1 * m_shift + (1.0 - beta1) * grad_shift
        v_shift = beta2 * v_shift + (1.0 - beta2) * grad_shift * grad_shift
        scale -= lr * (m_scale / (1.0 - beta1**step)) / (np.sqrt(v_scale / (1.0 - beta2**step)) + eps)
        shift -= lr * (m_shift / (1.0 - beta1**step)) / (np.sqrt(v_shift / (1.0 - beta2**step)) + eps)
    return scale * pred + shift


def depth_metrics(prediction: np.ndarray, ground_truth: np.ndarray, mode: str, max_depth: float) -> dict:
    mask = np.isfinite(ground_truth) & (ground_truth > 0) & (ground_truth < max_depth)
    pred = prediction[mask].astype(np.float64)
    gt = ground_truth[mask].astype(np.float64)
    if pred.size == 0:
        raise ValueError("No valid depth pixels available for evaluation.")
    pred = align_depth(pred, gt, mode)
    log_pred = np.maximum(pred, 1e-5)
    ratio = np.maximum(pred / gt, gt / pred)
    return {
        "depth_eval_protocol": "pi3_videodepth",
        "Abs Rel": float(np.mean(np.abs(pred - gt) / gt)),
        "Sq Rel": float(np.mean((pred - gt) ** 2 / gt)),
        "RMSE": float(np.sqrt(np.mean((pred - gt) ** 2))),
        "Log RMSE": float(np.sqrt(np.mean((np.log(log_pred) - np.log(gt)) ** 2))),
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
    )
    pred_root = Path(args.pred_dir) / dataset
    sequence_rows = []

    for seq in sequences:
        images = limit_frames(
            sequence_images(dataset, dataset_root, seq, args.bonn_rgb_dir),
            args.max_frames_per_seq,
            args.frame_sample_mode,
            args.frame_sample_seed,
            seq,
        )
        gt_paths = limit_frames(
            sequence_depths(dataset, dataset_root, seq, args.bonn_depth_dir, args.bonn_rgb_dir),
            args.max_frames_per_seq,
            args.frame_sample_mode,
            args.frame_sample_seed,
            seq,
        )
        pred_paths = sorted((pred_root / seq).glob("frame_*.npy"))
        if not (len(pred_paths) == len(gt_paths) == len(images)):
            raise ValueError(
                f"{dataset}/{seq}: found {len(pred_paths)} predictions, "
                f"{len(gt_paths)} ground truth depths, and {len(images)} images"
            )
        with (pred_root / seq / "_time.json").open() as handle:
            timing = json.load(handle)
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
        row["token_merging_tstm_threshold"] = timing.get("token_merging_tstm_threshold")
        row["token_merging_tstm_neighbor_size"] = timing.get("token_merging_tstm_neighbor_size")
        row["token_merging_flashvid_alpha"] = timing.get("token_merging_flashvid_alpha")
        row["token_merging_flashvid_expansion"] = timing.get("token_merging_flashvid_expansion")
        row["token_merging_flashvid_pool_stride"] = timing.get("token_merging_flashvid_pool_stride")
        sequence_rows.append(row)
        print(f"{dataset}/{seq}: Abs Rel={row['Abs Rel']:.4f}, RMSE={row['RMSE']:.4f}, fps={row['fps']:.2f}")

    write_csv(Path(args.output_dir) / f"{dataset}-sequences-{args.align}.csv", sequence_rows)
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
            "token_merging_tstm_threshold",
            "token_merging_tstm_neighbor_size",
            "token_merging_flashvid_alpha",
            "token_merging_flashvid_expansion",
            "token_merging_flashvid_pool_stride",
            "depth_eval_protocol",
        }
    }
    total_frames = int(sum(row["frames"] for row in sequence_rows))
    total_time = float(sum(row["time"] for row in sequence_rows))
    summary["frames"] = total_frames
    summary["time"] = total_time
    summary["fps"] = float(np.average([row["fps"] for row in sequence_rows]))
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
    token_merging_thresholds = sorted({
        str(row["token_merging_tstm_threshold"])
        for row in sequence_rows
        if row["token_merging_tstm_threshold"] is not None
    })
    if token_merging_thresholds:
        summary["token_merging_tstm_threshold"] = ",".join(token_merging_thresholds)
    token_merging_neighbors = sorted({
        str(row["token_merging_tstm_neighbor_size"])
        for row in sequence_rows
        if row["token_merging_tstm_neighbor_size"] is not None
    })
    if token_merging_neighbors:
        summary["token_merging_tstm_neighbor_size"] = ",".join(token_merging_neighbors)
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
    write_csv(Path(args.output_dir) / f"{dataset}-metric-{args.align}.csv", [summary])
    pose_rows = []
    pred_root = Path(args.pred_dir) / dataset
    for seq in sequences:
        pose_path = pred_root / seq / "_pose_auc.json"
        pred_pose_path = pred_root / seq / "pred_poses.npy"
        images = limit_frames(
            sequence_images(dataset, dataset_root, seq, args.bonn_rgb_dir),
            args.max_frames_per_seq,
            args.frame_sample_mode,
            args.frame_sample_seed,
            seq,
        )
        gt_poses = sequence_poses(dataset, dataset_root, seq, len(images), images)
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
        write_json(Path(args.output_dir) / f"{dataset}-pose-auc.json", pose_summary)
    complete = {"video_depth": summary, "pose_auc": pose_summary, "speed": {
        "frames": summary.get("frames"),
        "time": summary.get("time"),
        "fps": summary.get("fps"),
        "seconds_per_frame": summary.get("seconds_per_frame"),
        "peak_memory_allocated_gb": summary.get("peak_memory_allocated_gb"),
        "peak_memory_reserved_gb": summary.get("peak_memory_reserved_gb"),
    }}
    write_json(Path(args.output_dir) / f"{dataset}-complete-{args.align}.json", complete)
    print(f"{dataset} {args.align}: {summary}")
    return {"dataset": dataset, "scenes": len(sequence_rows), **summary}


def main() -> None:
    args = parse_args()
    if args.dataset == "all" and args.dataset_root:
        raise ValueError("--dataset-root cannot be used with --dataset all; use the default dataset roots.")
    datasets = ["sintel", "bonn", "7scenes", "tum_dynamic"] if args.dataset == "all" else [args.dataset]
    summaries = [evaluate_dataset(args, dataset) for dataset in datasets]
    if len(summaries) > 1:
        summary_path = Path(args.output_dir) / f"video-depth-summary-{args.align}.csv"
        write_csv(summary_path, summaries)
        print(f"Combined summary written to {summary_path}")


if __name__ == "__main__":
    main()
