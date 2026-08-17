#!/usr/bin/env python3
"""Pi3 inference and evaluation using the VGGT-Omega video-depth protocol.

The model forward path is Pi3's regular full-video inference.  Evaluation is
performed in the same process with VGGT-Omega's robust sequence-level
scale-shift depth alignment and official relative-pose AUC implementation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import zlib
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SHARED_ROOT = Path("/data/mmc_syang")
if str(SHARED_ROOT) not in sys.path:
    sys.path.insert(0, str(SHARED_ROOT))
from geometry_eval import depth_to_world_points, evaluate_pi3_geometry, scaled_intrinsics

from pi3.models.pi3 import Pi3
from utils.interfaces import load_and_resize14


DEFAULT_ROOTS = {
    "tum_dynamic": ROOT.parent / "dataset" / "TUM-Dynamics",
    "7scenes": ROOT.parent / "dataset" / "7scenes",
}
# Match VGGT-Omega's standard TUM-Dynamics/7Scenes depth evaluation bounds.
MAX_DEPTHS = {"tum_dynamic": 10.0, "7scenes": 10.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["tum_dynamic", "7scenes", "all"], default="all")
    parser.add_argument("--dataset-root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pretrained", default="checkpoints/Pi3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-frames-per-seq", type=int, default=300)
    parser.add_argument("--frame-sample-mode", choices=["uniform", "first", "random"], default="uniform")
    parser.add_argument("--frame-sample-seed", type=int, default=0)
    parser.add_argument("--sampling-pool-frames", type=int, default=0,
                        help="Uniformly form this many source candidates, then use the first --max-frames-per-seq; short sequences use source first frames.")
    parser.add_argument("--pose-eval-frames", type=int, default=0)
    parser.add_argument("--pose-eval-seed", type=int, default=0)
    parser.add_argument("--timing-repeats", type=int, default=3,
                        help="Formal CUDA Event timing repeats after one untimed warmup.")
    parser.add_argument("--load-img-size", type=int, default=512)
    parser.add_argument("--seven-scenes-split", choices=["test", "train", "all"], default="test")
    parser.add_argument("--sequences", nargs="*")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument(
        "--acceleration-method",
        choices=["none", "fastvggt", "sparse-vggt", "da-vggt", "u-m"],
        default="none",
        help=(
            "Unified method selector. fastvggt and sparse-vggt are native Pi3 "
            "adapters. DA-VGGT and U-M use Pi3's sequence/global-attention "
            "adapters."
        ),
    )
    parser.add_argument("--token-merging-method", choices=["none", "fastvggt", "frame_persistent_spatial"], default="none",
                        help="Legacy Pi3 selector; cannot be combined with --acceleration-method.")
    parser.add_argument("--token-merging-ratio", type=float, default=0.9)
    parser.add_argument("--token-merging-frame-alpha", type=float, default=0.1)
    parser.add_argument("--token-merging-frame-segment-threshold", type=float, default=0.9)
    parser.add_argument("--token-merging-frame-merge-threshold", type=float, default=0.1)
    parser.add_argument("--token-merging-frame-max-window", type=int, default=20)
    parser.add_argument("--token-merging-frame-pool-stride", type=int, default=2)
    parser.add_argument("--token-merging-frame-multi-max-group-size", type=int, default=4)
    parser.add_argument("--token-merging-frame-multi-pair-threshold", type=float, default=0.95)
    parser.add_argument("--token-merging-frame-multi-span-threshold", type=float, default=0.93)
    parser.add_argument("--sparse-vggt-sparse-ratio", type=float, default=0.5)
    parser.add_argument("--sparse-vggt-cdf-threshold", type=float, default=None)
    parser.add_argument("--sparse-vggt-pool-mode", choices=["avg", "max"], default="avg")
    parser.add_argument("--um-lambda", type=float, default=0.03)
    parser.add_argument("--um-spatial-radius", type=int, default=2)
    parser.add_argument("--um-temporal-window", type=int, default=4)
    parser.add_argument("--um-refresh-layers", default="0,9")
    parser.add_argument("--attention-probe-output", default=None,
                        help="Optional directory: export full-token global-attention TIFFs for one forward; disabled by default.")
    parser.add_argument("--attention-probe-query-chunk", type=int, default=128)
    parser.add_argument("--attention-probe-frame-gap", type=int, default=None,
                        help="When probing, select exactly 10 consecutive sampled positions at this source-frame gap.")
    parser.add_argument("--da-chunk-size", type=int, default=50)
    return parser.parse_args()


def parse_7scenes_split_line(line: str) -> str | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("sequence") and line.removeprefix("sequence").isdigit():
        return f"seq-{int(line.removeprefix('sequence')):02d}"
    if line.startswith("seq-"):
        return line
    raise ValueError(f"Unsupported 7Scenes split entry: {line!r}")


def read_7scenes_split(scene_dir: Path, split: str) -> list[str]:
    if split == "all":
        return sorted(path.name for path in scene_dir.glob("seq-*") if path.is_dir())
    split_path = scene_dir / ("TestSplit.txt" if split == "test" else "TrainSplit.txt")
    if not split_path.is_file():
        raise FileNotFoundError(f"Missing split file: {split_path}")
    return [name for line in split_path.read_text().splitlines() if (name := parse_7scenes_split_line(line))]


def list_sequences(dataset: str, root: Path, args: argparse.Namespace) -> list[str]:
    if args.sequences:
        sequences = list(args.sequences)
    elif dataset == "tum_dynamic":
        sequences = sorted(path.name for path in root.iterdir() if (path / "rgb").is_dir())
    else:
        sequences = []
        for scene in sorted(path for path in root.iterdir() if path.is_dir() and path.name not in {"train", "test"}):
            sequences.extend(f"{scene.name}/{name}" for name in read_7scenes_split(scene, args.seven_scenes_split))
    missing = [seq for seq in sequences if not sequence_dir(dataset, root, seq).is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing sequences below {root}: {missing}")
    return sequences


def sequence_dir(dataset: str, root: Path, sequence: str) -> Path:
    return root / sequence


def sequence_images(dataset: str, root: Path, sequence: str) -> list[Path]:
    path = sequence_dir(dataset, root, sequence)
    images = sorted((path / "rgb").glob("*.png")) if dataset == "tum_dynamic" else sorted(path.glob("*.color.png"))
    if not images:
        raise FileNotFoundError(f"No RGB images for {dataset}/{sequence}")
    return images


def timestamp(path: Path) -> float:
    return float(path.stem)


def tum_depth_paths(root: Path, sequence: str, images: list[Path]) -> list[Path]:
    entries = []
    with (root / sequence / "depth.txt").open() as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                stamp, relative = line.split()[:2]
                entries.append((float(stamp), root / sequence / relative))
    if not entries:
        raise FileNotFoundError(f"No depth entries for TUM sequence {sequence}")
    depth_times = np.asarray([item[0] for item in entries])
    return [entries[int(np.argmin(np.abs(depth_times - timestamp(image))))][1] for image in images]


def sequence_depths(dataset: str, root: Path, sequence: str, images: list[Path]) -> list[Path]:
    if dataset == "tum_dynamic":
        return tum_depth_paths(root, sequence, images)
    depths = []
    for image in images:
        projected = image.with_name(image.name.replace(".color.png", ".depth.proj.png"))
        raw = image.with_name(image.name.replace(".color.png", ".depth.png"))
        depths.append(projected if projected.is_file() else raw)
    if any(not path.is_file() for path in depths):
        raise FileNotFoundError(f"Missing 7Scenes depth files for {sequence}")
    return depths


def tum_row_to_c2w(row: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = Rotation.from_quat(row[4:8]).as_matrix()
    pose[:3, 3] = row[1:4]
    return pose


def sequence_poses(dataset: str, root: Path, sequence: str, images: list[Path]) -> np.ndarray:
    if dataset == "7scenes":
        return np.stack([np.loadtxt(image.with_name(image.name.replace(".color.png", ".pose.txt"))) for image in images])
    values = np.atleast_2d(np.loadtxt(root / sequence / "groundtruth.txt", comments="#"))
    pose_times = values[:, 0]
    poses = np.stack([tum_row_to_c2w(row) for row in values])
    indices = [int(np.argmin(np.abs(pose_times - timestamp(image)))) for image in images]
    return poses[np.asarray(indices, dtype=np.int64)]


def limit_frames(paths: list[Path], max_frames: int, mode: str, seed: int, sequence: str) -> list[Path]:
    if max_frames <= 0 or len(paths) <= max_frames:
        return paths
    if mode == "first":
        return paths[:max_frames]
    if mode == "uniform":
        indices = np.linspace(0, len(paths) - 1, max_frames, dtype=np.int64)
        return [paths[int(index)] for index in indices]
    digest = int.from_bytes(hashlib.sha1(sequence.encode()).digest()[:4], "little")
    indices = np.sort(np.random.default_rng(seed + digest).choice(len(paths), size=max_frames, replace=False))
    return [paths[int(index)] for index in indices]


def compact_input_frames(paths: list[Path], input_frames: int, sampling_pool_frames: int) -> tuple[list[Path], list[int], str]:
    """Shared compact protocol: uniform pool first, then its leading input budget."""
    if sampling_pool_frames <= 0:
        selected = limit_frames(paths, input_frames, "uniform", 0, "")
        indices = np.rint(np.linspace(0, len(paths) - 1, len(selected))).astype(np.int64).tolist() if selected else []
        return selected, indices, "uniform"
    if len(paths) < sampling_pool_frames:
        return paths[:input_frames], list(range(min(input_frames, len(paths)))), "short_sequence_first"
    pool_indices = np.rint(np.linspace(0, len(paths) - 1, sampling_pool_frames)).astype(np.int64)
    indices = pool_indices[:input_frames].tolist()
    return [paths[int(index)] for index in indices], indices, "uniform_pool_then_first"


def read_depth(dataset: str, path: Path) -> np.ndarray:
    raw = np.asarray(Image.open(path))
    if dataset == "tum_dynamic":
        depth = raw.astype(np.float32) / 5000.0
        depth[raw == 0] = -1.0
    else:
        depth = raw.astype(np.float32) / 1000.0
        depth[(raw == 0) | (raw == 65535)] = -1.0
    return depth


def geometry_intrinsic(dataset: str) -> np.ndarray:
    # Matches Pi3's legacy 7Scenes/NRGBD geometry evaluator.
    fx = fy = 525.0 if dataset == "7scenes" else 554.2562584220408
    return np.array([[fx, 0.0, 320.0], [0.0, fy, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def geometry_from_prediction(dataset: str, prediction: dict, gt_depth: np.ndarray, gt_c2w: np.ndarray) -> dict[str, float | int]:
    """Pi3 legacy geometry protocol on a deterministic 100k-point budget."""
    if "points" not in prediction:
        raise KeyError("Pi3 prediction lacks global 'points' required for geometry evaluation")
    pred_points = prediction["points"][0].detach().float().cpu().numpy()
    height, width = pred_points.shape[1:3]
    gt_small = np.stack([cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST) for frame in gt_depth])
    gt_k = scaled_intrinsics(geometry_intrinsic(dataset), (gt_depth.shape[1], gt_depth.shape[2]), (height, width), len(gt_depth))
    gt_points = depth_to_world_points(gt_small, gt_c2w, gt_k)
    valid = np.isfinite(gt_small) & (gt_small > 0) & (gt_small < MAX_DEPTHS[dataset])
    return evaluate_pi3_geometry(pred_points, gt_points, valid)


def align_depth(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred64, gt64 = pred.astype(np.float64, copy=False), gt.astype(np.float64, copy=False)
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
            params = np.linalg.lstsq(design * np.sqrt(weights)[:, None], fit_gt * np.sqrt(weights), rcond=None)[0]
        else:
            scale = np.dot(weights, centered_pred * (fit_gt - mean_gt)) / variance
            params = np.array([scale, mean_gt - scale * mean_pred])
    return (params[0] * pred64 + params[1]).astype(np.float32)


def depth_metrics(prediction: np.ndarray, ground_truth: np.ndarray, max_depth: float) -> dict[str, float | int | str]:
    mask = np.isfinite(prediction) & np.isfinite(ground_truth) & (ground_truth > 0) & (ground_truth < max_depth)
    pred, gt = prediction[mask].astype(np.float32), ground_truth[mask].astype(np.float32)
    if not len(pred):
        raise ValueError("No valid depth pixels available for evaluation.")
    pred = np.clip(align_depth(pred, gt), 1e-5, max_depth)
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
        "valid_pixels": int(len(pred)),
    }


def rotation_error_deg(reference: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    difference = reference @ np.transpose(prediction, (0, 2, 1))
    cosine = np.clip((np.trace(difference, axis1=1, axis2=2) - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def translation_error_deg(reference: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    reference = reference / np.maximum(np.linalg.norm(reference, axis=1, keepdims=True), 1e-15)
    prediction = prediction / np.maximum(np.linalg.norm(prediction, axis=1, keepdims=True), 1e-15)
    cosine = np.clip(np.abs(np.sum(reference * prediction, axis=1)), 0.0, 1.0)
    return np.degrees(np.arccos(cosine))


def pose_auc(errors: np.ndarray, threshold: int) -> float:
    hist, _ = np.histogram(errors, bins=np.arange(threshold + 1))
    return float(np.mean(np.cumsum(hist.astype(np.float64) / len(errors))) * 100.0) if len(errors) else 0.0


def sampled_indices(count: int, num_frames: int, seed: int, sequence: str) -> np.ndarray:
    if num_frames <= 0 or count <= num_frames:
        return np.arange(count, dtype=np.int64)
    return np.sort(np.random.default_rng(seed ^ zlib.crc32(sequence.encode())).choice(count, size=num_frames, replace=False))


def evaluate_pose_auc(pred_c2w: np.ndarray, gt_c2w: np.ndarray, args: argparse.Namespace, sequence: str) -> dict:
    count = min(len(pred_c2w), len(gt_c2w))
    indices = sampled_indices(count, args.pose_eval_frames, args.pose_eval_seed, sequence)
    pred_w2c = np.linalg.inv(np.asarray(pred_c2w, dtype=np.float64)[indices])
    gt_w2c = np.linalg.inv(np.asarray(gt_c2w, dtype=np.float64)[indices])
    first, second = np.triu_indices(len(indices), k=1)
    pred_rel = pred_w2c[first] @ np.linalg.inv(pred_w2c[second])
    gt_rel = gt_w2c[first] @ np.linalg.inv(gt_w2c[second])
    rotation_errors = rotation_error_deg(gt_rel[:, :3, :3], pred_rel[:, :3, :3])
    translation_errors = translation_error_deg(gt_rel[:, :3, 3], pred_rel[:, :3, 3])
    errors = np.maximum(rotation_errors, translation_errors)
    return {
        "AUC@3": pose_auc(errors, 3), "AUC@15": pose_auc(errors, 15), "AUC@30": pose_auc(errors, 30),
        "frames": int(len(indices)), "pairs": int(len(errors)),
        "sampled_frame_indices": [int(index) for index in indices],
        "pose_eval_frames": int(args.pose_eval_frames), "pose_eval_seed": int(args.pose_eval_seed),
        "pose_auc_protocol": "vggt_official_relative_pose_auc",
        "pose_convention": "input_c2w_converted_to_w2c",
        "relative_pose_pairs": "all_i_less_than_j",
        "auc_integration": "histogram_cumsum_1deg_bins",
        "sim3_alignment": False, "translation_angle_ambiguity": True,
        "mean_pose_error_deg": float(np.mean(errors)), "median_pose_error_deg": float(np.median(errors)),
        "mean_rotation_error_deg": float(np.mean(rotation_errors)),
        "mean_translation_error_deg": float(np.mean(translation_errors)),
        "pose_errors_deg": errors.tolist(), "rotation_errors_deg": rotation_errors.tolist(),
        "translation_errors_deg": translation_errors.tolist(),
    }


def summarize_pose(rows: list[dict]) -> dict:
    errors = np.asarray([value for row in rows for value in row["pose_errors_deg"]], dtype=np.float64)
    rotations = np.asarray([value for row in rows for value in row["rotation_errors_deg"]], dtype=np.float64)
    translations = np.asarray([value for row in rows for value in row["translation_errors_deg"]], dtype=np.float64)
    return {
        "AUC@3": pose_auc(errors, 3), "AUC@15": pose_auc(errors, 15), "AUC@30": pose_auc(errors, 30), "sequences": len(rows), "pairs": int(len(errors)),
        "mean_pose_error_deg": float(np.mean(errors)), "median_pose_error_deg": float(np.median(errors)),
        "mean_rotation_error_deg": float(np.mean(rotations)), "mean_translation_error_deg": float(np.mean(translations)),
        "pose_eval_frames": rows[0]["pose_eval_frames"], "pose_eval_seed": rows[0]["pose_eval_seed"],
        "pose_auc_protocol": "vggt_official_relative_pose_auc", "pose_convention": "input_c2w_converted_to_w2c",
        "relative_pose_pairs": "all_i_less_than_j", "auc_integration": "histogram_cumsum_1deg_bins",
        "sim3_alignment": False, "translation_angle_ambiguity": True,
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_model(args: argparse.Namespace, device: torch.device) -> Pi3:
    if args.acceleration_method != "none" and args.token_merging_method != "none":
        raise ValueError("Use either --acceleration-method or --token-merging-method, not both")
    token_method = "fastvggt" if args.acceleration_method == "fastvggt" else args.token_merging_method
    enabled = token_method != "none"
    model = Pi3.from_pretrained(
        args.pretrained, enable_token_merging=enabled, token_merging_method=token_method,
        token_merging_ratio=args.token_merging_ratio, token_merging_frame_alpha=args.token_merging_frame_alpha,
        token_merging_frame_segment_threshold=args.token_merging_frame_segment_threshold,
        token_merging_frame_merge_threshold=args.token_merging_frame_merge_threshold,
        token_merging_frame_max_window=args.token_merging_frame_max_window,
        token_merging_frame_pool_stride=args.token_merging_frame_pool_stride,
        token_merging_frame_multi_max_group_size=args.token_merging_frame_multi_max_group_size,
        token_merging_frame_multi_pair_threshold=args.token_merging_frame_multi_pair_threshold,
        token_merging_frame_multi_span_threshold=args.token_merging_frame_multi_span_threshold,
        um_lambda_cost=args.um_lambda if args.acceleration_method == "u-m" else None,
        um_spatial_radius=args.um_spatial_radius,
        um_temporal_window=args.um_temporal_window,
        um_refresh_layers=args.um_refresh_layers,
    ).to(device).eval()
    if args.acceleration_method == "sparse-vggt":
        sparse_root = Path("/data/mmc_syang/sparse-vggt/src")
        sparge_root = Path("/data/mmc_syang/sparse-vggt/external/SpargeAttn")
        sparse_vggt_root = Path("/data/mmc_syang/sparse-vggt/external/vggt")
        if not sparse_root.is_dir():
            raise FileNotFoundError(f"Sparse-VGGT source not found: {sparse_root}")
        if not sparge_root.is_dir():
            raise FileNotFoundError(f"SpargeAttn source not found: {sparge_root}")
        if not sparse_vggt_root.is_dir():
            raise FileNotFoundError(f"Sparse-VGGT's vendored VGGT source not found: {sparse_vggt_root}")
        sys.path.insert(0, str(sparse_root))
        sys.path.insert(0, str(sparge_root))
        # sparse_vggt.__init__ exports both VGGT and Pi3 adapters, so its
        # eager VGGT import must resolve even when this caller uses Pi3 only.
        sys.path.insert(0, str(sparse_vggt_root))
        from sparse_vggt.models.pi3 import sparse_model_from_pi3

        model, _ = sparse_model_from_pi3(
            model,
            sparse_ratio=args.sparse_vggt_sparse_ratio,
            cdf_threshold=args.sparse_vggt_cdf_threshold,
            pool_mode=args.sparse_vggt_pool_mode,
            verbose=True,
        )
    return model


def forward_da(model: Pi3, tensor: torch.Tensor, chunk_size: int) -> dict:
    """Full DA-VGGT: cached encoder tokens and pose-weighted re-chunking."""
    return model.forward_da_vggt(tensor, chunk_size=chunk_size)


def run_dataset(args: argparse.Namespace, model: Pi3, device: torch.device, dataset: str) -> None:
    root = Path(args.dataset_root) if args.dataset_root else DEFAULT_ROOTS[dataset]
    output_root = Path(args.output_dir) / dataset
    output_root.mkdir(parents=True, exist_ok=True)
    depth_rows, pose_rows = [], []
    total_time = 0.0
    total_frames = 0
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16

    for sequence in tqdm(list_sequences(dataset, root, args), desc=f"Pi3 {dataset} VGGT-Omega eval"):
        seq_out = output_root / sequence
        metric_path = seq_out / "_metrics.json"
        if metric_path.is_file() and not args.overwrite:
            cached = json.loads(metric_path.read_text())
            depth_rows.append(cached["depth"])
            pose_rows.append(cached["pose"])
            total_time += cached["speed"]["time"]
            total_frames += cached["speed"]["frames"]
            continue

        all_images = sequence_images(dataset, root, sequence)
        if args.attention_probe_frame_gap is not None:
            gap = args.attention_probe_frame_gap
            if len(all_images) < 1 + 9 * gap:
                raise ValueError(f"{sequence} has insufficient frames for attention-probe gap={gap}")
            images = all_images[0 : 1 + 9 * gap : gap]
        else:
            if args.sampling_pool_frames:
                images, source_indices, sampling_mode = compact_input_frames(all_images, args.max_frames_per_seq, args.sampling_pool_frames)
            else:
                images = limit_frames(all_images, args.max_frames_per_seq, args.frame_sample_mode, args.frame_sample_seed, sequence)
                source_indices = [int(index) for index in np.linspace(0, len(all_images) - 1, len(images), dtype=np.int64)] if args.frame_sample_mode == "uniform" else []
                sampling_mode = args.frame_sample_mode
            depths = sequence_depths(dataset, root, sequence, images)
        gt_poses = sequence_poses(dataset, root, sequence, images)
        tensor = load_and_resize14([str(path) for path in images], new_width=args.load_img_size, device=str(device), verbose=False)
        if args.timing_repeats < 1:
            raise ValueError("--timing-repeats must be at least 1")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        probe = None
        if args.attention_probe_output:
            from pi3.models.attention_probe import GlobalAttentionImageProbe
            probe = GlobalAttentionImageProbe(model, Path(args.attention_probe_output) / dataset / sequence, args.attention_probe_query_chunk)
        with torch.inference_mode(), torch.amp.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
            if probe is None:
                prediction = forward_da(model, tensor, args.da_chunk_size) if args.acceleration_method == "da-vggt" else model(tensor)
            else:
                with probe:
                    prediction = model(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            events = []
            for _ in range(args.timing_repeats):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                with torch.inference_mode(), torch.amp.autocast(device.type, dtype=dtype, enabled=True):
                    timed_prediction = forward_da(model, tensor, args.da_chunk_size) if args.acceleration_method == "da-vggt" else model(tensor)
                end_event.record()
                end_event.synchronize()
                events.append(start_event.elapsed_time(end_event))
            elapsed = float(np.median(events)) / 1000.0
            prediction = timed_prediction
        else:
            start = time.perf_counter()
            with torch.inference_mode():
                prediction = forward_da(model, tensor, args.da_chunk_size) if args.acceleration_method == "da-vggt" else model(tensor)
            elapsed = time.perf_counter() - start

        pred_depth = prediction["local_points"][0, ..., -1].detach().float().cpu().numpy()
        pred_poses = prediction["camera_poses"][0].detach().float().cpu().numpy()
        gt_depth = np.stack([read_depth(dataset, path) for path in depths])
        pred_depth = np.stack([cv2.resize(frame, (gt_depth.shape[2], gt_depth.shape[1]), interpolation=cv2.INTER_CUBIC) for frame in pred_depth])
        depth = depth_metrics(pred_depth, gt_depth, MAX_DEPTHS[dataset])
        depth.update({"sequence": sequence, "frames": len(images), "time": elapsed, "fps": len(images) / elapsed,
                      "sampling_mode": sampling_mode, "sampling_pool_frames_requested": args.sampling_pool_frames,
                      "source_frame_indices": source_indices,
                      "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else None,
                      "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / (1024**3) if device.type == "cuda" else None})
        depth.update(geometry_from_prediction(dataset, prediction, gt_depth, gt_poses))
        pose = evaluate_pose_auc(pred_poses, gt_poses, args, sequence)
        depth_rows.append(depth)
        pose_rows.append(pose)
        total_time += elapsed
        total_frames += len(images)

        seq_out.mkdir(parents=True, exist_ok=True)
        if args.save_predictions:
            np.save(seq_out / "pred_depth.npy", pred_depth)
            np.save(seq_out / "pred_poses.npy", pred_poses)
        write_json(metric_path, {"depth": depth, "pose": pose, "speed": {"time": elapsed, "frames": len(images)},
                                 "sampling": {"mode": sampling_mode, "pool_frames_requested": args.sampling_pool_frames,
                                              "source_frame_indices": source_indices}})

    weights = np.asarray([row["valid_pixels"] for row in depth_rows], dtype=np.float64)
    metric_keys = ["Abs Rel", "Sq Rel", "RMSE", "Log RMSE", "delta < 1.", "delta < 1.25", "delta < 1.25^2", "delta < 1.25^3"]
    depth_summary = {key: float(np.average([row[key] for row in depth_rows], weights=weights)) for key in metric_keys}
    pose_summary = summarize_pose(pose_rows)
    selected_method = args.acceleration_method if args.acceleration_method != "none" else args.token_merging_method
    depth_summary.update({
        "dataset": dataset, "sequences": len(depth_rows), "frames": total_frames, "time": total_time,
        "fps": total_frames / total_time, "seconds_per_frame": total_time / total_frames,
        "valid_pixels": int(weights.sum()), "depth_eval_protocol": "vggt_omega_robust_scale_shift",
        "acceleration_method": selected_method,
        "token_merging_method": selected_method if selected_method == "fastvggt" else "none",
        "token_merging_ratio": args.token_merging_ratio if selected_method == "fastvggt" else None,
        "auc_15_percent": pose_summary.get("AUC@15", 0.0),
        "peak_allocated_gib_max": float(np.max([row["peak_allocated_gib"] for row in depth_rows])) if device.type == "cuda" else None,
        "peak_reserved_gib_max": float(np.max([row["peak_reserved_gib"] for row in depth_rows])) if device.type == "cuda" else None,
    })
    for geometry_key in ("acc_mean_m", "acc_median_m", "comp_mean_m", "comp_median_m", "nc_mean", "nc_median", "chamfer_mean_m", "chamfer_median_m"):
        depth_summary[geometry_key] = float(np.mean([row[geometry_key] for row in depth_rows]))
    complete = {"video_depth": depth_summary, "pose_auc": pose_summary, "speed": {"frames": total_frames, "time": total_time, "fps": depth_summary["fps"]}}
    complete["overall"] = {
        "abs_rel": depth_summary["Abs Rel"],
        "delta_1_25_percent": 100.0 * depth_summary["delta < 1.25"],
        "auc_3_percent": pose_summary.get("AUC@3", 0.0),
        "auc_15_percent": pose_summary.get("AUC@15", 0.0),
        "auc_30_percent": pose_summary.get("AUC@30", 0.0),
        "model_latency_ms_mean": 1000.0 * total_time / max(len(depth_rows), 1),
        "fps": depth_summary["fps"],
        "peak_allocated_gib_max": depth_summary["peak_allocated_gib_max"],
        "peak_reserved_gib_max": depth_summary["peak_reserved_gib_max"],
    }
    complete["overall"].update({key: depth_summary[key] for key in ("acc_mean_m", "acc_median_m", "comp_mean_m", "comp_median_m", "nc_mean", "nc_median", "chamfer_mean_m", "chamfer_median_m")})
    complete["per_sequence"] = depth_rows
    write_csv(output_root / "_sequence_metrics_scale_shift.csv", depth_rows)
    write_json(output_root / "_summary_scale_shift.json", depth_summary)
    write_json(output_root / "_summary_pose_auc.json", pose_summary)
    write_json(output_root / "_summary_complete_scale_shift.json", complete)
    print(json.dumps(complete, indent=2))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model = build_model(args, device)
    for dataset in ("tum_dynamic", "7scenes") if args.dataset == "all" else (args.dataset,):
        run_dataset(args, model, device, dataset)


if __name__ == "__main__":
    main()
