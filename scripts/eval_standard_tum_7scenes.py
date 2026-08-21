#!/usr/bin/env python3
"""Standard VGGT evaluation aligned with VGGT-Omega's TUM/7Scenes protocol.

Each official test sequence is sampled uniformly to 300 frames, inferred as one
video, and scored with robust scale-shift depth metrics plus official relative
pose AUC. CUDA Event timing excludes image loading and metric computation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SHARED_ROOT = Path("/data/mmc_syang")
if str(SHARED_ROOT) not in sys.path:
    sys.path.insert(0, str(SHARED_ROOT))

from geometry_eval import depth_to_world_points, evaluate_pi3_geometry, scaled_intrinsics, trajectory_pose_metrics

from inference.infer_vggt import sequence_images, sequence_names, sequence_poses
from inference.pose_auc import evaluate_pose_auc, summarize_pose_auc
from vggt.evaluation import load_model
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

DATA_ROOTS = {"tum_dynamic": Path("/data/mmc_syang/dataset/TUM-Dynamics"),
              "7scenes": Path("/data/mmc_syang/dataset/7scenes"),
              "nrgbd": Path("/data/mmc_syang/dataset/NRGBD"),
              "scannet": Path("/data/mmc_syang/dataset/scannet30/raw")}
MAX_DEPTHS = {"tum_dynamic": 10.0, "7scenes": 10.0, "nrgbd": 10.0, "scannet": 10.0}


def args_parse():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("tum_dynamic", "7scenes", "nrgbd", "scannet"), required=True)
    p.add_argument("--dataset-root", type=Path)
    p.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/vggt_1b.pt")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-frames", type=int, default=300)
    p.add_argument(
        "--sampling-stride", type=int, default=3,
        help="For sequences longer than stride * num-frames, select frames from index 0 at this fixed stride.",
    )
    p.add_argument("--sampling-pool-frames", type=int, default=0,
                   help="Uniformly form this many source candidates, then use the first --num-frames; short sequences use source first frames.")
    p.add_argument("--timing-repeats", type=int, default=3)
    p.add_argument("--image-resolution", type=int, default=512)
    p.add_argument("--sequences", nargs="*")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--acceleration-method",
        choices=("none", "fastvggt", "sparse-vggt", "da-vggt", "u-m"),
        default="none",
        help="Unified acceleration selector. The legacy FastVGGT configuration is retained for compatibility.",
    )
    p.add_argument("--merge-ratio", type=float, default=0.9)
    p.add_argument("--sparse-vggt-sparse-ratio", type=float, default=0.5)
    p.add_argument("--sparse-vggt-cdf-threshold", type=float, default=None)
    p.add_argument("--sparse-vggt-pool-mode", choices=("avg", "max"), default="avg")
    p.add_argument("--um-lambda", type=float, default=0.04)
    p.add_argument("--um-spatial-radius", type=int, default=2)
    p.add_argument("--um-temporal-window", type=int, default=4)
    p.add_argument("--um-refresh-layers", default="0,9,21")
    p.add_argument(
        "--model-bfloat16", dest="model_bfloat16", action="store_true", default=True,
        help="Use the default full-model bf16 inference path.",
    )
    p.add_argument(
        "--model-fp32", dest="model_bfloat16", action="store_false",
        help="Disable full-model bf16 inference (diagnostic compatibility option).",
    )
    p.add_argument("--attention-probe-output", type=Path, default=None,
                   help="Optional directory for one full-token global-attention TIFF set; disabled by default.")
    p.add_argument("--attention-probe-query-chunk", type=int, default=128)
    p.add_argument("--attention-probe-frame-gap", type=int, default=None,
                   help="When probing, select exactly 10 source frames at this fixed gap.")
    p.add_argument("--da-chunk-size", type=int, default=50)
    return p.parse_args()


def uniform(paths: list[str], count: int) -> list[str]:
    if count <= 0 or len(paths) <= count:
        return paths
    return [paths[int(i)] for i in np.linspace(0, len(paths) - 1, count, dtype=np.int64)]


def compact_input_frames(paths: list[str], input_frames: int, sampling_pool_frames: int) -> tuple[list[str], list[int], str]:
    if sampling_pool_frames <= 0:
        selected = uniform(paths, input_frames)
        indices = np.rint(np.linspace(0, len(paths) - 1, len(selected))).astype(np.int64).tolist() if selected else []
        return selected, indices, "uniform"
    if len(paths) < sampling_pool_frames:
        return paths[:input_frames], list(range(min(input_frames, len(paths)))), "short_sequence_first"
    indices = np.rint(np.linspace(0, len(paths) - 1, sampling_pool_frames)).astype(np.int64)[:input_frames].tolist()
    return [paths[int(i)] for i in indices], indices, "uniform_pool_then_first"


def gt_depths(dataset: str, root: Path, seq: str, images: list[str]) -> list[Path]:
    if dataset == "nrgbd":
        return [root / seq / "depth" / f"depth{Path(image).stem.removeprefix('img')}.png" for image in images]
    if dataset == "scannet":
        return [root / seq / "depth" / f"{Path(image).stem}.png" for image in images]
    if dataset == "7scenes":
        out = []
        for image in images:
            p = Path(image).with_name(Path(image).name.replace(".color.png", ".depth.proj.png"))
            if not p.is_file():
                p = Path(image).with_name(Path(image).name.replace(".color.png", ".depth.png"))
            out.append(p)
        return out
    entries = []
    with (root / seq / "depth.txt").open() as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                stamp, path = line.split()[:2]
                entries.append((float(stamp), root / seq / path))
    times = np.asarray([x[0] for x in entries])
    return [entries[int(np.argmin(np.abs(times - float(Path(image).stem))))][1] for image in images]


def read_depth(dataset: str, path: Path) -> np.ndarray:
    raw = np.asarray(Image.open(path))
    if dataset == "tum_dynamic":
        result = raw.astype(np.float32) / 5000.0
        result[raw == 0] = -1.0
    else:
        result = raw.astype(np.float32) / 1000.0
        result[(raw == 0) | (raw == 65535)] = -1.0
    return result


def depth_metrics_batched(
    predictions: np.ndarray, ground_truth: np.ndarray, max_depth: float,
    fit_samples: int = 32768, batch_size: int = 32,
) -> dict[str, float]:
    """Per-frame robust scale/shift, fitted in CPU batches then scored at full size.

    This preserves the original protocol (one independent IRLS fit per frame),
    but replaces 300 Python calls with vectorised batches.  A fixed uniformly
    spaced pixel subset is used only for fitting; AbsRel and delta are still
    accumulated over every valid full-resolution depth pixel.
    """
    if predictions.shape != ground_truth.shape:
        raise ValueError(f"Depth shape mismatch: {predictions.shape} vs {ground_truth.shape}")
    frames, height, width = predictions.shape
    sample_ids = np.linspace(0, height * width - 1, min(fit_samples, height * width), dtype=np.int64)
    abs_rel_sum = sq_rel_sum = squared_sum = log_squared_sum = mae_sum = 0.0
    delta_count = delta2_count = delta3_count = 0
    valid_count = 0
    for start in range(0, frames, batch_size):
        stop = min(start + batch_size, frames)
        pred = predictions[start:stop].reshape(stop - start, -1).astype(np.float64, copy=False)
        gt = ground_truth[start:stop].reshape(stop - start, -1).astype(np.float64, copy=False)
        valid = np.isfinite(pred) & np.isfinite(gt) & (gt > 0) & (gt < max_depth)
        if not np.all(valid.any(axis=1)):
            raise ValueError("No valid depth pixels in at least one frame")
        x = pred[:, sample_ids]
        y = gt[:, sample_ids]
        sample_valid = valid[:, sample_ids]
        x = np.where(sample_valid, x, 0.0)
        y = np.where(sample_valid, y, 0.0)
        # NaNs let nanmedian retain the original frame-wise robust initialization.
        x_median = np.nanmedian(np.where(sample_valid, x, np.nan), axis=1)
        y_median = np.nanmedian(np.where(sample_valid, y, np.nan), axis=1)
        # Very sparse validity at the sampled sites is unlikely, but fall back
        # to the full valid set to preserve a usable initialization.
        fallback = ~np.isfinite(x_median) | ~np.isfinite(y_median)
        if fallback.any():
            x_median[fallback] = np.array([np.median(pred[i, valid[i]]) for i in np.flatnonzero(fallback)])
            y_median[fallback] = np.array([np.median(gt[i, valid[i]]) for i in np.flatnonzero(fallback)])
        scale = y_median / np.maximum(x_median, 1e-8)
        shift = np.zeros_like(scale)
        for _ in range(20):
            residual = scale[:, None] * x + shift[:, None] - y
            weights = np.where(sample_valid, 1.0 / np.maximum(np.abs(residual), 1e-4), 0.0)
            weight_sum = weights.sum(axis=1)
            mean_x = (weights * x).sum(axis=1) / weight_sum
            mean_y = (weights * y).sum(axis=1) / weight_sum
            centered_x = x - mean_x[:, None]
            variance = (weights * centered_x * centered_x).sum(axis=1)
            covariance = (weights * centered_x * (y - mean_y[:, None])).sum(axis=1)
            stable = variance > np.finfo(np.float64).eps
            scale[stable] = covariance[stable] / variance[stable]
            shift[stable] = mean_y[stable] - scale[stable] * mean_x[stable]
        aligned = np.clip(scale[:, None] * pred + shift[:, None], 1e-5, max_depth)
        safe_gt = np.where(valid, gt, 1.0)
        ratio = np.maximum(aligned / safe_gt, safe_gt / aligned)
        diff = aligned - safe_gt
        abs_rel_sum += float((np.where(valid, np.abs(diff) / safe_gt, 0.0)).sum())
        sq_rel_sum += float((np.where(valid, diff**2 / safe_gt, 0.0)).sum())
        squared_sum += float((np.where(valid, diff**2, 0.0)).sum())
        log_squared_sum += float((np.where(valid, np.log(aligned / safe_gt)**2, 0.0)).sum())
        mae_sum += float((np.where(valid, np.abs(diff), 0.0)).sum())
        delta_count += int((valid & (ratio < 1.25)).sum())
        delta2_count += int((valid & (ratio < 1.25**2)).sum())
        delta3_count += int((valid & (ratio < 1.25**3)).sum())
        valid_count += int(valid.sum())
    return {"abs_rel": abs_rel_sum / valid_count, "sq_rel": sq_rel_sum / valid_count,
            "rmse_m": float(np.sqrt(squared_sum / valid_count)), "rmse_log": float(np.sqrt(log_squared_sum / valid_count)),
            "mae_m": mae_sum / valid_count, "delta_1_25_percent": 100 * delta_count / valid_count,
            "delta_1_25_sq_percent": 100 * delta2_count / valid_count,
            "delta_1_25_cu_percent": 100 * delta3_count / valid_count, "valid_depth_pixels": float(valid_count)}


def geometry_metrics(
    dataset: str,
    root: Path,
    seq: str,
    predicted_depth: np.ndarray,
    pred_c2w: np.ndarray,
    ground_truth_depth: np.ndarray,
    gt_c2w: np.ndarray,
) -> dict[str, float | int]:
    """Pi3-compatible Sim(3)+ICP reconstruction metrics on model-resolution maps."""
    height, width = predicted_depth.shape[1:]
    if dataset == "scannet":
        # Raw depths are 640x480 but share the RGB viewing ray after scaling;
        # use the scene's RGB calibration and its native colour resolution.
        intrinsic = np.loadtxt(root / seq / "intrinsic" / "intrinsic_color.txt", dtype=np.float64)[:3, :3]
        native_shape = (968, 1296)
    else:
        focal = 554.2562584220408 if dataset == "tum_dynamic" else 525.0
        intrinsic = np.array([[focal, 0.0, 320.0], [0.0, focal, 240.0], [0.0, 0.0, 1.0]])
        native_shape = (480, 640)
    intrinsics = scaled_intrinsics(intrinsic, native_shape, (height, width), len(predicted_depth))
    pred_points = depth_to_world_points(predicted_depth, pred_c2w, intrinsics)
    gt_points = depth_to_world_points(ground_truth_depth, gt_c2w, intrinsics)
    valid = (
        np.isfinite(predicted_depth) & (predicted_depth > 0)
        & np.isfinite(ground_truth_depth) & (ground_truth_depth > 0)
        & (ground_truth_depth < MAX_DEPTHS[dataset])
    )
    return evaluate_pi3_geometry(pred_points, gt_points, valid)


def forward(model, image_paths: list[str], device: torch.device, resolution: int):
    images = load_and_preprocess_images(image_paths, mode="crop", target_size=resolution).to(device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        out = model(images)
    depths = out["depth"][0, ..., 0].float().cpu().numpy()
    poses = None
    if "pose_enc" in out:
        # Geometry post-processing uses matrix inversion, which is not
        # implemented for bf16.  Keep the model forward in bf16 but promote
        # only this small pose tensor for evaluation.
        w2c, _ = pose_encoding_to_extri_intri(out["pose_enc"].float(), out["images"].shape[-2:])
        bottom = torch.zeros((*w2c.shape[:-2], 1, 4), device=w2c.device, dtype=w2c.dtype)
        bottom[..., 0, 3] = 1.0
        poses = torch.linalg.inv(torch.cat([w2c, bottom], dim=-2)[0]).float().cpu().numpy()
    return depths, poses, images.shape[-1], images.shape[-2]


def forward_da(model, image_paths: list[str], device: torch.device, resolution: int, chunk_size: int):
    """Full native DA-VGGT: cached DINO, pose-weighted rechunking, token reuse."""
    images = load_and_preprocess_images(image_paths, mode="crop", target_size=resolution).to(device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        out = model.forward_da_vggt(images, chunk_size=chunk_size)
    depths = out["depth"][0, ..., 0].float().cpu().numpy()
    w2c = out["da_w2c"][0].float().cpu().numpy()
    bottom = np.broadcast_to(np.array([0.0, 0.0, 0.0, 1.0]), (len(w2c), 1, 4))
    poses = np.linalg.inv(np.concatenate((w2c, bottom), axis=1))
    return depths, poses, images.shape[-1], images.shape[-2]


def main():
    a = args_parse()
    if a.timing_repeats < 1:
        raise ValueError("--timing-repeats must be positive")
    if a.sampling_stride <= 0:
        raise ValueError("--sampling-stride must be positive")
    device = torch.device(a.device)
    root = a.dataset_root or DATA_ROOTS[a.dataset]
    model = load_model(
        a.checkpoint,
        device,
        enable_camera=True,
        enable_depth=True,
        inter_frame_attention="global",
        enable_token_merging=a.acceleration_method == "fastvggt",
        token_merging_ratio=a.merge_ratio,
        token_merging_method="spatial",
        um_lambda_cost=a.um_lambda if a.acceleration_method == "u-m" else None,
        um_spatial_radius=a.um_spatial_radius,
        um_temporal_window=a.um_temporal_window,
        um_refresh_layers=a.um_refresh_layers,
        model_bfloat16=a.model_bfloat16,
    )
    if a.acceleration_method == "sparse-vggt":
        sparse_root = Path("/data/mmc_syang/sparse-vggt/src")
        sparge_root = Path("/data/mmc_syang/sparse-vggt/external/SpargeAttn")
        if not sparse_root.is_dir():
            raise FileNotFoundError(f"Sparse-VGGT source not found: {sparse_root}")
        if not sparge_root.is_dir():
            raise FileNotFoundError(f"SpargeAttn source not found: {sparge_root}")
        sys.path.insert(0, str(sparse_root))
        sys.path.insert(0, str(sparge_root))
        from sparse_vggt.models.vggt import sparse_aggregator_from_vggt

        model.aggregator, _ = sparse_aggregator_from_vggt(
            model.aggregator,
            sparse_ratio=a.sparse_vggt_sparse_ratio,
            cdf_threshold=a.sparse_vggt_cdf_threshold,
            pool_mode=a.sparse_vggt_pool_mode,
            verbose=True,
        )
    model.eval()
    if a.dataset == "nrgbd":
        sequences = list(a.sequences) if a.sequences else sorted(path.name for path in root.iterdir() if (path / "images").is_dir() and (path / "depth").is_dir() and (path / "poses.txt").is_file())
    else:
        sequences = sequence_names(a.dataset, root, a.sequences, True, "test")
    rows = []; pose_rows = []; skipped_sequences = []; total_ms = 0.0
    for seq in sequences:
        all_images = sequence_images(a.dataset, root, seq)
        if len(all_images) < a.num_frames:
            skipped_sequences.append({"sequence": seq, "reason": "fewer_than_requested_frames", "available_frames": len(all_images)})
            print(f"{seq}: skipped ({len(all_images)} < {a.num_frames} frames)")
            continue
        if a.attention_probe_frame_gap is not None:
            if len(all_images) < 1 + 9 * a.attention_probe_frame_gap:
                raise ValueError(f"{seq} has insufficient frames for attention-probe gap")
            images = all_images[0 : 1 + 9 * a.attention_probe_frame_gap : a.attention_probe_frame_gap]
        else:
            if len(all_images) > a.sampling_stride * a.num_frames:
                source_indices = list(range(0, a.sampling_stride * a.num_frames, a.sampling_stride))
                images = [all_images[index] for index in source_indices]
                sampling_mode = f"fixed_stride_{a.sampling_stride}_from_first"
            else:
                images, source_indices, sampling_mode = compact_input_frames(all_images, a.num_frames, a.sampling_pool_frames)
        depths = gt_depths(a.dataset, root, seq, images)
        gt_pose = sequence_poses(a.dataset, root, seq, len(images), images)
        # Untimed warmup, then formal CUDA Event measurements.
        run_forward = forward_da if a.acceleration_method == "da-vggt" else forward
        if a.attention_probe_output:
            from vggt.models.attention_probe import GlobalAttentionImageProbe
            with GlobalAttentionImageProbe(model, a.attention_probe_output / a.dataset / seq, a.attention_probe_query_chunk):
                run_forward(model, images, device, a.image_resolution)
        else:
            run_forward(model, images, device, a.image_resolution, a.da_chunk_size) if a.acceleration_method == "da-vggt" else run_forward(model, images, device, a.image_resolution)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        times = []
        pred = pose = None
        for _ in range(a.timing_repeats):
            if device.type == "cuda":
                start, end = torch.cuda.Event(True), torch.cuda.Event(True)
                start.record()
                pred, pose, width, height = (forward_da(model, images, device, a.image_resolution, a.da_chunk_size)
                                             if a.acceleration_method == "da-vggt" else forward(model, images, device, a.image_resolution))
                end.record(); end.synchronize(); times.append(start.elapsed_time(end))
            else:
                import time; t = time.perf_counter(); pred, pose, width, height = (forward_da(model, images, device, a.image_resolution, a.da_chunk_size)
                    if a.acceleration_method == "da-vggt" else forward(model, images, device, a.image_resolution)); times.append((time.perf_counter()-t)*1000)
        latency = float(np.median(times)); total_ms += latency
        gt = np.stack([cv2.resize(read_depth(a.dataset, p), (width, height), interpolation=cv2.INTER_NEAREST) for p in depths])
        depth_metric = depth_metrics_batched(pred, gt, MAX_DEPTHS[a.dataset])
        pose_metric = evaluate_pose_auc(pose, gt_pose) if pose is not None and gt_pose is not None else {"AUC@3": 0.0, "AUC@15": 0.0, "AUC@30": 0.0, "pose_errors_deg": [], "rotation_errors_deg": [], "translation_errors_deg": []}
        merge_stats = getattr(model.aggregator, "last_token_merging_stats", [])
        retention = (
            100.0 * float(np.mean([stat["full_attention_token_ratio"] for stat in merge_stats]))
            if merge_stats else 100.0
        )
        um_stats = [stat for stat in merge_stats if stat.get("mode") == "u-m"]
        row = {"sequence": seq, "acceleration_method": a.acceleration_method, **depth_metric, "auc_3_percent": pose_metric["AUC@3"], "auc_5_percent": pose_metric["AUC@5"], "auc_10_percent": pose_metric["AUC@10"], "auc_15_percent": pose_metric["AUC@15"], "auc_30_percent": pose_metric["AUC@30"], "model_latency_ms": latency, "fps": len(images)/(latency/1000), "frames": len(images), "sampling_mode": sampling_mode, "sampling_pool_frames_requested": a.sampling_pool_frames, "source_frame_indices": source_indices, "patch_retention_percent": retention, "um_edge_score_backend": um_stats[-1].get("um_edge_score_backend") if um_stats else None, "um_layer_retention_percent": [100.0 * float(stat["full_attention_token_ratio"]) for stat in um_stats], "depth_alignment": "per-frame-batched-irls-20x32768", "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else None, "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / (1024**3) if device.type == "cuda" else None}
        if pose is not None and gt_pose is not None:
            row.update(geometry_metrics(a.dataset, root, seq, pred, pose, gt, gt_pose))
            pose_metric.update(trajectory_pose_metrics(pose, gt_pose))
        rows.append(row); pose_rows.append(pose_metric)
        print(f"{seq}: AbsRel={row['abs_rel']:.4f}, AUC@3={row['auc_3_percent']:.2f}, latency={latency:.1f}ms")
    pose_summary = summarize_pose_auc(pose_rows)
    valid_rows = rows
    overall = {key: float(np.mean([r[key] for r in valid_rows])) for key in ("abs_rel", "sq_rel", "rmse_m", "rmse_log", "mae_m", "delta_1_25_percent", "delta_1_25_sq_percent", "delta_1_25_cu_percent")}
    overall.update({"auc_3_percent": pose_summary["AUC@3"], "auc_5_percent": pose_summary["AUC@5"], "auc_10_percent": pose_summary["AUC@10"], "auc_15_percent": pose_summary["AUC@15"], "auc_30_percent": pose_summary["AUC@30"], "model_latency_ms_mean": total_ms/len(rows), "fps": a.num_frames/(total_ms/len(rows)/1000), "peak_allocated_gib_max": float(np.max([r["peak_allocated_gib"] for r in valid_rows])) if device.type == "cuda" else None, "peak_reserved_gib_max": float(np.max([r["peak_reserved_gib"] for r in valid_rows])) if device.type == "cuda" else None})
    overall["active_token_ratio_percent"] = float(np.mean([r["patch_retention_percent"] for r in valid_rows]))
    overall["keep_ratio_percent"] = overall["active_token_ratio_percent"]
    for key in ("ate_rmse_m", "are_deg", "rpe_translation_rmse_m", "rpe_rotation_rmse_deg", "rra_30_percent", "rta_30_percent"):
        overall[key] = float(np.mean([row[key] for row in pose_rows]))
    for geometry_key in ("acc_mean_m", "acc_median_m", "comp_mean_m", "comp_median_m", "nc_mean", "nc_median", "chamfer_mean_m", "chamfer_median_m"):
        overall[geometry_key] = float(np.mean([r[geometry_key] for r in valid_rows if geometry_key in r]))
    payload = {"protocol": {"dataset": a.dataset, "acceleration_method": a.acceleration_method, "num_frames_per_sequence": a.num_frames, "sampling_stride": a.sampling_stride, "sampling_strategy": "uniform_pool_then_first" if a.sampling_pool_frames else "uniform", "sampling_pool_frames_requested": a.sampling_pool_frames, "timing": "cuda_event_median_after_warmup", "timing_repeats": a.timing_repeats}, "overall": overall, "per_sequence": rows, "skipped_sequences": skipped_sequences}
    a.output_dir.mkdir(parents=True, exist_ok=True); (a.output_dir / "metrics.json").write_text(json.dumps(payload, indent=2)); print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
