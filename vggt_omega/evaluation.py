"""Shared inference and dataset helpers for VGGT-Omega evaluation scripts."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch
from PIL import Image

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

INTER_FRAME_ATTENTION_MODES = (
    "partial",
    "global",
    "register",
    "alternate1",
    "alternate2",
    "partial_plus17",
    "partial_plus3",
    "partial_pairs",
    "partial_plus3_17",
    "partial_2_10_17_20",
)
AGGREGATOR_DEPTH = 24


SINTEL_SEQUENCES = [
    "alley_2",
    "ambush_4",
    "ambush_5",
    "ambush_6",
    "cave_2",
    "cave_4",
    "market_2",
    "market_5",
    "market_6",
    "shaman_3",
    "sleeping_1",
    "sleeping_2",
    "temple_2",
    "temple_3",
]
SINTEL_TAG_FLOAT = 202021.25


def write_csv(path: str | Path, rows: Sequence[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sequence_names(dataset_root: str | Path, requested: Sequence[str] | None = None) -> list[str]:
    final_root = Path(dataset_root) / "final"
    requested = list(requested) if requested else SINTEL_SEQUENCES
    missing = [seq for seq in requested if not (final_root / seq).is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing Sintel sequences below {final_root}: {missing}")
    return requested


def sequence_images(dataset_root: str | Path, seq: str) -> list[str]:
    paths = sorted((Path(dataset_root) / "final" / seq).glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No input images found for Sintel sequence {seq}")
    return [str(path) for path in paths]


def sequence_cameras(dataset_root: str | Path, seq: str) -> list[str]:
    paths = sorted((Path(dataset_root) / "camdata_left" / seq).glob("*.cam"))
    if not paths:
        raise FileNotFoundError(f"No camera annotations found for Sintel sequence {seq}")
    return [str(path) for path in paths]


def read_sintel_depth(filename: str | Path) -> np.ndarray:
    with Path(filename).open("rb") as handle:
        tag = np.fromfile(handle, dtype=np.float32, count=1)[0]
        if tag != SINTEL_TAG_FLOAT:
            raise ValueError(f"Invalid Sintel depth tag in {filename}: {tag}")
        width = int(np.fromfile(handle, dtype=np.int32, count=1)[0])
        height = int(np.fromfile(handle, dtype=np.int32, count=1)[0])
        return np.fromfile(handle, dtype=np.float32).reshape(height, width)


def read_bonn_depth(filename: str | Path) -> np.ndarray:
    depth_png = np.asarray(Image.open(filename))
    depth = depth_png.astype(np.float32) / 5000.0
    depth[depth_png == 0] = -1.0
    return depth


def read_sintel_camera(filename: str | Path) -> np.ndarray:
    """Read Sintel world-to-camera annotation and return a 4x4 c2w pose."""
    with Path(filename).open("rb") as handle:
        tag = np.fromfile(handle, dtype=np.float32, count=1)[0]
        if tag != SINTEL_TAG_FLOAT:
            raise ValueError(f"Invalid Sintel camera tag in {filename}: {tag}")
        np.fromfile(handle, dtype=np.float64, count=9)
        w2c = np.fromfile(handle, dtype=np.float64, count=12).reshape(3, 4)
    w2c = np.vstack([w2c, [0.0, 0.0, 0.0, 1.0]])
    return np.linalg.inv(w2c)


def _crop_depth_like_image(depth: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    if depth.shape[:2] != (height, width):
        raise ValueError(f"Image shape {(height, width)} does not match depth shape {depth.shape[:2]}")
    aspect_ratio = height / max(width, 1)
    if aspect_ratio < 0.5:
        crop_width = min(width, max(1, int(round(height / 0.5))))
        left = max((width - crop_width) // 2, 0)
        return depth[:, left : left + crop_width]
    if aspect_ratio > 2.0:
        crop_height = min(height, max(1, int(round(width * 2.0))))
        top = max((height - crop_height) // 2, 0)
        return depth[top : top + crop_height]
    return depth


def preprocess_depth(
    image_filename: str | Path,
    depth: np.ndarray,
    resolution: tuple[int, int],
) -> np.ndarray:
    """Apply VGGT-Omega's center crop and resize geometry to a GT depth map."""
    with Image.open(image_filename) as image:
        depth = _crop_depth_like_image(depth, image.size)
    return cv2.resize(depth, resolution, interpolation=cv2.INTER_NEAREST).astype(np.float32)


def preprocess_sintel_depth(
    image_filename: str | Path, depth_filename: str | Path, resolution: tuple[int, int]
) -> np.ndarray:
    return preprocess_depth(image_filename, read_sintel_depth(depth_filename), resolution)


def preprocess_bonn_depth(
    image_filename: str | Path, depth_filename: str | Path, resolution: tuple[int, int]
) -> np.ndarray:
    return preprocess_depth(image_filename, read_bonn_depth(depth_filename), resolution)


def load_model(
    checkpoint_path: str | Path,
    device: torch.device,
    *,
    enable_camera: bool = True,
    enable_depth: bool = True,
    inter_frame_attention: str = "partial",
    register_patch_sample_tokens: int = 0,
    register_patch_sample_ratio: float = 0.0,
    register_patch_sample_mode: str = "uniform",
    register_patch_merge_sources: bool = False,
    register_patch_merge_protect_first_frame: bool = False,
    enable_token_merging: bool = False,
    token_merging_start: int = 0,
    token_merging_ratio: float = 0.9,
    token_merging_layer_ratios: str = "",
    token_merging_method: str = "spatial",
    token_merging_flashvid_alpha: float = 0.7,
    token_merging_flashvid_expansion: float = 1.25,
    token_merging_flashvid_pool_stride: int = 2,
    token_merging_flashvid_tstm_threshold: float = 0.8,
    token_merging_fastvggt_destination_selector: str = "random",
    token_merging_fastvggt_destination_policy: str = "grid_2x2",
    token_merging_fastvggt_uniform_protect_ratio: float = 0.0,
    token_merging_fastvggt_exclusive_protection: bool = True,
    token_merging_fastvggt_protect_anchor_frames: bool = True,
    token_merging_frame_restore_layer: int = 16,
    token_merging_frame_alpha: float = 0.9,
    token_merging_frame_segment_threshold: float = 0.8,
    token_merging_frame_merge_threshold: float = 0.8,
    token_merging_frame_max_window: int = 6,
    token_merging_frame_pool_stride: int = 2,
    token_merging_frame_multi_max_group_size: int = 2,
    token_merging_frame_multi_pair_threshold: float = 0.95,
    token_merging_frame_multi_span_threshold: float = 0.93,
    token_merging_frame_upper_adaptive: bool = False,
    token_merging_frame_staged_ranges: str = "0-9,10-17,18-23",
    token_merging_frame_staged_late_segment_threshold: float | None = None,
    token_merging_frame_staged_late_pair_threshold: float | None = None,
    token_merging_frame_staged_late_span_threshold: float | None = None,
    token_merging_frame_group_strategy: str = "local",
    token_merging_frame_protect_period: int = 0,
    token_merging_frame_protect_prefix: int = 0,
    token_merging_frame_anchor_count: int = 4,
    token_merging_frame_anchor_selection: str = "uniform",
    token_merging_frame_adaptive_boundary_z: float = 2.5,
    token_merging_frame_adaptive_medoid_z: float = 1.5,
    token_merging_frame_patch_fusion_quantile: float = 0.75,
    token_merging_frame_special_cross_attention: bool = False,
    token_merging_frame_special_cross_attention_alpha: float = 0.1,
    token_merging_segment_bank_pair_threshold: float = 0.986,
    token_merging_segment_bank_span_threshold: float = 0.948,
    token_merging_segment_bank_max_group_size: int = 4,
    omega_accelerator: str = "none",
    sparse_vggt_sparse_ratio: float | None = 0.5,
    sparse_vggt_cdf_threshold: float | None = None,
    sparse_vggt_pool_mode: str = "avg",
    da_vggt_max_frames: int = 0,
    da_vggt_sampling_method: str = "fl_maxmin",
    da_vggt_n_anchors: int = 1,
    da_vggt_dino_batch_size: int = 256,
    da_vggt_lambda_div: float = 0.0,
    da_chunk_strided_groups: int = 5,
    da_chunk_strided_anchor_count: int = 5,
    shared_anchor_num_chunks: int = 10,
    shared_anchor_count: int = 10,
    dynamic_fastvggt_schedule: str = "all",
    skip_global_attention_blocks: str = "",
    skip_inter_frame_attention_blocks: str = "",
    frame_only_inter_frame_blocks: str = "",
    register_only_blocks: str = "",
    enable_adaptive_frame_token_fusion: bool = False,
    adaptive_frame_representation: str = "global_pool",
    adaptive_representation_pca_dim: int = 512,
    adaptive_representation_clusters: int = 3,
    adaptive_spatial_grid: int = 4,
    adaptive_grouping: str = "serial",
    adaptive_reference_selection: str = "first",
    adaptive_reference_participates: bool = True,
    adaptive_group_similarity_threshold: float = 0.98,
    adaptive_group_max_size: int = 4,
    adaptive_parallel_window: int = 10,
    adaptive_update_policy: str = "initial_only",
    adaptive_update_after_blocks: str = "9,17",
    adaptive_frame_fusion: str = "direct",
    adaptive_frame_fusion_weighting: str = "similarity",
    adaptive_frame_token_similarity_threshold: float = 0.95,
    adaptive_token_merging: str = "fast_bipartite",
    adaptive_token_keep_ratio: float = 0.1,
    adaptive_token_clusters: int = 4,
    adaptive_token_cluster_budget: str = "proportional",
    adaptive_token_kmeans_iterations: int = 12,
) -> VGGTOmega:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    register_attention_block_indices = (
        parse_block_indices(register_only_blocks)
        if register_only_blocks.strip()
        else register_attention_indices(inter_frame_attention)
    )
    # Avoid first allocating and initializing another full 1B-parameter copy
    # before assigning tensors from the released checkpoint.
    with torch.device("meta"):
        model = VGGTOmega(
            enable_camera=enable_camera,
            enable_depth=enable_depth,
            register_attention_block_indices=register_attention_block_indices,
            register_patch_sample_tokens=register_patch_sample_tokens,
            register_patch_sample_ratio=register_patch_sample_ratio,
            register_patch_sample_mode=register_patch_sample_mode,
            register_patch_merge_sources=register_patch_merge_sources,
            register_patch_merge_protect_first_frame=register_patch_merge_protect_first_frame,
            enable_token_merging=enable_token_merging,
            token_merging_start=token_merging_start,
            token_merging_ratio=token_merging_ratio,
            token_merging_layer_ratios=token_merging_layer_ratios,
            token_merging_method=token_merging_method,
            token_merging_flashvid_alpha=token_merging_flashvid_alpha,
            token_merging_flashvid_expansion=token_merging_flashvid_expansion,
            token_merging_flashvid_pool_stride=token_merging_flashvid_pool_stride,
            token_merging_flashvid_tstm_threshold=token_merging_flashvid_tstm_threshold,
            token_merging_fastvggt_destination_selector=token_merging_fastvggt_destination_selector,
            token_merging_fastvggt_destination_policy=token_merging_fastvggt_destination_policy,
            token_merging_fastvggt_uniform_protect_ratio=token_merging_fastvggt_uniform_protect_ratio,
            token_merging_fastvggt_exclusive_protection=token_merging_fastvggt_exclusive_protection,
            token_merging_fastvggt_protect_anchor_frames=token_merging_fastvggt_protect_anchor_frames,
            token_merging_frame_restore_layer=token_merging_frame_restore_layer,
            token_merging_frame_alpha=token_merging_frame_alpha,
            token_merging_frame_segment_threshold=token_merging_frame_segment_threshold,
            token_merging_frame_merge_threshold=token_merging_frame_merge_threshold,
            token_merging_frame_max_window=token_merging_frame_max_window,
            token_merging_frame_pool_stride=token_merging_frame_pool_stride,
            token_merging_frame_multi_max_group_size=token_merging_frame_multi_max_group_size,
            token_merging_frame_multi_pair_threshold=token_merging_frame_multi_pair_threshold,
            token_merging_frame_multi_span_threshold=token_merging_frame_multi_span_threshold,
            token_merging_frame_upper_adaptive=token_merging_frame_upper_adaptive,
            token_merging_frame_staged_ranges=token_merging_frame_staged_ranges,
            token_merging_frame_staged_late_segment_threshold=token_merging_frame_staged_late_segment_threshold,
            token_merging_frame_staged_late_pair_threshold=token_merging_frame_staged_late_pair_threshold,
            token_merging_frame_staged_late_span_threshold=token_merging_frame_staged_late_span_threshold,
            token_merging_frame_group_strategy=token_merging_frame_group_strategy,
            token_merging_frame_protect_period=token_merging_frame_protect_period,
            token_merging_frame_protect_prefix=token_merging_frame_protect_prefix,
            token_merging_frame_anchor_count=token_merging_frame_anchor_count,
            token_merging_frame_anchor_selection=token_merging_frame_anchor_selection,
            token_merging_frame_adaptive_boundary_z=token_merging_frame_adaptive_boundary_z,
            token_merging_frame_adaptive_medoid_z=token_merging_frame_adaptive_medoid_z,
            token_merging_frame_patch_fusion_quantile=token_merging_frame_patch_fusion_quantile,
            token_merging_frame_special_cross_attention=token_merging_frame_special_cross_attention,
            token_merging_frame_special_cross_attention_alpha=token_merging_frame_special_cross_attention_alpha,
            token_merging_segment_bank_pair_threshold=token_merging_segment_bank_pair_threshold,
            token_merging_segment_bank_span_threshold=token_merging_segment_bank_span_threshold,
            token_merging_segment_bank_max_group_size=token_merging_segment_bank_max_group_size,
            omega_accelerator=omega_accelerator,
            sparse_vggt_sparse_ratio=sparse_vggt_sparse_ratio,
            sparse_vggt_cdf_threshold=sparse_vggt_cdf_threshold,
            sparse_vggt_pool_mode=sparse_vggt_pool_mode,
            da_vggt_max_frames=da_vggt_max_frames,
            da_vggt_sampling_method=da_vggt_sampling_method,
            da_vggt_n_anchors=da_vggt_n_anchors,
            da_vggt_dino_batch_size=da_vggt_dino_batch_size,
            da_vggt_lambda_div=da_vggt_lambda_div,
            da_chunk_strided_groups=da_chunk_strided_groups,
            da_chunk_strided_anchor_count=da_chunk_strided_anchor_count,
            shared_anchor_num_chunks=shared_anchor_num_chunks,
            shared_anchor_count=shared_anchor_count,
            dynamic_fastvggt_schedule=dynamic_fastvggt_schedule,
            skip_global_attention_blocks=skip_global_attention_blocks,
            skip_inter_frame_attention_blocks=skip_inter_frame_attention_blocks,
            frame_only_inter_frame_blocks=frame_only_inter_frame_blocks,
            enable_adaptive_frame_token_fusion=enable_adaptive_frame_token_fusion,
            adaptive_frame_representation=adaptive_frame_representation,
            adaptive_representation_pca_dim=adaptive_representation_pca_dim,
            adaptive_representation_clusters=adaptive_representation_clusters,
            adaptive_spatial_grid=adaptive_spatial_grid,
            adaptive_grouping=adaptive_grouping,
            adaptive_reference_selection=adaptive_reference_selection,
            adaptive_reference_participates=adaptive_reference_participates,
            adaptive_group_similarity_threshold=adaptive_group_similarity_threshold,
            adaptive_group_max_size=adaptive_group_max_size,
            adaptive_parallel_window=adaptive_parallel_window,
            adaptive_update_policy=adaptive_update_policy,
            adaptive_update_after_blocks=adaptive_update_after_blocks,
            adaptive_frame_fusion=adaptive_frame_fusion,
            adaptive_frame_fusion_weighting=adaptive_frame_fusion_weighting,
            adaptive_frame_token_similarity_threshold=adaptive_frame_token_similarity_threshold,
            adaptive_token_merging=adaptive_token_merging,
            adaptive_token_keep_ratio=adaptive_token_keep_ratio,
            adaptive_token_clusters=adaptive_token_clusters,
            adaptive_token_cluster_budget=adaptive_token_cluster_budget,
            adaptive_token_kmeans_iterations=adaptive_token_kmeans_iterations,
        ).eval()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model" in state:
        state = state["model"]
    missing, _ = model.load_state_dict(state, strict=False, assign=True)
    if missing:
        raise RuntimeError(f"Checkpoint is missing model parameters: {missing[:5]}")
    return model.to(device)


def register_attention_indices(inter_frame_attention: str) -> list[int] | None:
    if inter_frame_attention == "partial":
        return None
    if inter_frame_attention == "global":
        return []
    if inter_frame_attention == "register":
        return list(range(AGGREGATOR_DEPTH))
    if inter_frame_attention == "partial_plus17":
        return [2, 6, 9, 14, 17, 20]
    if inter_frame_attention == "partial_plus3":
        return [2, 3, 6, 9, 14, 20]
    if inter_frame_attention == "partial_pairs":
        return [2, 3, 6, 7, 9, 10, 14, 15, 20, 21]
    if inter_frame_attention == "partial_plus3_17":
        return [2, 3, 6, 9, 14, 17, 20]
    if inter_frame_attention == "partial_2_10_17_20":
        return list(range(2, 11)) + list(range(17, 21))
    if inter_frame_attention == "alternate1":
        return list(range(1, AGGREGATOR_DEPTH, 2))
    if inter_frame_attention == "alternate2":
        return list(range(2, AGGREGATOR_DEPTH, 3))
    raise ValueError(
        f"Unknown inter-frame attention mode {inter_frame_attention!r}; "
        f"expected one of {INTER_FRAME_ATTENTION_MODES}"
    )


def parse_block_indices(value: str) -> list[int]:
    """Parse comma-separated 0-based block indices/ranges such as ``1-7,9``."""
    indices: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise ValueError(f"Invalid descending block range {item!r}")
            indices.update(range(start, end + 1))
        else:
            indices.add(int(item))
    invalid = sorted(index for index in indices if not 0 <= index < AGGREGATOR_DEPTH)
    if invalid:
        raise ValueError(f"register-only block indices must be in [0, {AGGREGATOR_DEPTH - 1}], got {invalid}")
    return sorted(indices)


def windows(paths: Sequence[str], window_size: int) -> Iterable[Sequence[str]]:
    if window_size <= 0 or window_size >= len(paths):
        yield paths
        return
    for start in range(0, len(paths), window_size):
        yield paths[start : start + window_size]


def infer_sequence(
    image_paths: Sequence[str],
    model: VGGTOmega,
    device: torch.device,
    *,
    window_size: int = 4,
    image_resolution: int = 512,
    input_mode: str = "balanced",
    use_amp: bool = True,
) -> tuple[
    float,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    tuple[int, int],
    dict[str, float | int | str | None],
]:
    all_depths = []
    all_poses = []
    all_confidences = []
    frame_merge_stats = []
    frame_special_cross_attention_stats = []
    token_merging_stats = []
    adaptive_fusion_stats = []
    segment_patch_bank_stats = []
    sparse_vggt_stats = []
    da_vggt_stats = []
    da_chunk_strided_shared_anchor_stats = []
    shared_anchor_chunk_stats = []
    output_resolution = None
    elapsed = 0.0
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        memory_before_bytes = torch.cuda.memory_allocated(device)
        reserved_before_bytes = torch.cuda.memory_reserved(device)
    else:
        memory_before_bytes = 0
        reserved_before_bytes = 0

    for image_window in windows(image_paths, window_size):
        images = load_and_preprocess_images(
            list(image_window),
            mode=input_mode,
            image_resolution=image_resolution,
        ).to(device, non_blocking=True)
        output_resolution = (images.shape[-1], images.shape[-2])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.inference_mode():
            if _uses_depth_uncertainty_sampling(model):
                first_predictions = model(images, use_amp=use_amp)
                patch_importance = _patch_importance_from_depth_conf(
                    first_predictions["depth_conf"],
                    images,
                    patch_size=model.aggregator.patch_size,
                )
                predictions = model(images, use_amp=use_amp, patch_importance=patch_importance)
            else:
                predictions = model(images, use_amp=use_amp)
        frame_merge_stats.extend(predictions.get("frame_merge_stats", []))
        frame_special_cross_attention_stats.extend(predictions.get("frame_special_cross_attention_stats", []))
        token_merging_stats.extend(predictions.get("token_merging_stats", []))
        adaptive_fusion_stats.extend(predictions.get("adaptive_frame_token_fusion_stats", []))
        segment_patch_bank_stats.extend(predictions.get("segment_patch_bank_stats", []))
        sparse_vggt_stats.extend(predictions.get("sparse_vggt_stats", []))
        if "da_vggt_stats" in predictions:
            da_vggt_stats.append(predictions["da_vggt_stats"])
        if "da_chunk_strided_shared_anchor_stats" in predictions:
            da_chunk_strided_shared_anchor_stats.append(predictions["da_chunk_strided_shared_anchor_stats"])
        if "shared_anchor_chunk_stats" in predictions:
            shared_anchor_chunk_stats.append(predictions["shared_anchor_chunk_stats"])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed += time.perf_counter() - start

        if "depth" in predictions:
            all_depths.append(predictions["depth"][0, ..., 0].detach().cpu())
            all_confidences.append(predictions["depth_conf"][0].detach().cpu())
        if "pose_enc" in predictions:
            w2c, _ = encoding_to_camera(predictions["pose_enc"], predictions["images"].shape[-2:])
            bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=w2c.device, dtype=w2c.dtype)
            bottom = bottom.view(1, 1, 1, 4).expand(w2c.shape[0], w2c.shape[1], -1, -1)
            c2w = torch.linalg.inv(torch.cat([w2c, bottom], dim=-2))
            all_poses.append(c2w[0].detach().cpu())

    if output_resolution is None:
        raise ValueError("At least one input image is required for inference.")
    depths = torch.cat(all_depths) if all_depths else None
    poses = torch.cat(all_poses) if all_poses else None
    confidences = torch.cat(all_confidences) if all_confidences else None
    if device.type == "cuda":
        peak_memory_allocated_bytes = torch.cuda.max_memory_allocated(device)
        peak_memory_reserved_bytes = torch.cuda.max_memory_reserved(device)
        memory_after_bytes = torch.cuda.memory_allocated(device)
        reserved_after_bytes = torch.cuda.memory_reserved(device)
    else:
        peak_memory_allocated_bytes = 0
        peak_memory_reserved_bytes = 0
        memory_after_bytes = 0
        reserved_after_bytes = 0
    frames = len(image_paths)
    speed_metrics = {
        "omega_accelerator": getattr(model, "omega_accelerator", "none"),
        "time": elapsed,
        "frames": frames,
        "fps": frames / elapsed if elapsed > 0 else None,
        "seconds_per_frame": elapsed / frames if frames > 0 else None,
        "peak_memory_allocated_bytes": peak_memory_allocated_bytes,
        "peak_memory_allocated_gb": peak_memory_allocated_bytes / 1024**3,
        "peak_memory_reserved_bytes": peak_memory_reserved_bytes,
        "peak_memory_reserved_gb": peak_memory_reserved_bytes / 1024**3,
        "memory_before_allocated_bytes": memory_before_bytes,
        "memory_before_allocated_gb": memory_before_bytes / 1024**3,
        "memory_after_allocated_bytes": memory_after_bytes,
        "memory_after_allocated_gb": memory_after_bytes / 1024**3,
        "reserved_before_bytes": reserved_before_bytes,
        "reserved_before_gb": reserved_before_bytes / 1024**3,
        "reserved_after_bytes": reserved_after_bytes,
        "reserved_after_gb": reserved_after_bytes / 1024**3,
        "device": str(device),
    }
    if frame_merge_stats:
        active_means = [float(stat["active_frames_mean"]) for stat in frame_merge_stats]
        retention_ratios = [float(stat["retention_ratio_mean"]) for stat in frame_merge_stats]
        merge_ratios = [float(stat["merge_ratio_mean"]) for stat in frame_merge_stats]
        raw_merge_ratios = [
            float(stat["raw_merge_ratio_mean"])
            for stat in frame_merge_stats
            if stat.get("raw_merge_ratio_mean") is not None
        ]
        policies = [stat.get("adaptive_policy") for stat in frame_merge_stats if stat.get("adaptive_policy")]
        selected_pair_thresholds = [
            float(stat["selected_pair_threshold_mean"])
            for stat in frame_merge_stats
            if stat.get("selected_pair_threshold_mean") is not None
        ]
        selected_span_thresholds = [
            float(stat["selected_span_threshold_mean"])
            for stat in frame_merge_stats
            if stat.get("selected_span_threshold_mean") is not None
        ]
        anchor_counts = [int(stat.get("anchor_count", 0)) for stat in frame_merge_stats]
        anchor_selections = [stat.get("anchor_selection") for stat in frame_merge_stats if stat.get("anchor_selection")]
        frame_fusion_cuda_ms = [
            float(stat["frame_fusion_cuda_ms"])
            for stat in frame_merge_stats
            if stat.get("frame_fusion_cuda_ms") is not None
        ]
        frame_fusion_host_wall_ms = [
            float(stat["frame_fusion_host_wall_ms"])
            for stat in frame_merge_stats
            if stat.get("frame_fusion_host_wall_ms") is not None
        ]
        speed_metrics.update(
            {
                "frame_merge_events": len(frame_merge_stats),
                "frame_merge_original_frames": int(frame_merge_stats[0]["original_frames"]),
                "frame_merge_active_frames_min": min(int(stat["active_frames_min"]) for stat in frame_merge_stats),
                "frame_merge_active_frames_mean": sum(active_means) / len(active_means),
                "frame_merge_active_frames_max": max(int(stat["active_frames_max"]) for stat in frame_merge_stats),
                "frame_merge_retention_ratio_mean": sum(retention_ratios) / len(retention_ratios),
                "frame_merge_merge_ratio_mean": sum(merge_ratios) / len(merge_ratios),
                "frame_merge_raw_merge_ratio_mean": sum(raw_merge_ratios) / len(raw_merge_ratios)
                if raw_merge_ratios
                else None,
                "frame_merge_adaptive_policy": policies[0] if len(set(policies)) == 1 else None,
                "frame_merge_selected_pair_threshold_mean": sum(selected_pair_thresholds)
                / len(selected_pair_thresholds)
                if selected_pair_thresholds
                else None,
                "frame_merge_selected_span_threshold_mean": sum(selected_span_thresholds)
                / len(selected_span_thresholds)
                if selected_span_thresholds
                else None,
                "frame_merge_anchor_count": int(round(sum(anchor_counts) / len(anchor_counts))),
                "frame_merge_anchor_selection": anchor_selections[0] if anchor_selections else None,
                "frame_merge_anchor_frames": frame_merge_stats[0].get("anchor_frames", []),
                "frame_fusion_cuda_ms_mean": sum(frame_fusion_cuda_ms) / len(frame_fusion_cuda_ms)
                if frame_fusion_cuda_ms
                else None,
                "frame_fusion_cuda_ms_total": sum(frame_fusion_cuda_ms) if frame_fusion_cuda_ms else None,
                "frame_fusion_host_wall_ms_mean": sum(frame_fusion_host_wall_ms) / len(frame_fusion_host_wall_ms)
                if frame_fusion_host_wall_ms
                else None,
                "frame_fusion_host_wall_ms_total": sum(frame_fusion_host_wall_ms)
                if frame_fusion_host_wall_ms
                else None,
                "frame_merge_stats": frame_merge_stats,
            }
        )
    if frame_special_cross_attention_stats:
        speed_metrics.update(
            {
                "frame_special_cross_attention_events": len(frame_special_cross_attention_stats),
                "frame_special_cross_attention_alpha": float(
                    frame_special_cross_attention_stats[0]["alpha"]
                ),
                "frame_special_cross_attention_stats": frame_special_cross_attention_stats,
            }
        )
    if token_merging_stats:
        active_tokens = [float(stat["active_tokens"]) for stat in token_merging_stats]
        token_ratios = [float(stat["full_attention_token_ratio"]) for stat in token_merging_stats]
        merged_ratios = [float(stat["merged_away_token_ratio"]) for stat in token_merging_stats]
        active_over_frame_merged_ratios = [
            float(stat.get("active_over_frame_merged_token_ratio", stat["full_attention_token_ratio"]))
            for stat in token_merging_stats
        ]
        active_over_frame_original_ratios = [
            float(stat.get("active_over_frame_original_token_ratio", stat["full_attention_token_ratio"]))
            for stat in token_merging_stats
        ]
        dynamic_patch_tokens = sum(int(stat.get("dynamic_patch_tokens", 0)) for stat in token_merging_stats)
        static_patch_tokens = sum(int(stat.get("static_patch_tokens", 0)) for stat in token_merging_stats)
        dynamic_merged_tokens = sum(
            int(stat.get("dynamic_merged_source_tokens", 0)) for stat in token_merging_stats
        )
        static_merged_tokens = sum(
            int(stat.get("static_merged_source_tokens", 0)) for stat in token_merging_stats
        )
        cross_type_merged_tokens = sum(
            int(stat.get("cross_type_merged_source_tokens", 0)) for stat in token_merging_stats
        )
        speed_metrics.update(
            {
                "token_merging_events": len(token_merging_stats),
                "token_merging_original_tokens": int(token_merging_stats[0]["original_tokens"]),
                "token_merging_active_tokens_mean": sum(active_tokens) / len(active_tokens),
                "token_merging_full_attention_token_ratio_mean": sum(token_ratios) / len(token_ratios),
                "token_merging_merged_away_token_ratio_mean": sum(merged_ratios) / len(merged_ratios),
                "token_merging_active_over_frame_merged_token_ratio_mean": sum(
                    active_over_frame_merged_ratios
                )
                / len(active_over_frame_merged_ratios),
                "token_merging_active_over_frame_original_token_ratio_mean": sum(
                    active_over_frame_original_ratios
                )
                / len(active_over_frame_original_ratios),
                "dynamic_fastvggt_dynamic_patch_tokens": dynamic_patch_tokens,
                "dynamic_fastvggt_static_patch_tokens": static_patch_tokens,
                "dynamic_fastvggt_dynamic_merged_tokens": dynamic_merged_tokens,
                "dynamic_fastvggt_static_merged_tokens": static_merged_tokens,
                "dynamic_fastvggt_dynamic_merge_ratio": float(
                    dynamic_merged_tokens / dynamic_patch_tokens if dynamic_patch_tokens else 0.0
                ),
                "dynamic_fastvggt_static_merge_ratio": float(
                    static_merged_tokens / static_patch_tokens if static_patch_tokens else 0.0
                ),
                "dynamic_fastvggt_patch_merge_ratio": float(
                    (dynamic_merged_tokens + static_merged_tokens)
                    / (dynamic_patch_tokens + static_patch_tokens)
                    if dynamic_patch_tokens + static_patch_tokens
                    else 0.0
                ),
                "dynamic_fastvggt_cross_type_merged_tokens": cross_type_merged_tokens,
                "token_merging_stats": token_merging_stats,
            }
        )
    if adaptive_fusion_stats:
        adaptive_batches = [batch for stat in adaptive_fusion_stats for batch in stat["batches"]]
        frame_ratios = [float(batch["frame_merge_ratio"]) for batch in adaptive_batches]
        frame_token_ratios = [float(batch["frame_fusion_token_ratio"]) for batch in adaptive_batches]
        token_ratios = [float(batch["token_retention_ratio"]) for batch in adaptive_batches]
        token_pre_frame_ratios = [
            float(batch["token_merging_over_pre_frame_token_ratio"]) for batch in adaptive_batches
        ]
        active_frames = [float(batch["active_frames"]) for batch in adaptive_batches]
        speed_metrics.update(
            {
                "adaptive_fusion_events": len(adaptive_fusion_stats),
                "adaptive_fusion_active_frames_mean": sum(active_frames) / len(active_frames),
                "adaptive_fusion_frame_merge_ratio_mean": sum(frame_ratios) / len(frame_ratios),
                "adaptive_fusion_frame_token_ratio_mean": sum(frame_token_ratios) / len(frame_token_ratios),
                "adaptive_fusion_token_retention_ratio_mean": sum(token_ratios) / len(token_ratios),
                "adaptive_fusion_token_over_frame_fused_token_ratio_mean": sum(token_ratios) / len(token_ratios),
                "adaptive_fusion_token_over_pre_frame_token_ratio_mean": sum(token_pre_frame_ratios)
                / len(token_pre_frame_ratios),
                "adaptive_fusion_stats": adaptive_fusion_stats,
            }
        )
    if segment_patch_bank_stats:
        compressed_frames = [int(stat["compressed_frames"]) for stat in segment_patch_bank_stats]
        compressed_segments = [int(stat["compressed_segments"]) for stat in segment_patch_bank_stats]
        speed_metrics.update(
            {
                "segment_patch_bank_events": len(segment_patch_bank_stats),
                "segment_patch_bank_compressed_frames_mean": sum(compressed_frames) / len(compressed_frames),
                "segment_patch_bank_compressed_segments_mean": sum(compressed_segments) / len(compressed_segments),
                "segment_patch_bank_stats": segment_patch_bank_stats,
            }
        )
    if sparse_vggt_stats:
        sparsities = [float(stat["sparsity"]) for stat in sparse_vggt_stats]
        speed_metrics.update(
            {
                "sparse_vggt_events": len(sparse_vggt_stats),
                "sparse_vggt_sparsity_mean": sum(sparsities) / len(sparsities),
                "sparse_vggt_sparse_ratio": sparse_vggt_stats[0].get("sparse_ratio"),
                "sparse_vggt_cdf_threshold": sparse_vggt_stats[0].get("cdf_threshold"),
                "sparse_vggt_pool_mode": sparse_vggt_stats[0].get("pool_mode"),
                "sparse_vggt_stats": sparse_vggt_stats,
            }
        )
    if da_vggt_stats:
        speed_metrics.update(
            {
                "da_vggt_events": len(da_vggt_stats),
                "da_vggt_num_chunks_mean": sum(float(stat["da_vggt_num_chunks"]) for stat in da_vggt_stats)
                / len(da_vggt_stats),
                "da_vggt_chunk_size": da_vggt_stats[0].get("da_vggt_chunk_size"),
                "da_vggt_sampling_method": da_vggt_stats[0].get("da_vggt_sampling_method"),
                "da_vggt_stats": da_vggt_stats,
            }
        )
    if da_chunk_strided_shared_anchor_stats:
        speed_metrics.update(
            {
                "da_chunk_strided_shared_anchor_events": len(da_chunk_strided_shared_anchor_stats),
                "da_chunk_strided_outer_chunk_count_mean": sum(
                    float(stat["outer_chunk_count"]) for stat in da_chunk_strided_shared_anchor_stats
                )
                / len(da_chunk_strided_shared_anchor_stats),
                "da_chunk_strided_group_count": da_chunk_strided_shared_anchor_stats[0].get("group_count"),
                "da_chunk_strided_group_sizes": da_chunk_strided_shared_anchor_stats[0].get("group_sizes"),
                "da_chunk_strided_global_anchor_count": da_chunk_strided_shared_anchor_stats[0].get(
                    "global_anchor_count"
                ),
                "da_chunk_strided_shared_anchor_stats": da_chunk_strided_shared_anchor_stats,
            }
        )
    if shared_anchor_chunk_stats:
        speed_metrics.update(
            {
                "shared_anchor_chunk_events": len(shared_anchor_chunk_stats),
                "shared_anchor_num_chunks_mean": sum(
                    float(stat["shared_anchor_num_chunks"]) for stat in shared_anchor_chunk_stats
                )
                / len(shared_anchor_chunk_stats),
                "shared_anchor_chunk_size": shared_anchor_chunk_stats[0].get("shared_anchor_chunk_size"),
                "shared_anchor_count": shared_anchor_chunk_stats[0].get("shared_anchor_count"),
                "shared_anchor_selection": shared_anchor_chunk_stats[0].get("shared_anchor_selection"),
                "shared_anchor_chunk_stats": shared_anchor_chunk_stats,
            }
        )
    return elapsed, depths, poses, confidences, output_resolution, speed_metrics


def _uses_depth_uncertainty_sampling(model: VGGTOmega) -> bool:
    aggregator = getattr(model, "aggregator", None)
    return getattr(aggregator, "register_patch_sample_mode", None) == "depth_uncertainty"


def _patch_importance_from_depth_conf(
    depth_conf: torch.Tensor,
    images: torch.Tensor,
    *,
    patch_size: int,
) -> torch.Tensor:
    if depth_conf.ndim != 4:
        raise ValueError(f"Expected depth_conf shape [B, F, H, W], got {tuple(depth_conf.shape)}")
    if images.ndim == 4:
        images = images.unsqueeze(0)
    batch_size, num_frames, _, height, width = images.shape
    patch_h = height // patch_size
    patch_w = width // patch_size
    if depth_conf.shape[:2] != (batch_size, num_frames):
        raise ValueError(
            "depth_conf batch/frame shape does not match images: "
            f"{tuple(depth_conf.shape[:2])} vs {(batch_size, num_frames)}"
        )
    uncertainty = -depth_conf.float()
    uncertainty = uncertainty[:, :, : patch_h * patch_size, : patch_w * patch_size]
    uncertainty = uncertainty.view(batch_size, num_frames, patch_h, patch_size, patch_w, patch_size)
    return uncertainty.mean(dim=(3, 5)).reshape(batch_size, num_frames, patch_h * patch_w)


def save_depth_preview(depth: np.ndarray, filename: str | Path) -> None:
    valid = np.isfinite(depth)
    if valid.any():
        lo, hi = np.percentile(depth[valid], [1, 99])
        scaled = np.clip((depth - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    else:
        scaled = np.zeros_like(depth)
    colored = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    cv2.imwrite(str(filename), colored)
