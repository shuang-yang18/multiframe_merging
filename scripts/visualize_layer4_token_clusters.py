#!/usr/bin/env python3
"""Visualize PCA + K-means patch clusters from a VGGT-Omega global layer.

This is a read-only diagnostic utility. It does not alter the regular
inference path or any acceleration method. The script extracts the complete
patch-token output after a requested global aggregator block, normalizes it
in both temporal and spatial directions, then clusters selected frames'
patches jointly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from vggt_omega.evaluation import load_model
from vggt_omega.utils.load_fn import load_and_preprocess_images


PALETTE = np.asarray(
    [
        (228, 26, 28),
        (55, 126, 184),
        (77, 175, 74),
        (152, 78, 163),
        (255, 127, 0),
        (255, 255, 51),
        (166, 86, 40),
        (247, 129, 191),
        (27, 158, 119),
        (217, 95, 2),
        (117, 112, 179),
        (102, 166, 30),
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("tum_dynamic", "7scenes", "nrgbd"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--sequence",
        required=True,
        help="TUM/NRGBD directory name or 7Scenes relative path, e.g. chess/seq-01.",
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/vggt_omega_1b_512.pt"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--feature-block",
        type=int,
        default=4,
        help="Global aggregator block whose patch tokens are clustered.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--input-mode", choices=("balanced", "max_size"), default="balanced")
    parser.add_argument("--max-source-frames", type=int, default=300)
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument(
        "--processing-window-size",
        type=int,
        default=0,
        help="Run layer-4 extraction in consecutive windows; 0 processes all selected frames jointly.",
    )
    parser.add_argument(
        "--frame-indices",
        default="",
        help="Comma-separated source indices. If omitted, sample evenly from the first max-source-frames.",
    )
    parser.add_argument("--pca-dim", type=int, default=32, help="Single PCA dimension for a one-off run.")
    parser.add_argument("--clusters", type=int, default=4, help="Single K for a one-off run.")
    parser.add_argument(
        "--pca-dims",
        default="",
        help="Comma-separated PCA dimensions for a grid. Overrides --pca-dim when supplied.",
    )
    parser.add_argument(
        "--cluster-counts",
        default="",
        help="Comma-separated K values for a grid. Overrides --clusters when supplied.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overlay-alpha", type=float, default=0.46)
    parser.add_argument("--save-visualizations", action="store_true", help="Write PNG overlays and grids (disabled by default).")
    parser.add_argument("--quality-sample-size", type=int, default=3000)
    parser.add_argument("--save-pca-features", action="store_true")
    parser.add_argument("--label-smoothing", choices=("none", "potts", "crf"), default="none")
    parser.add_argument("--potts-spatial-weight", type=float, default=0.12)
    parser.add_argument("--potts-temporal-weight", type=float, default=0.06)
    parser.add_argument("--potts-iterations", type=int, default=5)
    parser.add_argument("--crf-spatial-weight", type=float, default=0.9)
    parser.add_argument("--crf-temporal-weight", type=float, default=0.08)
    parser.add_argument("--crf-color-sigma", type=float, default=0.18)
    parser.add_argument("--crf-unary-temperature", type=float, default=1.0)
    parser.add_argument("--crf-iterations", type=int, default=10)
    parser.add_argument("--camera-attention", action="store_true", help="Render each frame camera token's own-frame patch attention.")
    parser.add_argument("--camera-attention-block", type=int, default=4)
    parser.add_argument(
        "--camera-attention-global-dynamic",
        action="store_true",
        help="Mark the globally highest mean camera-attention cluster dynamic across every selected frame.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def list_images(dataset: str, root: Path, sequence: str) -> list[Path]:
    sequence_root = root / sequence
    if dataset == "tum_dynamic":
        paths = sorted((sequence_root / "rgb").glob("*.png"))
    elif dataset == "nrgbd":
        paths = sorted(
            (sequence_root / "images").glob("img*.png"),
            key=lambda path: int(path.stem.removeprefix("img")),
        )
    else:
        paths = sorted(sequence_root.glob("*.color.png"))
    if not paths:
        raise FileNotFoundError(f"No RGB images found in {sequence_root}")
    return paths


def select_indices(total: int, args: argparse.Namespace) -> np.ndarray:
    if args.frame_indices:
        indices = np.asarray([int(value) for value in args.frame_indices.split(",")], dtype=np.int64)
    else:
        available = min(total, args.max_source_frames)
        if not 1 <= args.num_frames <= available:
            raise ValueError(f"num-frames must be in [1, {available}], got {args.num_frames}")
        indices = np.linspace(0, available - 1, args.num_frames).round().astype(np.int64)
    if len(indices) == 0 or np.any(indices < 0) or np.any(indices >= total):
        raise ValueError(f"Invalid selected indices for {total} source frames: {indices.tolist()}")
    if len(np.unique(indices)) != len(indices):
        raise ValueError(f"Selected indices must be unique: {indices.tolist()}")
    return indices


def parse_positive_csv(value: str, fallback: int, name: str) -> list[int]:
    values = [int(item) for item in value.split(",") if item.strip()] if value else [fallback]
    values = sorted(set(values))
    if not values or any(item < 1 for item in values):
        raise ValueError(f"{name} must contain positive integers, got {value!r}")
    return values


def normalize_tokens(tokens: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Apply complementary temporal and spatial z-score normalization.

    `tokens` has shape [frames, patch_height, patch_width, channels].
    Temporal normalization compares the same spatial patch through time;
    spatial normalization compares patches within every frame.  Averaging the
    two retains both kinds of variation while removing their scale imbalance.
    """
    temporal = (tokens - tokens.mean(axis=0, keepdims=True)) / (tokens.std(axis=0, keepdims=True) + eps)
    spatial = (tokens - tokens.mean(axis=(1, 2), keepdims=True)) / (
        tokens.std(axis=(1, 2), keepdims=True) + eps
    )
    normalized = 0.5 * (temporal + spatial)
    return normalized / (np.linalg.norm(normalized, axis=-1, keepdims=True) + eps)


def spatial_smoothness(labels: np.ndarray) -> float:
    matches: list[np.ndarray] = []
    if labels.shape[2] > 1:
        matches.append(labels[:, :, :-1] == labels[:, :, 1:])
    if labels.shape[1] > 1:
        matches.append(labels[:, :-1, :] == labels[:, 1:, :])
    return float(np.concatenate([match.reshape(-1) for match in matches]).mean()) if matches else float("nan")


def temporal_consistency(labels: np.ndarray) -> float:
    if labels.shape[0] < 2:
        return float("nan")
    return float((labels[:-1] == labels[1:]).mean())


def crop_box(image: Image.Image) -> tuple[int, int, int, int]:
    width, height = image.size
    ratio = height / max(width, 1)
    if ratio < 0.5:
        crop_width = min(width, max(1, int(round(height / 0.5))))
        left = max((width - crop_width) // 2, 0)
        return left, 0, left + crop_width, height
    if ratio > 2.0:
        crop_height = min(height, max(1, int(round(width * 2.0))))
        top = max((height - crop_height) // 2, 0)
        return 0, top, width, top + crop_height
    return 0, 0, width, height


def render_overlay(image_path: Path, labels: np.ndarray, alpha: float, caption: str) -> Image.Image:
    with Image.open(image_path) as source:
        original = source.convert("RGB")
    box = crop_box(original)
    crop = original.crop(box)
    color = Image.fromarray(PALETTE[labels % len(PALETTE)], mode="RGB")
    color = color.resize(crop.size, Image.Resampling.NEAREST)
    overlay = Image.blend(crop, color, alpha)
    rendered = original.copy()
    rendered.paste(overlay, box[:2])
    draw = ImageDraw.Draw(rendered)
    draw.rectangle((0, 0, min(rendered.width, 460), 28), fill=(0, 0, 0))
    draw.text((6, 7), caption, fill=(255, 255, 255))
    return rendered


def render_dynamic_overlay(image_path: Path, dynamic_mask: np.ndarray, alpha: float, caption: str) -> Image.Image:
    """Render blue static and red dynamic regions on the original image."""
    with Image.open(image_path) as source:
        original = source.convert("RGB")
    box = crop_box(original)
    crop = original.crop(box)
    colors = np.zeros((*dynamic_mask.shape, 3), dtype=np.uint8)
    colors[~dynamic_mask] = (55, 126, 184)
    colors[dynamic_mask] = (228, 26, 28)
    color = Image.fromarray(colors, mode="RGB").resize(crop.size, Image.Resampling.NEAREST)
    overlay = Image.blend(crop, color, alpha)
    rendered = original.copy()
    rendered.paste(overlay, box[:2])
    draw = ImageDraw.Draw(rendered)
    draw.rectangle((0, 0, min(rendered.width, 520), 48), fill=(0, 0, 0))
    draw.text((6, 6), caption, fill=(255, 255, 255))
    draw.text((6, 26), "blue=static, red=dynamic", fill=(255, 255, 255))
    return rendered


def render_camera_attention_overlay(
    image_path: Path,
    attention: np.ndarray,
    labels: np.ndarray,
    caption: str,
) -> tuple[Image.Image, list[dict[str, float | int]]]:
    """Overlay one camera token's same-frame patch attention on the RGB image."""
    with Image.open(image_path) as source:
        original = source.convert("RGB")
    box = crop_box(original)
    crop = original.crop(box)
    low, high = np.quantile(attention, (0.05, 0.99))
    normalized = np.clip((attention - low) / max(float(high - low), 1e-8), 0.0, 1.0)
    heatmap = cv2.applyColorMap(np.round(normalized * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = Image.fromarray(heatmap[:, :, ::-1], mode="RGB").resize(crop.size, Image.Resampling.NEAREST)
    overlay = Image.blend(crop, heatmap, 0.44)
    rendered = original.copy()
    rendered.paste(overlay, box[:2])
    mean_attention = float(attention.mean())
    region_stats: list[dict[str, float | int]] = []
    for cluster_id in range(int(labels.max()) + 1):
        region = attention[labels == cluster_id]
        region_mean = float(region.mean()) if len(region) else 0.0
        region_stats.append(
            {
                "cluster": cluster_id,
                "patch_count": int(len(region)),
                "attention_mass": float(region.sum()),
                "mean_attention": region_mean,
                "relative_density": region_mean / max(mean_attention, 1e-12),
            }
        )
    density_text = " ".join(f"C{item['cluster']}={item['relative_density']:.2f}x" for item in region_stats)
    draw = ImageDraw.Draw(rendered)
    draw.rectangle((0, 0, min(rendered.width, 640), 48), fill=(0, 0, 0))
    draw.text((6, 6), caption, fill=(255, 255, 255))
    draw.text((6, 26), f"blue=low, red=high | {density_text}", fill=(255, 255, 255))
    return rendered, region_stats


def make_grid(images: list[Image.Image]) -> Image.Image:
    widths = [image.width for image in images]
    heights = [image.height for image in images]
    cell_width = max(widths)
    cell_height = max(heights)
    columns = min(2, len(images))
    rows = (len(images) + columns - 1) // columns
    grid = Image.new("RGB", (columns * cell_width, rows * cell_height), color=(18, 18, 18))
    for index, image in enumerate(images):
        col = index % columns
        row = index // columns
        grid.paste(image, (col * cell_width, row * cell_height))
    return grid


def sampled_quality_metrics(reduced: np.ndarray, labels: np.ndarray, sample_size: int, seed: int) -> dict[str, float | int]:
    if sample_size < 2:
        raise ValueError("quality-sample-size must be at least 2")
    rng = np.random.default_rng(seed)
    sample_count = min(sample_size, len(labels))
    sample_indices = np.arange(len(labels)) if sample_count == len(labels) else rng.choice(len(labels), sample_count, replace=False)
    sampled_features = reduced[sample_indices]
    sampled_labels = labels[sample_indices]
    return {
        "quality_sample_count": int(sample_count),
        "silhouette": float(silhouette_score(sampled_features, sampled_labels)),
        "calinski_harabasz": float(calinski_harabasz_score(sampled_features, sampled_labels)),
        "davies_bouldin": float(davies_bouldin_score(sampled_features, sampled_labels)),
    }


def potts_smooth_labels(
    initial_labels: np.ndarray,
    unary_cost: np.ndarray,
    spatial_weight: float,
    temporal_weight: float,
    iterations: int,
) -> np.ndarray:
    """Apply parallel ICM updates for a spatial-temporal Potts model."""
    if spatial_weight < 0.0 or temporal_weight < 0.0 or iterations < 1:
        raise ValueError("Potts weights must be non-negative and iterations must be positive")
    labels = initial_labels.copy()
    num_clusters = unary_cost.shape[-1]
    candidates = np.arange(num_clusters, dtype=labels.dtype)
    for _ in range(iterations):
        energy = unary_cost.copy()
        if labels.shape[2] > 1:
            energy[:, :, 1:, :] += spatial_weight * (labels[:, :, :-1, None] != candidates)
            energy[:, :, :-1, :] += spatial_weight * (labels[:, :, 1:, None] != candidates)
        if labels.shape[1] > 1:
            energy[:, 1:, :, :] += spatial_weight * (labels[:, :-1, :, None] != candidates)
            energy[:, :-1, :, :] += spatial_weight * (labels[:, 1:, :, None] != candidates)
        if labels.shape[0] > 1:
            energy[1:, :, :, :] += temporal_weight * (labels[:-1, :, :, None] != candidates)
            energy[:-1, :, :, :] += temporal_weight * (labels[1:, :, :, None] != candidates)
        updated = energy.argmin(axis=-1).astype(labels.dtype, copy=False)
        if np.array_equal(updated, labels):
            break
        labels = updated
    return labels


def load_patch_rgb_features(image_paths: list[Path], grid_height: int, grid_width: int) -> np.ndarray:
    """Downsample the cropped original RGB images to the model patch grid."""
    colors = []
    for image_path in image_paths:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        image = image.crop(crop_box(image)).resize((grid_width, grid_height), Image.Resampling.BILINEAR)
        colors.append(np.asarray(image, dtype=np.float32) / 255.0)
    return np.stack(colors, axis=0)


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(axis=-1, keepdims=True)


def edge_aware_crf_labels(
    unary_cost: np.ndarray,
    patch_rgb: np.ndarray,
    spatial_weight: float,
    temporal_weight: float,
    color_sigma: float,
    unary_temperature: float,
    iterations: int,
) -> np.ndarray:
    """Mean-field inference for an RGB edge-aware local spatial-temporal CRF.

    This is deliberately local (four spatial neighbours plus adjacent selected
    frames), rather than a fully connected DenseCRF.  It is a better fit for
    the 28x37 patch lattice and avoids a heavyweight external dependency.
    """
    if min(spatial_weight, temporal_weight) < 0.0 or color_sigma <= 0.0 or unary_temperature <= 0.0 or iterations < 1:
        raise ValueError("CRF weights must be non-negative; sigma, temperature, and iterations must be positive")
    probabilities = _softmax(-unary_cost / unary_temperature)

    def affinity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        squared_color_distance = ((left - right) ** 2).mean(axis=-1)
        return np.exp(-squared_color_distance / (2.0 * color_sigma**2)).astype(np.float32)

    for _ in range(iterations):
        message = np.zeros_like(probabilities)
        if probabilities.shape[2] > 1:
            weights = spatial_weight * affinity(patch_rgb[:, :, 1:], patch_rgb[:, :, :-1])
            message[:, :, 1:] += weights[..., None] * probabilities[:, :, :-1]
            message[:, :, :-1] += weights[..., None] * probabilities[:, :, 1:]
        if probabilities.shape[1] > 1:
            weights = spatial_weight * affinity(patch_rgb[:, 1:], patch_rgb[:, :-1])
            message[:, 1:] += weights[..., None] * probabilities[:, :-1]
            message[:, :-1] += weights[..., None] * probabilities[:, 1:]
        if probabilities.shape[0] > 1:
            weights = temporal_weight * affinity(patch_rgb[1:], patch_rgb[:-1])
            message[1:] += weights[..., None] * probabilities[:-1]
            message[:-1] += weights[..., None] * probabilities[1:]
        probabilities = _softmax(-unary_cost / unary_temperature + message)
    return probabilities.argmax(axis=-1).astype(np.int64)


def camera_attention_global_dynamic_mask(attention_maps: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    """Select one K=3 cluster by equal-weighted per-frame attention means."""
    if int(labels.max()) != 2:
        raise ValueError("camera-attention-global-dynamic requires exactly K=3")
    per_frame_means = np.full((len(labels), 3), np.nan, dtype=np.float64)
    for frame_index, (frame_attention, frame_labels) in enumerate(zip(attention_maps, labels)):
        for class_id in range(3):
            values = frame_attention[frame_labels == class_id]
            if values.size:
                per_frame_means[frame_index, class_id] = float(values.mean())
    means = np.nanmean(per_frame_means, axis=0).tolist()
    dynamic_class = int(np.argmax(means))
    dynamic_masks = labels == dynamic_class
    serialized_per_frame_means = [
        [float(value) if np.isfinite(value) else None for value in row] for row in per_frame_means
    ]
    return dynamic_masks, {
        "rule": "dynamic_class = argmax_i mean_t(mean(camera_attention[t] | class[t]=i))",
        "per_frame_class_mean_attention": serialized_per_frame_means,
        "equal_frame_class_mean_attention": means,
        "dynamic_class": dynamic_class,
    }


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.overlay_alpha <= 1.0:
        raise ValueError("overlay-alpha must be in [0, 1]")
    if args.potts_spatial_weight < 0.0 or args.potts_temporal_weight < 0.0 or args.potts_iterations < 1:
        raise ValueError("Potts weights must be non-negative and potts-iterations must be positive")
    if (
        args.crf_spatial_weight < 0.0
        or args.crf_temporal_weight < 0.0
        or args.crf_color_sigma <= 0.0
        or args.crf_unary_temperature <= 0.0
        or args.crf_iterations < 1
    ):
        raise ValueError("CRF parameters are invalid")
    if args.camera_attention_global_dynamic and not args.camera_attention:
        raise ValueError("camera-attention-global-dynamic requires --camera-attention")
    if args.processing_window_size < 0:
        raise ValueError("processing-window-size must be non-negative")
    pca_dims = parse_positive_csv(args.pca_dims, args.pca_dim, "pca-dims")
    cluster_counts = parse_positive_csv(args.cluster_counts, args.clusters, "cluster-counts")
    if any(cluster_count < 2 for cluster_count in cluster_counts):
        raise ValueError("Every cluster count must be at least 2")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    image_paths = list_images(args.dataset, args.dataset_root, args.sequence)
    indices = select_indices(len(image_paths), args)
    selected_paths = [image_paths[index] for index in indices]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.checkpoint, device, enable_camera=False, enable_depth=False)
    model.eval()
    if not 0 <= args.feature_block < model.aggregator.depth:
        raise ValueError(f"feature-block must be in [0, {model.aggregator.depth - 1}]")
    if model.aggregator.inter_frame_attention_types[args.feature_block] != "global":
        raise ValueError(f"feature-block={args.feature_block} is not a global attention layer")
    if args.camera_attention and args.camera_attention_block != args.feature_block:
        raise ValueError("camera-attention-block must equal feature-block for layer-wise diagnostics")
    window_size = args.processing_window_size or len(selected_paths)
    patch_token_chunks: list[np.ndarray] = []
    attention_chunks: list[np.ndarray] = []
    patch_start: int | None = None
    processed_size_hw: tuple[int, int] | None = None
    for start in range(0, len(selected_paths), window_size):
        image_window = selected_paths[start : start + window_size]
        images = load_and_preprocess_images(
            [str(path) for path in image_window],
            mode=args.input_mode,
            image_resolution=args.image_resolution,
            patch_size=model.aggregator.patch_size,
        ).unsqueeze(0).to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16,
            enabled=device.type == "cuda",
        ):
            cached_outputs, current_patch_start = model.aggregator(
                images,
                stop_after_block=args.feature_block,
                capture_camera_attention_block=args.camera_attention_block if args.camera_attention else None,
                capture_output_block=args.feature_block,
            )
        feature_tokens = cached_outputs[args.feature_block]
        if feature_tokens is None:
            raise RuntimeError(f"Aggregator did not cache block {args.feature_block} output.")
        current_size = (int(images.shape[-2]), int(images.shape[-1]))
        if patch_start is None:
            patch_start = current_patch_start
            processed_size_hw = current_size
        elif patch_start != current_patch_start or processed_size_hw != current_size:
            raise RuntimeError("Layer-4 extraction windows produced inconsistent token layouts")
        patch_token_chunks.append(feature_tokens[0, :, current_patch_start:, :].float().cpu().numpy())
        if args.camera_attention:
            captured_attention = model.aggregator.last_camera_patch_attention
            if captured_attention is None:
                raise RuntimeError("Camera attention capture was requested but no attention map was returned")
            attention_chunks.append(captured_attention[0].float().cpu().numpy())
        del images, cached_outputs, feature_tokens
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if patch_start is None or processed_size_hw is None:
        raise RuntimeError("No image windows were processed")
    patch_tokens = np.concatenate(patch_token_chunks, axis=0)
    processed_height, processed_width = processed_size_hw
    grid_height = processed_height // model.aggregator.patch_size
    grid_width = processed_width // model.aggregator.patch_size
    if patch_tokens.shape[1] != grid_height * grid_width:
        raise RuntimeError(f"Patch grid mismatch: got {patch_tokens.shape[1]} tokens, expected {grid_height} x {grid_width}.")
    camera_attention = np.concatenate(attention_chunks, axis=0) if args.camera_attention else None
    if camera_attention is not None and camera_attention.shape != (len(selected_paths), grid_height * grid_width):
        raise RuntimeError(f"Unexpected camera attention shape: {camera_attention.shape}")
    tokens_grid = patch_tokens.reshape(len(selected_paths), grid_height, grid_width, -1)
    normalized = normalize_tokens(tokens_grid)
    flattened = normalized.reshape(-1, normalized.shape[-1])
    max_pca_dim = min(max(pca_dims), flattened.shape[0], flattened.shape[1])
    pca = PCA(n_components=max_pca_dim, svd_solver="randomized", random_state=args.seed)
    reduced_max = pca.fit_transform(flattened)
    common_config = {
        "dataset": args.dataset,
        "dataset_root": str(args.dataset_root),
        "sequence": args.sequence,
        "source_frame_count": len(image_paths),
        "selected_indices": indices.tolist(),
        "selected_images": [str(path) for path in selected_paths],
        "processing_window_size": window_size,
        "attention_context": "consecutive extraction window; all selected frames are jointly clustered after extraction",
        "aggregator_feature_block_index": args.feature_block,
        "aggregator_stop_after_block": args.feature_block,
        "patch_token_start": int(patch_start),
        "processed_size_hw": [int(processed_height), int(processed_width)],
        "patch_grid_hw": [int(grid_height), int(grid_width)],
        "normalization": "0.5 * temporal_zscore(same patch across selected frames) + spatial_zscore(patches within frame), then L2",
        "seed": args.seed,
        "overlay_alpha": args.overlay_alpha,
        "save_visualizations": args.save_visualizations,
        "pca_dims_requested": pca_dims,
        "cluster_counts_requested": cluster_counts,
        "label_smoothing": args.label_smoothing,
        "potts_spatial_weight": args.potts_spatial_weight if args.label_smoothing == "potts" else None,
        "potts_temporal_weight": args.potts_temporal_weight if args.label_smoothing == "potts" else None,
        "potts_iterations": args.potts_iterations if args.label_smoothing == "potts" else None,
        "crf_spatial_weight": args.crf_spatial_weight if args.label_smoothing == "crf" else None,
        "crf_temporal_weight": args.crf_temporal_weight if args.label_smoothing == "crf" else None,
        "crf_color_sigma": args.crf_color_sigma if args.label_smoothing == "crf" else None,
        "crf_unary_temperature": args.crf_unary_temperature if args.label_smoothing == "crf" else None,
        "crf_iterations": args.crf_iterations if args.label_smoothing == "crf" else None,
        "camera_attention": args.camera_attention,
        "camera_attention_block": args.camera_attention_block if args.camera_attention else None,
        "camera_attention_global_dynamic": args.camera_attention_global_dynamic,
    }
    (args.output_dir / "common_config.json").write_text(json.dumps(common_config, indent=2) + "\n")
    patch_rgb = load_patch_rgb_features(selected_paths, grid_height, grid_width) if args.label_smoothing == "crf" else None
    summaries = []
    for pca_dim in pca_dims:
        actual_dim = min(pca_dim, reduced_max.shape[1])
        reduced = reduced_max[:, :actual_dim]
        explained_variance = float(pca.explained_variance_ratio_[:actual_dim].sum())
        for cluster_count in cluster_counts:
            if cluster_count > len(reduced):
                raise ValueError(f"K={cluster_count} exceeds {len(reduced)} patch tokens")
            result_dir = args.output_dir / f"pca{actual_dim:03d}_k{cluster_count:02d}"
            marker = result_dir / "metrics.json"
            if marker.is_file() and not args.overwrite:
                print(f"Skipping complete result: {result_dir}")
                continue
            result_dir.mkdir(parents=True, exist_ok=True)
            kmeans = KMeans(n_clusters=cluster_count, n_init=20, random_state=args.seed)
            raw_flat_labels = kmeans.fit_predict(reduced)
            raw_labels = raw_flat_labels.reshape(len(selected_paths), grid_height, grid_width)
            if args.label_smoothing in {"potts", "crf"}:
                distances = kmeans.transform(reduced).astype(np.float32) ** 2
                distances -= distances.min(axis=1, keepdims=True)
                margin = np.partition(distances, 1, axis=1)[:, 1].mean()
                unary = distances / max(float(margin), 1e-6)
            if args.label_smoothing == "potts":
                labels = potts_smooth_labels(
                    raw_labels,
                    unary.reshape(len(selected_paths), grid_height, grid_width, cluster_count),
                    args.potts_spatial_weight,
                    args.potts_temporal_weight,
                    args.potts_iterations,
                )
            elif args.label_smoothing == "crf":
                if patch_rgb is None:
                    raise RuntimeError("CRF requires patch RGB features")
                labels = edge_aware_crf_labels(
                    unary.reshape(len(selected_paths), grid_height, grid_width, cluster_count),
                    patch_rgb,
                    args.crf_spatial_weight,
                    args.crf_temporal_weight,
                    args.crf_color_sigma,
                    args.crf_unary_temperature,
                    args.crf_iterations,
                )
            else:
                labels = raw_labels
            flat_labels = labels.reshape(-1)
            counts = np.bincount(flat_labels, minlength=cluster_count)
            metrics = {
                "pca_explained_variance_ratio_sum": explained_variance,
                **sampled_quality_metrics(reduced, raw_flat_labels, args.quality_sample_size, args.seed),
                "temporal_consistency": temporal_consistency(labels),
                "spatial_smoothness": spatial_smoothness(labels),
                "potts_changed_fraction": float((labels != raw_labels).mean()) if args.label_smoothing == "potts" else None,
                "crf_changed_fraction": float((labels != raw_labels).mean()) if args.label_smoothing == "crf" else None,
                "cluster_counts": counts.tolist(),
                "cluster_fractions": (counts / counts.sum()).round(8).tolist(),
            }
            config = {**common_config, "pca_dim": int(actual_dim), "clusters": cluster_count}
            np.save(result_dir / "labels.npy", labels)
            if args.label_smoothing in {"potts", "crf"}:
                np.save(result_dir / "labels_raw.npy", raw_labels)
            if args.save_pca_features:
                np.save(result_dir / "pca_features.npy", reduced.astype(np.float32))
            (result_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
            (result_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
            if args.save_visualizations:
                rendered: list[Image.Image] = []
                for frame_index, (source_index, image_path, frame_labels) in enumerate(zip(indices, selected_paths, labels)):
                    caption = f"source frame {int(source_index)} | PCA {actual_dim} | K {cluster_count}"
                    image = render_overlay(image_path, frame_labels, args.overlay_alpha, caption)
                    image.save(result_dir / f"frame_{frame_index:02d}_source_{int(source_index):04d}_overlay.png")
                    rendered.append(image)
                make_grid(rendered).save(result_dir / "overlay_grid.png")
            if camera_attention is not None:
                attention_dir = result_dir / "camera_attention"
                attention_dir.mkdir(exist_ok=True)
                attention_maps = camera_attention.reshape(len(selected_paths), grid_height, grid_width)
                np.save(attention_dir / "same_frame_patch_attention.npy", attention_maps)
                if args.save_visualizations:
                    attention_images: list[Image.Image] = []
                    attention_summary: list[dict[str, object]] = []
                    for frame_index, (source_index, image_path, attention_map, frame_labels) in enumerate(
                        zip(indices, selected_paths, attention_maps, labels)
                    ):
                        caption = f"source frame {int(source_index)} | camera token self-frame attention"
                        image, region_stats = render_camera_attention_overlay(image_path, attention_map, frame_labels, caption)
                        image.save(attention_dir / f"frame_{frame_index:02d}_source_{int(source_index):04d}_overlay.png")
                        attention_images.append(image)
                        attention_summary.append(
                            {
                                "source_frame": int(source_index),
                                "attention_sum": float(attention_map.sum()),
                                "regions": region_stats,
                            }
                        )
                    make_grid(attention_images).save(attention_dir / "overlay_grid.png")
                    (attention_dir / "region_attention.json").write_text(json.dumps(attention_summary, indent=2) + "\n")
            if args.camera_attention_global_dynamic:
                if camera_attention is None:
                    raise RuntimeError("Camera attention maps are unavailable")
                attention_maps = camera_attention.reshape(len(selected_paths), grid_height, grid_width)
                dynamic_masks, dynamic_summary = camera_attention_global_dynamic_mask(attention_maps, labels)
                dynamic_dir = result_dir / "camera_attention_global_dynamic"
                dynamic_dir.mkdir(exist_ok=True)
                np.save(dynamic_dir / "dynamic_mask.npy", dynamic_masks)
                (dynamic_dir / "scores.json").write_text(
                    json.dumps(dynamic_summary, indent=2, allow_nan=False) + "\n"
                )
                if args.save_visualizations:
                    dynamic_images: list[Image.Image] = []
                    for frame_index, (source_index, image_path, frame_mask) in enumerate(zip(indices, selected_paths, dynamic_masks)):
                        caption = f"source frame {int(source_index)} | global camera-attention dynamic class {dynamic_summary['dynamic_class']}"
                        image = render_dynamic_overlay(image_path, frame_mask, args.overlay_alpha, caption)
                        image.save(dynamic_dir / f"frame_{frame_index:02d}_source_{int(source_index):04d}_overlay.png")
                        dynamic_images.append(image)
                    make_grid(dynamic_images).save(dynamic_dir / "overlay_grid.png")
            summaries.append({"result_dir": str(result_dir), "pca_dim": actual_dim, "clusters": cluster_count, **metrics})
            print(json.dumps(summaries[-1], ensure_ascii=False))
    if summaries:
        with (args.output_dir / "summary.jsonl").open("a") as handle:
            for summary in summaries:
                handle.write(json.dumps(summary, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
