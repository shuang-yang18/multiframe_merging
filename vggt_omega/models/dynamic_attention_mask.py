"""Online layer-4 dynamic/static patch segmentation for asymmetric attention."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def _normalize_tokens(tokens: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    temporal = (tokens - tokens.mean(dim=0, keepdim=True)) / (tokens.std(dim=0, keepdim=True, unbiased=False) + eps)
    spatial = (tokens - tokens.mean(dim=(1, 2), keepdim=True)) / (
        tokens.std(dim=(1, 2), keepdim=True, unbiased=False) + eps
    )
    normalized = 0.5 * (temporal + spatial)
    return F.normalize(normalized, dim=-1, eps=eps)


def _kmeans(features: torch.Tensor, clusters: int = 3, iterations: int = 20) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic GPU K-means using farthest-point initialization."""
    centers = [features[0]]
    nearest_distance = ((features - centers[0]) ** 2).sum(dim=-1)
    for _ in range(1, clusters):
        center = features[nearest_distance.argmax()]
        centers.append(center)
        nearest_distance = torch.minimum(nearest_distance, ((features - center) ** 2).sum(dim=-1))
    centers = torch.stack(centers)
    for _ in range(iterations):
        distances = torch.cdist(features, centers).square()
        labels = distances.argmin(dim=1)
        updated = torch.zeros_like(centers)
        updated.index_add_(0, labels, features)
        counts = torch.bincount(labels, minlength=clusters).clamp_min(1).to(features.dtype).unsqueeze(-1)
        updated = updated / counts
        if torch.allclose(updated, centers, rtol=1e-4, atol=1e-5):
            centers = updated
            break
        centers = updated
    distances = torch.cdist(features, centers).square()
    return distances.argmin(dim=1), distances


def _crf_labels(
    unary_cost: torch.Tensor,
    patch_rgb: torch.Tensor,
    spatial_weight: float = 0.9,
    temporal_weight: float = 0.08,
    color_sigma: float = 0.18,
    iterations: int = 10,
) -> torch.Tensor:
    """The same local RGB edge-aware CRF used by the diagnostic segmenter."""
    probabilities = torch.softmax(-unary_cost, dim=-1)

    def affinity(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        squared_distance = ((left - right) ** 2).mean(dim=-1)
        return torch.exp(-squared_distance / (2.0 * color_sigma**2))

    for _ in range(iterations):
        message = torch.zeros_like(probabilities)
        horizontal = spatial_weight * affinity(patch_rgb[:, :, 1:], patch_rgb[:, :, :-1])
        vertical = spatial_weight * affinity(patch_rgb[:, 1:], patch_rgb[:, :-1])
        message[:, :, 1:] += horizontal[..., None] * probabilities[:, :, :-1]
        message[:, :, :-1] += horizontal[..., None] * probabilities[:, :, 1:]
        message[:, 1:] += vertical[..., None] * probabilities[:, :-1]
        message[:, :-1] += vertical[..., None] * probabilities[:, 1:]
        if len(probabilities) > 1:
            temporal = temporal_weight * affinity(patch_rgb[1:], patch_rgb[:-1])
            message[1:] += temporal[..., None] * probabilities[:-1]
            message[:-1] += temporal[..., None] * probabilities[1:]
        probabilities = torch.softmax(-unary_cost + message, dim=-1)
    return probabilities.argmax(dim=-1)


def infer_dynamic_patch_mask(
    patch_tokens: torch.Tensor,
    camera_patch_attention: torch.Tensor,
    images: torch.Tensor,
    patch_grid_size: tuple[int, int],
    *,
    pca_dim: int = 128,
    seed: int = 0,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Return one whole-cluster dynamic mask for every input sequence.

    The segmentation rule matches the diagnostic pipeline: temporal + spatial
    normalized block-4 patch features, PCA-128, K=3, local CRF, then the class
    with the largest equal-frame camera-attention mean is dynamic.
    """
    batch_size, num_frames, patches_per_frame, channels = patch_tokens.shape
    grid_height, grid_width = patch_grid_size
    if patches_per_frame != grid_height * grid_width:
        raise ValueError("Dynamic segmentation patch layout does not match patch_grid_size")
    if camera_patch_attention.shape != (batch_size, num_frames, patches_per_frame):
        raise ValueError("Camera attention layout does not match layer-4 patch tokens")
    if images.shape[:2] != (batch_size, num_frames):
        raise ValueError("Image and patch-token frame layouts differ")

    colors = F.interpolate(
        images.float().reshape(batch_size * num_frames, *images.shape[2:]),
        size=patch_grid_size,
        mode="bilinear",
        align_corners=False,
    ).reshape(batch_size, num_frames, 3, grid_height, grid_width)
    colors = colors.permute(0, 1, 3, 4, 2)
    tokens = patch_tokens.float().reshape(batch_size, num_frames, grid_height, grid_width, channels)
    attention = camera_patch_attention.float().reshape(batch_size, num_frames, grid_height, grid_width)

    masks: list[torch.Tensor] = []
    summaries: list[dict[str, Any]] = []
    for batch_index in range(batch_size):
        normalized = _normalize_tokens(tokens[batch_index])
        flattened = normalized.reshape(-1, channels)
        actual_dim = min(pca_dim, flattened.shape[0], flattened.shape[1])
        centered = flattened - flattened.mean(dim=0, keepdim=True)
        # QR inside pca_lowrank has no CUDA BF16 kernel.  The surrounding model
        # remains autocast; this small segmentation substep intentionally uses FP32.
        with torch.autocast(device_type=patch_tokens.device.type, enabled=False):
            _, _, components = torch.pca_lowrank(centered.float(), q=actual_dim, center=False, niter=2)
        reduced = centered @ components[:, :actual_dim]
        _, distances = _kmeans(reduced)
        distances = distances - distances.min(dim=1, keepdim=True).values
        margin = distances.topk(k=2, dim=1, largest=False).values[:, 1].mean()
        labels = _crf_labels(
            distances.reshape(num_frames, grid_height, grid_width, 3) / margin.clamp_min(1e-6),
            colors[batch_index],
        )
        per_frame_means = torch.full((num_frames, 3), float("nan"), device=labels.device)
        for frame_index in range(num_frames):
            for class_index in range(3):
                values = attention[batch_index, frame_index][labels[frame_index] == class_index]
                if values.numel():
                    per_frame_means[frame_index, class_index] = values.mean()
        class_means = torch.nanmean(per_frame_means, dim=0)
        dynamic_class = int(class_means.argmax().item())
        masks.append((labels == dynamic_class).reshape(num_frames, patches_per_frame))
        summaries.append(
            {
                "source_block": 4,
                "pca_dim": int(actual_dim),
                "clusters": 3,
                "dynamic_class": dynamic_class,
                "equal_frame_class_mean_attention": [float(value) for value in class_means.detach().cpu()],
                "dynamic_patch_fraction": float((labels == dynamic_class).float().mean().item()),
            }
        )
    return torch.stack(masks, dim=0).to(device=patch_tokens.device, dtype=torch.bool), summaries
