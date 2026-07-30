"""Independent adaptive frame/token fusion for VGGT-Omega global attention.

This module deliberately does not use any legacy frame-merging or FastVGGT
helpers. Every global layer receives a temporary compact layout and returns to
the original [frames, tokens] layout immediately after attention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class AdaptiveFusionConfig:
    frame_representation: str = "global_pool"
    representation_pca_dim: int = 512
    representation_clusters: int = 3
    spatial_grid: int = 4
    grouping: str = "serial"
    reference_selection: str = "first"
    reference_participates: bool = True
    group_similarity_threshold: float = 0.98
    group_max_size: int = 4
    parallel_window: int = 10
    update_policy: str = "initial_only"
    update_after_blocks: tuple[int, ...] = (9, 17)
    frame_fusion: str = "direct"
    frame_fusion_weighting: str = "similarity"
    frame_token_similarity_threshold: float = 0.95
    token_merging: str = "fast_bipartite"
    token_keep_ratio: float = 0.1
    token_clusters: int = 4
    token_cluster_budget: str = "proportional"
    token_kmeans_iterations: int = 12


@dataclass
class FrameFusionState:
    groups: list[list[int]]
    references: list[int]
    active_members: list[list[int]]
    active_inputs: torch.Tensor
    original_tokens: torch.Tensor


@dataclass
class TokenFusionState:
    inverse: torch.Tensor
    packed_input: torch.Tensor
    original_tokens: torch.Tensor
    selected_indices: torch.Tensor
    assignment_weights: torch.Tensor
    selected_count: int
    original_count: int


@dataclass
class SelectiveFrameTokenState:
    """Reversible packed layout for selective frame fusion plus token merging."""

    inverse: torch.Tensor
    packed_input: torch.Tensor
    original_tokens: torch.Tensor
    frame_input_count: int
    reference_patch_count: int
    residual_patch_count: int
    selected_count: int
    original_count: int


def parse_block_list(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    blocks: set[int] = set()
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        if "-" in item:
            start, end = (int(part) for part in item.split("-", 1))
            if end < start:
                raise ValueError(f"Invalid descending block range: {item}")
            blocks.update(range(start, end + 1))
        else:
            blocks.add(int(item))
    return tuple(sorted(blocks))


def _pca(features: torch.Tensor, out_dim: int) -> torch.Tensor:
    """PCA projection with deterministic padding for a requested larger width."""
    # QR, used internally by pca_lowrank, has no CUDA bfloat16 kernel. This
    # representation is only used for discrete grouping, so keep it in fp32
    # even when the surrounding VGGT forward pass uses AMP.
    with torch.autocast(device_type=features.device.type, enabled=False):
        features = features.to(dtype=torch.float32)
        if features.numel() == 0:
            return features.new_zeros((features.shape[0], out_dim))
        centered = features - features.mean(dim=0, keepdim=True)
        rank = min(centered.shape[0], centered.shape[1], out_dim)
        if rank <= 0:
            return features.new_zeros((features.shape[0], out_dim))
        _, _, basis = torch.pca_lowrank(centered, q=rank, center=False)
        projected = centered @ basis[:, :rank]
        if rank == out_dim:
            return projected
        return F.pad(projected, (0, out_dim - rank))


def _kmeans(features: torch.Tensor, clusters: int, iterations: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Small deterministic cosine K-means used only by this new path."""
    count = features.shape[0]
    clusters = max(1, min(int(clusters), count))
    normalized = F.normalize(features.float(), dim=-1)
    initial = torch.linspace(0, count - 1, clusters, device=features.device).round().long()
    centers = normalized[initial]
    labels = torch.zeros(count, dtype=torch.long, device=features.device)
    for _ in range(iterations):
        labels = (normalized @ centers.T).argmax(dim=1)
        sums = torch.zeros_like(centers)
        sums.index_add_(0, labels, normalized)
        sizes = torch.bincount(labels, minlength=clusters).to(sums.dtype).unsqueeze(1)
        centers = F.normalize(torch.where(sizes > 0, sums / sizes.clamp_min(1), centers), dim=-1)
    return labels, centers


def _project_to_frame_dim(features: torch.Tensor, frame_dim: int) -> torch.Tensor:
    if features.shape[-1] == frame_dim:
        return features.float()
    return _pca(features, frame_dim)


def frame_representations(
    tokens: torch.Tensor,
    *,
    patch_start: int,
    grid_size: tuple[int, int],
    config: AdaptiveFusionConfig,
) -> torch.Tensor:
    """Return one shared-width descriptor for every frame in one batch item."""
    patch = tokens[:, patch_start:].float()
    frames, patch_count, channels = patch.shape
    if config.frame_representation == "global_pool":
        return patch.mean(dim=1)
    if config.frame_representation == "cluster_center":
        reduced = _pca(patch.reshape(-1, channels), config.representation_pca_dim)
        labels, centers = _kmeans(reduced, config.representation_clusters, config.token_kmeans_iterations)
        reduced = reduced.view(frames, patch_count, -1)
        labels = labels.view(frames, patch_count)
        per_frame = []
        for frame_idx in range(frames):
            parts = []
            for cluster_idx in range(centers.shape[0]):
                member = reduced[frame_idx][labels[frame_idx] == cluster_idx]
                parts.append(member.mean(dim=0) if member.numel() else centers[cluster_idx])
            per_frame.append(torch.cat(parts))
        return _project_to_frame_dim(torch.stack(per_frame), channels)
    if config.frame_representation == "spatial_grid":
        grid_h, grid_w = grid_size
        if patch_count != grid_h * grid_w:
            raise ValueError("Spatial frame representation received a mismatched patch grid")
        grid = patch.view(frames, grid_h, grid_w, channels)
        parts = []
        for y in range(config.spatial_grid):
            y0, y1 = y * grid_h // config.spatial_grid, (y + 1) * grid_h // config.spatial_grid
            for x in range(config.spatial_grid):
                x0, x1 = x * grid_w // config.spatial_grid, (x + 1) * grid_w // config.spatial_grid
                parts.append(grid[:, y0:y1, x0:x1].mean(dim=(1, 2)))
        return _project_to_frame_dim(torch.cat(parts, dim=-1), channels)
    raise ValueError(f"Unknown adaptive frame representation: {config.frame_representation}")


def similarity_matrix(representations: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(representations.float(), dim=-1)
    return (normalized @ normalized.T).clamp(-1.0, 1.0)


def _reference(indices: list[int], matrix: torch.Tensor, strategy: str) -> int:
    if strategy == "first" or len(indices) == 1:
        return indices[0]
    member = torch.tensor(indices, device=matrix.device)
    scores = matrix[member][:, member].mean(dim=1)
    position = int(scores.argmax().item()) if strategy == "medoid" else int(scores.argmin().item())
    return indices[position]


def _serial_groups(matrix: torch.Tensor, config: AdaptiveFusionConfig) -> tuple[list[list[int]], list[int]]:
    count = matrix.shape[0]
    groups: list[list[int]] = []
    references: list[int] = []
    cursor = 0
    while cursor < count:
        group = [cursor]
        cursor += 1
        while cursor < count and len(group) < config.group_max_size:
            candidate = cursor
            members = torch.tensor(group + [candidate], device=matrix.device)
            if matrix[members][:, members].min() < config.group_similarity_threshold:
                break
            group.append(candidate)
            cursor += 1
        groups.append(group)
        references.append(_reference(group, matrix, config.reference_selection))
    return groups, references


def _parallel_groups(matrix: torch.Tensor, config: AdaptiveFusionConfig) -> tuple[list[list[int]], list[int]]:
    """Create groups around a reference and search its left/right W-frame window."""
    count = matrix.shape[0]
    available = set(range(count))
    grouped: list[tuple[list[int], int]] = []
    while available:
        seed = min(available)
        seed_window = [index for index in available if abs(index - seed) <= config.parallel_window]
        reference = _reference(sorted(seed_window), matrix, config.reference_selection)
        # The actual search region is centered on the selected reference, not
        # on the provisional unassigned seed used to start this group.
        candidates = [
            index
            for index in available
            if index != reference and abs(index - reference) <= config.parallel_window
        ]
        candidates.sort(key=lambda index: float(matrix[reference, index]), reverse=True)
        group = [reference]
        for candidate in candidates:
            if len(group) >= config.group_max_size:
                break
            proposal = torch.tensor(group + [candidate], device=matrix.device)
            if matrix[proposal][:, proposal].min() >= config.group_similarity_threshold:
                group.append(candidate)
        group.sort()
        for index in group:
            available.remove(index)
        grouped.append((group, reference))
    grouped.sort(key=lambda item: item[0][0])
    return [group for group, _ in grouped], [reference for _, reference in grouped]


def build_groups(matrix: torch.Tensor, config: AdaptiveFusionConfig) -> tuple[list[list[int]], list[int]]:
    if config.grouping == "serial":
        return _serial_groups(matrix, config)
    if config.grouping == "parallel":
        return _parallel_groups(matrix, config)
    raise ValueError(f"Unknown adaptive frame grouping: {config.grouping}")


def _fuse_group(
    tokens: torch.Tensor,
    members: list[int],
    reference: int,
    config: AdaptiveFusionConfig,
    patch_start: int,
) -> torch.Tensor:
    reference_tokens = tokens[reference]
    sources = [index for index in members if index != reference]
    if not sources:
        return reference_tokens
    source_tokens = tokens[torch.tensor(sources, device=tokens.device)]
    if config.frame_fusion == "direct":
        if config.frame_fusion_weighting == "uniform":
            weights = torch.ones(len(sources), device=tokens.device, dtype=torch.float32)
        else:
            descriptor = F.normalize(tokens[:, :].float().mean(dim=1), dim=-1)
            weights = (descriptor[sources] @ descriptor[reference]).clamp_min(1e-4)
        merged = reference_tokens.float().clone()
        token_slice = slice(patch_start, None)
        merged[token_slice] = (
            reference_tokens.float()[token_slice]
            + (source_tokens.float()[:, token_slice] * weights[:, None, None]).sum(dim=0)
        ) / (1.0 + weights.sum())
        return merged
    if config.frame_fusion == "token_wise":
        token_slice = slice(patch_start, None)
        similarity = F.cosine_similarity(
            source_tokens.float()[:, token_slice], reference_tokens.float()[token_slice].unsqueeze(0), dim=-1
        )
        selected = similarity >= config.frame_token_similarity_threshold
        weights = selected.float()
        source_sum = (source_tokens.float()[:, token_slice] * weights.unsqueeze(-1)).sum(dim=0)
        source_count = weights.sum(dim=0).clamp_min(1.0).unsqueeze(-1)
        merged = reference_tokens.float().clone()
        candidate = (reference_tokens.float()[token_slice] + source_sum) / (1.0 + source_count)
        merged[token_slice] = torch.where(selected.any(dim=0).unsqueeze(-1), candidate, reference_tokens.float()[token_slice])
        return merged
    raise ValueError(f"Unknown adaptive frame fusion: {config.frame_fusion}")


def fuse_frames(
    tokens: torch.Tensor,
    groups: list[list[int]],
    references: list[int],
    config: AdaptiveFusionConfig,
    *,
    patch_start: int,
) -> FrameFusionState:
    """Create a temporary active-frame layout for one batch item."""
    active_inputs: list[torch.Tensor] = []
    active_members: list[list[int]] = []
    for group, reference in zip(groups, references):
        if len(group) == 1:
            active_inputs.append(tokens[group[0]])
            active_members.append(group)
        elif config.reference_participates:
            active_inputs.append(_fuse_group(tokens, group, reference, config, patch_start).to(tokens.dtype))
            active_members.append(group)
        else:
            active_inputs.append(tokens[reference])
            active_members.append([reference])
            other = [index for index in group if index != reference]
            if other:
                pseudo_reference = other[0]
                active_inputs.append(_fuse_group(tokens, other, pseudo_reference, config, patch_start).to(tokens.dtype))
                active_members.append(other)
    return FrameFusionState(groups, references, active_members, torch.stack(active_inputs), tokens)


def restore_frames(active_output: torch.Tensor, state: FrameFusionState) -> torch.Tensor:
    restored = state.original_tokens.clone()
    for active_idx, members in enumerate(state.active_members):
        delta = active_output[active_idx] - state.active_inputs[active_idx]
        restored[torch.tensor(members, device=restored.device)] += delta
    return restored


def _nearest_assignments(
    features: torch.Tensor,
    source_indices: torch.Tensor,
    destination_indices: torch.Tensor,
    *,
    chunk: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map sources to destination feature indices and return their cosine scores."""
    if destination_indices.numel() == 0:
        raise ValueError("Every token merge partition must retain at least one destination token")
    destination_features = F.normalize(features[destination_indices].float(), dim=-1)
    assignments = torch.empty(source_indices.numel(), dtype=torch.long, device=features.device)
    scores = torch.empty(source_indices.numel(), dtype=torch.float32, device=features.device)
    for start in range(0, source_indices.numel(), chunk):
        end = min(start + chunk, source_indices.numel())
        source_features = F.normalize(features[source_indices[start:end]].float(), dim=-1)
        similarity = source_features @ destination_features.T
        scores[start:end], local = similarity.max(dim=1)
        assignments[start:end] = destination_indices[local]
    return assignments, scores


def _fast_bipartite_assignments(
    patch_features: torch.Tensor,
    selected_indices: torch.Tensor,
    patch_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FastVGGT-style representatives stay within their active frame."""
    assignments = torch.empty(patch_features.shape[0], dtype=torch.long, device=patch_features.device)
    scores = torch.empty(patch_features.shape[0], dtype=torch.float32, device=patch_features.device)
    frames = patch_features.shape[0] // patch_count
    for frame_idx in range(frames):
        start, end = frame_idx * patch_count, (frame_idx + 1) * patch_count
        source = torch.arange(start, end, device=patch_features.device)
        destinations = selected_indices[(selected_indices >= start) & (selected_indices < end)]
        assigned, similarity = _nearest_assignments(patch_features, source, destinations)
        assignments[start:end] = assigned
        scores[start:end] = similarity
    return assignments, scores


def _category_assignments(
    patch_features: torch.Tensor,
    selected_indices: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Only tokens from the same clustering category may share a destination."""
    assignments = torch.empty(patch_features.shape[0], dtype=torch.long, device=patch_features.device)
    scores = torch.empty(patch_features.shape[0], dtype=torch.float32, device=patch_features.device)
    for category in labels.unique(sorted=True):
        source = torch.nonzero(labels == category, as_tuple=False).flatten()
        destinations = selected_indices[labels[selected_indices] == category]
        assigned, similarity = _nearest_assignments(patch_features, source, destinations)
        assignments[source] = assigned
        scores[source] = similarity
    return assignments, scores


def _fast_bipartite_selected(patch: torch.Tensor, grid_size: tuple[int, int], keep_ratio: float) -> torch.Tensor:
    frames, patches, _ = patch.shape
    height, width = grid_size
    if patches != height * width:
        raise ValueError("Fast bipartite token merging received a mismatched patch grid")
    keep_per_frame = max(1, math.ceil(patches * keep_ratio))
    candidates = []
    norm = patch.float().norm(dim=-1)
    for frame in range(frames):
        grid = torch.zeros((height, width), dtype=torch.bool, device=patch.device)
        grid[::2, ::2] = True
        preferred = torch.nonzero(grid.flatten(), as_tuple=False).flatten()
        if preferred.numel() < keep_per_frame:
            preferred = torch.topk(norm[frame], k=keep_per_frame).indices
        else:
            preferred = preferred[torch.topk(norm[frame, preferred], k=keep_per_frame).indices]
        candidates.append(frame * patches + preferred)
    return torch.cat(candidates)


def _category_selected(patch: torch.Tensor, config: AdaptiveFusionConfig) -> tuple[torch.Tensor, torch.Tensor]:
    flat = patch.reshape(-1, patch.shape[-1])
    reduced = _pca(flat, min(config.representation_pca_dim, flat.shape[-1]))
    labels, _ = _kmeans(reduced, config.token_clusters, config.token_kmeans_iterations)
    total_budget = max(1, math.ceil(flat.shape[0] * config.token_keep_ratio))
    selected_parts = []
    class_sizes = torch.bincount(labels, minlength=int(labels.max().item()) + 1).float()
    if config.token_cluster_budget == "dispersion":
        dispersion = torch.zeros_like(class_sizes)
        for index in range(class_sizes.numel()):
            member = reduced[labels == index]
            dispersion[index] = member.var(dim=0).mean() if member.shape[0] > 1 else 0.0
        weights = class_sizes * dispersion.clamp_min(1e-6)
    else:
        weights = class_sizes
    budget = torch.floor(total_budget * weights / weights.sum()).long()
    budget[weights.argmax()] += total_budget - int(budget.sum())
    norms = flat.float().norm(dim=-1)
    for index in range(class_sizes.numel()):
        member = torch.nonzero(labels == index, as_tuple=False).flatten()
        if member.numel() == 0:
            continue
        keep = min(member.numel(), max(1, int(budget[index].item())))
        selected_parts.append(member[torch.topk(norms[member], k=keep).indices])
    return torch.cat(selected_parts).sort().values, labels


def merge_tokens(tokens: torch.Tensor, *, patch_start: int, grid_size: tuple[int, int], config: AdaptiveFusionConfig) -> TokenFusionState:
    """Merge source features into representatives and retain inverse positions for restoration."""
    frames, tokens_per_frame, channels = tokens.shape
    patches = tokens[:, patch_start:]
    patch_count = patches.shape[1]
    flat = tokens.reshape(-1, channels)
    special = (torch.arange(frames, device=tokens.device)[:, None] * tokens_per_frame + torch.arange(patch_start, device=tokens.device)).flatten()
    if config.token_merging == "fast_bipartite":
        selected_patch_indices = _fast_bipartite_selected(patches, grid_size, config.token_keep_ratio)
        patch_assignments, patch_similarity = _fast_bipartite_assignments(
            patches.reshape(-1, channels), selected_patch_indices, patch_count
        )
    elif config.token_merging == "category_topk_norm":
        selected_patch_indices, category_labels = _category_selected(patches, config)
        patch_assignments, patch_similarity = _category_assignments(
            patches.reshape(-1, channels), selected_patch_indices, category_labels
        )
    else:
        raise ValueError(f"Unknown adaptive token merging: {config.token_merging}")
    patch_original = (torch.arange(frames, device=tokens.device)[:, None] * tokens_per_frame + patch_start + torch.arange(patch_count, device=tokens.device)).reshape(-1)
    selected_patch = patch_original[selected_patch_indices]
    selected = torch.cat([special, selected_patch]).unique(sorted=True)
    inverse = torch.empty(frames * tokens_per_frame, dtype=torch.long, device=tokens.device)
    selected_to_packed = torch.full((frames * tokens_per_frame,), -1, dtype=torch.long, device=tokens.device)
    selected_to_packed[selected] = torch.arange(selected.numel(), device=tokens.device)
    inverse[special] = selected_to_packed[special]
    inverse[patch_original] = selected_to_packed[patch_original[patch_assignments]]

    # Every discarded patch explicitly writes into its assigned representative.
    # The cosine-derived weight is positive and selected tokens receive weight 1.
    assignment_weights = torch.ones(frames * tokens_per_frame, dtype=torch.float32, device=tokens.device)
    assignment_weights[patch_original] = ((patch_similarity + 1.0) * 0.5).clamp_min(1e-6)
    packed_sum = torch.zeros((selected.numel(), channels), dtype=torch.float32, device=tokens.device)
    packed_sum.index_add_(0, inverse, flat.float() * assignment_weights.unsqueeze(-1))
    packed_weight = torch.zeros(selected.numel(), dtype=torch.float32, device=tokens.device)
    packed_weight.index_add_(0, inverse, assignment_weights)
    packed = (packed_sum / packed_weight.clamp_min(1e-6).unsqueeze(-1)).to(tokens.dtype)
    return TokenFusionState(
        inverse=inverse,
        packed_input=packed,
        original_tokens=flat,
        selected_indices=selected,
        assignment_weights=assignment_weights,
        selected_count=int(selected.numel()),
        original_count=int(flat.shape[0]),
    )


def restore_tokens(packed_output: torch.Tensor, state: TokenFusionState) -> torch.Tensor:
    centers = state.packed_input[state.inverse]
    return state.original_tokens + (packed_output[state.inverse] - centers)


def _fused_reference_patches(
    tokens: torch.Tensor,
    members: list[int],
    reference: int,
    config: AdaptiveFusionConfig,
    patch_start: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse only eligible source patches into one reference patch grid.

    The returned boolean mask has one row per source frame.  False entries
    remain standalone patches and must stay in the global-attention sequence.
    """
    reference_patch = tokens[reference, patch_start:].float()
    sources = [index for index in members if index != reference]
    if not sources:
        return reference_patch, torch.empty((0, reference_patch.shape[0]), dtype=torch.bool, device=tokens.device)
    source_patch = tokens[torch.tensor(sources, device=tokens.device), patch_start:].float()
    if config.frame_fusion == "direct":
        if config.frame_fusion_weighting == "uniform":
            weights = torch.ones(len(sources), dtype=torch.float32, device=tokens.device)
        else:
            descriptor = F.normalize(tokens.float().mean(dim=1), dim=-1)
            weights = (descriptor[sources] @ descriptor[reference]).clamp_min(1e-4)
        fused = (reference_patch + (source_patch * weights[:, None, None]).sum(dim=0)) / (1.0 + weights.sum())
        return fused, torch.ones(source_patch.shape[:2], dtype=torch.bool, device=tokens.device)
    if config.frame_fusion == "token_wise":
        similarity = F.cosine_similarity(source_patch, reference_patch.unsqueeze(0), dim=-1)
        selected = similarity >= config.frame_token_similarity_threshold
        weights = selected.float()
        source_sum = (source_patch * weights.unsqueeze(-1)).sum(dim=0)
        source_count = weights.sum(dim=0).clamp_min(1.0).unsqueeze(-1)
        candidate = (reference_patch + source_sum) / (1.0 + source_count)
        fused = torch.where(selected.any(dim=0).unsqueeze(-1), candidate, reference_patch)
        return fused, selected
    raise ValueError(f"Unknown adaptive frame fusion: {config.frame_fusion}")


def _merge_reference_patches(
    patches: torch.Tensor,
    grid_size: tuple[int, int],
    config: AdaptiveFusionConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the configured token merger to a complete reference patch grid."""
    patch_count, channels = patches.shape
    dense = patches.unsqueeze(0)
    if config.token_merging == "fast_bipartite":
        selected = _fast_bipartite_selected(dense, grid_size, config.token_keep_ratio)
        assignments, similarity = _fast_bipartite_assignments(patches, selected, patch_count)
    elif config.token_merging == "category_topk_norm":
        selected, labels = _category_selected(dense, config)
        assignments, similarity = _category_assignments(patches, selected, labels)
    else:
        raise ValueError(f"Unknown adaptive token merging: {config.token_merging}")
    selected_to_packed = torch.full((patch_count,), -1, dtype=torch.long, device=patches.device)
    selected_to_packed[selected] = torch.arange(selected.numel(), device=patches.device)
    inverse = selected_to_packed[assignments]
    weights = ((similarity + 1.0) * 0.5).clamp_min(1e-6)
    packed_sum = torch.zeros((selected.numel(), channels), dtype=torch.float32, device=patches.device)
    packed_sum.index_add_(0, inverse, patches.float() * weights.unsqueeze(-1))
    packed_weight = torch.zeros(selected.numel(), dtype=torch.float32, device=patches.device)
    packed_weight.index_add_(0, inverse, weights)
    packed = (packed_sum / packed_weight.clamp_min(1e-6).unsqueeze(-1)).to(patches.dtype)
    return packed, inverse, selected


def selective_fuse_and_merge_tokens(
    tokens: torch.Tensor,
    groups: list[list[int]],
    references: list[int],
    *,
    patch_start: int,
    grid_size: tuple[int, int],
    config: AdaptiveFusionConfig,
) -> SelectiveFrameTokenState:
    """Pack selective frame fusion without dropping per-frame special tokens.

    Camera/register tokens always have an identity mapping.  In token-wise
    frame fusion, only source patches above the similarity threshold map to a
    fused reference patch; every remaining source patch stays as its own token
    through global attention.  Token merging then compresses reference grids
    only, leaving those residual source patches untouched.
    """
    frames, tokens_per_frame, channels = tokens.shape
    patch_count = tokens_per_frame - patch_start
    total = frames * tokens_per_frame
    raw_inverse = torch.empty(total, dtype=torch.long, device=tokens.device)

    # Candidate entries are sorted back into original token order before the
    # global block, preserving the baseline's frame-major token layout.
    special_positions = (
        torch.arange(frames, device=tokens.device)[:, None] * tokens_per_frame
        + torch.arange(patch_start, device=tokens.device)
    ).reshape(-1)
    candidate_features = [tokens[:, :patch_start].reshape(-1, channels)]
    candidate_positions = [special_positions]
    raw_inverse[special_positions] = torch.arange(special_positions.numel(), device=tokens.device)
    candidate_count = int(special_positions.numel())
    frame_input_count = candidate_count
    reference_patch_count = 0
    residual_patch_count = 0

    for group, reference in zip(groups, references):
        entities: list[tuple[list[int], int]]
        if config.reference_participates:
            entities = [(group, reference)]
        else:
            others = [index for index in group if index != reference]
            entities = [([reference], reference)]
            if others:
                entities.append((others, others[0]))
        for members, anchor in entities:
            fused_patch, fused_sources = _fused_reference_patches(
                tokens,
                members,
                anchor,
                config,
                patch_start,
            )
            packed_reference, reference_inverse, selected_positions = _merge_reference_patches(
                fused_patch,
                grid_size,
                config,
            )
            reference_candidates = candidate_count + torch.arange(
                packed_reference.shape[0], device=tokens.device
            )
            candidate_features.append(packed_reference)
            candidate_positions.append(anchor * tokens_per_frame + patch_start + selected_positions)
            reference_mapping = reference_candidates[reference_inverse]
            anchor_positions = anchor * tokens_per_frame + patch_start + torch.arange(patch_count, device=tokens.device)
            raw_inverse[anchor_positions] = reference_mapping
            candidate_count += int(packed_reference.shape[0])
            frame_input_count += patch_count
            reference_patch_count += patch_count

            sources = [index for index in members if index != anchor]
            for source_idx, source in enumerate(sources):
                source_positions = source * tokens_per_frame + patch_start + torch.arange(
                    patch_count, device=tokens.device
                )
                mapped = fused_sources[source_idx]
                source_mapping = torch.empty(patch_count, dtype=torch.long, device=tokens.device)
                source_mapping[mapped] = reference_mapping[mapped]
                residual = torch.nonzero(~mapped, as_tuple=False).flatten()
                if residual.numel():
                    residual_candidates = candidate_count + torch.arange(residual.numel(), device=tokens.device)
                    candidate_features.append(tokens[source, patch_start:][residual])
                    candidate_positions.append(source_positions[residual])
                    source_mapping[residual] = residual_candidates
                    candidate_count += int(residual.numel())
                    frame_input_count += int(residual.numel())
                    residual_patch_count += int(residual.numel())
                raw_inverse[source_positions] = source_mapping

    positions = torch.cat(candidate_positions)
    features = torch.cat(candidate_features, dim=0)
    order = positions.argsort()
    packed_input = features[order]
    candidate_to_packed = torch.empty(candidate_count, dtype=torch.long, device=tokens.device)
    candidate_to_packed[order] = torch.arange(candidate_count, device=tokens.device)
    inverse = candidate_to_packed[raw_inverse].view(frames, tokens_per_frame)
    return SelectiveFrameTokenState(
        inverse=inverse,
        packed_input=packed_input,
        original_tokens=tokens,
        frame_input_count=frame_input_count,
        reference_patch_count=reference_patch_count,
        residual_patch_count=residual_patch_count,
        selected_count=int(packed_input.shape[0]),
        original_count=total,
    )


def restore_selective_frame_token_tokens(
    packed_output: torch.Tensor,
    state: SelectiveFrameTokenState,
) -> torch.Tensor:
    """Restore original token slots from a selective packed global output."""
    flat_original = state.original_tokens.reshape(-1, state.original_tokens.shape[-1])
    centers = state.packed_input[state.inverse.reshape(-1)]
    restored = flat_original + (packed_output[state.inverse.reshape(-1)] - centers)
    return restored.view_as(state.original_tokens)
