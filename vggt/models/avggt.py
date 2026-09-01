"""Training-free AVGGT global-attention inference primitives.

This module implements the fixed VGGT configuration from AVGGT:

* global blocks 0--8 are evaluated frame-wise (G2F); and
* later global blocks retain all queries and special-token K/V, keep the
  first/reference frame intact, and uniformly subsample other-frame patch K/V.

For dropped patch columns, the attention distribution contains the original
diagonal self term plus a mean-filled virtual column.  These terms and the
selected K/V columns share one softmax normalization.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


_FACTOR_TO_GRID_STRIDE = {2: (1, 2), 4: (2, 2), 6: (2, 3), 9: (3, 3)}


@dataclass(frozen=True)
class AVGGTSamplingPlan:
    selected_indices: torch.Tensor
    dropped_indices: torch.Tensor
    total_tokens: int
    selected_tokens: int


def build_avggt_sampling_plan(
    *,
    num_frames: int,
    tokens_per_frame: int,
    patch_start: int,
    grid_size: tuple[int, int],
    subsample_factor: int,
    device: torch.device,
) -> AVGGTSamplingPlan:
    """Build AVGGT's fixed reference-preserving K/V sampling set."""
    if subsample_factor not in _FACTOR_TO_GRID_STRIDE:
        raise ValueError(
            "AVGGT supports the paper's subsampling factors "
            f"{sorted(_FACTOR_TO_GRID_STRIDE)}, got {subsample_factor}."
        )
    height, width = grid_size
    patches = tokens_per_frame - patch_start
    if patches != height * width:
        raise ValueError("AVGGT patch-token count does not match the patch grid")
    if num_frames < 1:
        raise ValueError("AVGGT requires at least one frame")

    stride_h, stride_w = _FACTOR_TO_GRID_STRIDE[subsample_factor]
    rows = torch.arange(0, height, stride_h, device=device)
    cols = torch.arange(0, width, stride_w, device=device)
    sampled_patch = (rows[:, None] * width + cols[None, :]).reshape(-1)
    special = torch.arange(patch_start, device=device)
    full_reference = torch.arange(tokens_per_frame, device=device)
    compact = torch.cat((special, patch_start + sampled_patch))
    per_frame = [full_reference]
    per_frame.extend(compact + frame * tokens_per_frame for frame in range(1, num_frames))
    selected = torch.cat(per_frame).unique(sorted=True)
    total = num_frames * tokens_per_frame
    selected_mask = torch.zeros(total, dtype=torch.bool, device=device)
    selected_mask[selected] = True
    dropped = torch.nonzero(~selected_mask, as_tuple=False).flatten()
    return AVGGTSamplingPlan(
        selected_indices=selected,
        dropped_indices=dropped,
        total_tokens=total,
        selected_tokens=int(selected.numel()),
    )


def _flash_attention_with_lse(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float):
    """Return FlashAttention output and log-normalizer when ATen exposes it."""
    op = getattr(torch.ops.aten, "_scaled_dot_product_flash_attention", None)
    if op is None or not q.is_cuda:
        return None
    try:
        result = op(q, k, v, 0.0, False, False, scale=scale)
        return result[0], result[1]
    except (RuntimeError, TypeError, AttributeError):
        return None


def _chunked_attention_with_lse(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float, query_chunk: int = 32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Portable fallback with bounded score memory for unsupported Flash kernels."""
    outputs, normalizers = [], []
    for start in range(0, q.shape[-2], query_chunk):
        part = q[:, :, start : start + query_chunk].float()
        scores = torch.matmul(part, k.float().transpose(-2, -1)) * scale
        lse = torch.logsumexp(scores, dim=-1)
        output = torch.matmul(torch.softmax(scores, dim=-1).to(v.dtype), v)
        outputs.append(output)
        normalizers.append(lse)
    return torch.cat(outputs, dim=-2), torch.cat(normalizers, dim=-1)


def avggt_attention(attention, x: torch.Tensor, pos: torch.Tensor | None, plan: AVGGTSamplingPlan) -> torch.Tensor:
    """AVGGT SGA attention with selected K/V, diagonal preservation and mean fill."""
    batch, queries, channels = x.shape
    qkv = attention.qkv(x).reshape(batch, queries, 3, attention.num_heads, attention.head_dim)
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    q, k = attention.q_norm(q), attention.k_norm(k)
    if attention.rope is not None:
        q = attention.rope(q, pos)
        k = attention.rope(k, pos)

    selected_k = k.index_select(2, plan.selected_indices)
    selected_v = v.index_select(2, plan.selected_indices)
    flash = _flash_attention_with_lse(q, selected_k, selected_v, attention.scale)
    if flash is None:
        selected_output, selected_lse = _chunked_attention_with_lse(
            q, selected_k, selected_v, attention.scale
        )
    else:
        selected_output, selected_lse = flash

    # Every dropped patch retains its original diagonal K/V term.  The remaining
    # dropped columns are represented by a multiplicity-aware mean K/V term.
    dropped_mask = torch.zeros(queries, dtype=torch.bool, device=x.device)
    dropped_mask[plan.dropped_indices] = True
    diagonal_logit = (q.float() * k.float()).sum(dim=-1) * attention.scale
    neg_inf = torch.full_like(diagonal_logit, -torch.inf)
    diagonal_logit = torch.where(dropped_mask.view(1, 1, -1), diagonal_logit, neg_inf)

    dropped_count = int(plan.dropped_indices.numel())
    if dropped_count:
        dropped_k = k.index_select(2, plan.dropped_indices)
        dropped_v = v.index_select(2, plan.dropped_indices)
        sum_k = dropped_k.float().sum(dim=2, keepdim=True)
        sum_v = dropped_v.float().sum(dim=2, keepdim=True)
        diagonal = dropped_mask.view(1, 1, -1, 1)
        remaining_count = torch.full((queries,), dropped_count, device=x.device, dtype=torch.float32)
        remaining_count[dropped_mask] -= 1.0
        safe_count = remaining_count.clamp_min(1.0).view(1, 1, -1)
        mean_dot = (q.float() * (sum_k - diagonal * k.float())).sum(dim=-1)
        mean_logit = mean_dot * attention.scale / safe_count
        valid_mean = remaining_count > 0
        mean_logit = torch.where(
            valid_mean.view(1, 1, -1),
            mean_logit + remaining_count.clamp_min(1.0).log().view(1, 1, -1),
            neg_inf,
        )
        mean_v = (sum_v - diagonal * v.float()) / safe_count.unsqueeze(-1)
    else:
        mean_logit = neg_inf
        mean_v = torch.zeros_like(v, dtype=torch.float32)

    # A stable three-component softmax: selected K/V, diagonal, and mean fill.
    total_lse = torch.logaddexp(torch.logaddexp(selected_lse.float(), diagonal_logit), mean_logit)
    selected_weight = torch.exp(selected_lse.float() - total_lse).unsqueeze(-1)
    diagonal_weight = torch.exp(diagonal_logit - total_lse).unsqueeze(-1)
    mean_weight = torch.exp(mean_logit - total_lse).unsqueeze(-1)
    output = (
        selected_weight * selected_output.float()
        + diagonal_weight * v.float()
        + mean_weight * mean_v
    ).to(dtype=x.dtype)
    output = output.transpose(1, 2).reshape(batch, queries, channels)
    return attention.proj_drop(attention.proj(output))
