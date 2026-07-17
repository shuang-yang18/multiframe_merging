# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math
from typing import List, Tuple

from torch import Tensor, nn
import torch
import torch.nn.functional as F

from .utils import cat_keep_shapes, uncat_with_shapes


# RoPE-related functions:
def rope_rotate_half(x: Tensor) -> Tensor:
    # x:   [ x0  x1  x2  x3  x4  x5]
    # out: [-x3 -x4 -x5  x0  x1  x2]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope_apply(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    # x:   [..., D], eg [x0,     x1,   x2,   x3,   x4,   x5]
    # sin: [..., D], eg [sin0, sin1, sin2, sin0, sin1, sin2]
    # cos: [..., D], eg [cos0, cos1, cos2, cos0, cos1, cos2]
    return (x * cos) + (rope_rotate_half(x) * sin)


class LinearKMaskedBias(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        o = self.out_features
        assert o % 3 == 0
        if self.bias is not None:
            self.register_buffer("bias_mask", torch.full_like(self.bias, fill_value=math.nan))

    def forward(self, input: Tensor) -> Tensor:
        masked_bias = self.bias * self.bias_mask.to(self.bias.dtype) if self.bias is not None else None
        return F.linear(input, self.weight, masked_bias)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mask_k_bias: bool = False,
        use_qk_norm: bool = False,
        device=None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        # VGGT-Omega change: the aggregator checkpoint was trained with Q/K
        # normalization, while upstream DINOv3 attention does not expose it.
        self.use_qk_norm = use_qk_norm
        if self.use_qk_norm:
            self.q_norm = nn.LayerNorm(head_dim, eps=1e-5)
            self.k_norm = nn.LayerNorm(head_dim, eps=1e-5)
        else:
            self.q_norm = None
            self.k_norm = None

        linear_class = LinearKMaskedBias if mask_k_bias else nn.Linear
        self.qkv = linear_class(dim, dim * 3, bias=qkv_bias, device=device)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias, device=device)
        self.proj_drop = nn.Dropout(proj_drop)
        self.fastvggt_merge_ratio = None
        self.fastvggt_protection = None
        self.fastvggt_patch_grid_size = None
        self.fastvggt_num_frames = None
        self.fastvggt_special_token_count = None
        self.fastvggt_protected_token_indices = None
        self.fastvggt_dynamic_frame_mask = None
        self.last_fastvggt_stats = None
        self.capture_cls_attention = False
        self.last_cls_attention = None

    def apply_rope(self, q: Tensor, k: Tensor, rope: Tensor | Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        # All operations will use the dtype of rope, the output is cast back to the dtype of q and k
        q_dtype = q.dtype
        k_dtype = k.dtype
        sin, cos = rope
        rope_dtype = sin.dtype
        q = q.to(dtype=rope_dtype)
        k = k.to(dtype=rope_dtype)
        N = q.shape[-2]
        prefix = N - sin.shape[-2]
        assert prefix >= 0
        q_prefix = q[:, :, :prefix, :]
        q = rope_apply(q[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        q = torch.cat((q_prefix, q), dim=-2)  # [B, head, N, D//head]
        k_prefix = k[:, :, :prefix, :]
        k = rope_apply(k[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        k = torch.cat((k_prefix, k), dim=-2)  # [B, head, N, D//head]
        q = q.to(dtype=q_dtype)
        k = k.to(dtype=k_dtype)
        return q, k

    def forward(self, x: Tensor, attn_bias=None, rope: Tensor = None) -> Tensor:
        qkv = self.qkv(x)
        attn_v = self.compute_attention(qkv=qkv, attn_bias=attn_bias, rope=rope)
        x = self.proj(attn_v)
        x = self.proj_drop(x)
        return x

    def forward_list(self, x_list, attn_bias=None, rope_list=None) -> List[Tensor]:
        assert len(x_list) == len(rope_list)  # should be enforced by the Block
        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
        qkv_flat = self.qkv(x_flat)
        qkv_list = uncat_with_shapes(qkv_flat, shapes, num_tokens)
        att_out = []
        for _, (qkv, _, rope) in enumerate(zip(qkv_list, shapes, rope_list)):
            att_out.append(self.compute_attention(qkv, attn_bias=attn_bias, rope=rope))
        x_flat, shapes, num_tokens = cat_keep_shapes(att_out)
        x_flat = self.proj(x_flat)
        return uncat_with_shapes(x_flat, shapes, num_tokens)

    def compute_attention(self, qkv: Tensor, attn_bias=None, rope=None) -> Tensor:
        assert attn_bias is None
        B, N, _ = qkv.shape
        C = self.qkv.in_features

        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if rope is not None:
            q, k = self.apply_rope(q, k, rope)
        if self.capture_cls_attention:
            cls_attention = (q[:, :, :1, :] @ k.transpose(-2, -1)) * self.scale
            cls_attention = cls_attention.softmax(dim=-1, dtype=torch.float32).mean(dim=1).squeeze(1)
            self.last_cls_attention = cls_attention.to(q.dtype)
        unmerge = None
        self.last_fastvggt_stats = None
        if self.fastvggt_merge_ratio is not None:
            original_token_count = N
            merge, unmerge = _fastvggt_token_merge_bipartite2d(
                q.transpose(1, 2).reshape(B, N, C),
                patch_grid_size=self.fastvggt_patch_grid_size,
                num_frames=self.fastvggt_num_frames,
                special_token_count=self.fastvggt_special_token_count,
                merge_ratio=self.fastvggt_merge_ratio,
                protection=self.fastvggt_protection,
                protected_token_indices=self.fastvggt_protected_token_indices,
                dynamic_frame_mask=self.fastvggt_dynamic_frame_mask,
            )
            q_merge = q.transpose(1, 2).reshape(B, N, C)
            k_merge = k.transpose(1, 2).reshape(B, N, C)
            v_merge = v.transpose(1, 2).reshape(B, N, C)
            q_merge, k_merge, v_merge = merge(q_merge, mode="mean", extra_tensors=k_merge, extra_tensors_2=v_merge)
            N = q_merge.shape[1]
            self.last_fastvggt_stats = {
                "original_tokens": int(original_token_count),
                "active_tokens": int(N),
                "full_attention_token_ratio": float(N / original_token_count if original_token_count else 0.0),
                "merged_away_token_ratio": float(1.0 - N / original_token_count if original_token_count else 0.0),
            }
            q = q_merge.reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
            k = k_merge.reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
            v = v_merge.reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2)
        x = x.reshape([B, N, C])
        if unmerge is not None:
            x = unmerge(x)
        return x


def _fastvggt_token_merge_bipartite2d(
    metric: Tensor,
    patch_grid_size: tuple[int, int],
    num_frames: int,
    special_token_count: int,
    merge_ratio: float,
    protection: str | None,
    protected_token_indices: Tensor | None = None,
    dynamic_frame_mask: Tensor | None = None,
):
    B, N, _ = metric.shape
    grid_h, grid_w = patch_grid_size
    tokens_per_frame = grid_h * grid_w + special_token_count
    if tokens_per_frame * num_frames != N:
        raise ValueError(
            f"Token count {N} does not match frames={num_frames}, "
            f"tokens_per_frame={tokens_per_frame}"
        )

    r = int(N * merge_ratio)
    if r <= 0:
        return _merge_do_nothing, _merge_do_nothing

    device = metric.device
    gather = torch.gather
    sx = sy = 2
    hsy = grid_h // sy
    wsx = grid_w // sx
    idx_buffer = torch.zeros(N, device=device, dtype=torch.long)

    idx_buffer[:tokens_per_frame] = -1
    if num_frames > 1:
        frame_offsets = torch.arange(1, num_frames, device=device) * tokens_per_frame
        special = frame_offsets[:, None] + torch.arange(special_token_count, device=device)
        idx_buffer[special.flatten()] = -1

        generator = torch.Generator(device=device)
        generator.manual_seed(33)
        rand = torch.randint(sy * sx, size=(num_frames - 1, hsy, wsx), device=device, generator=generator)
        block = torch.zeros(num_frames - 1, hsy, wsx, sy * sx, device=device, dtype=torch.long)
        block.scatter_(3, rand.unsqueeze(-1), -torch.ones_like(rand).unsqueeze(-1))
        block = block.view(num_frames - 1, hsy, wsx, sy, sx).transpose(2, 3).reshape(
            num_frames - 1, hsy * sy, wsx * sx
        )
        effective_h = hsy * sy
        effective_w = wsx * sx
        effective_grid_size = effective_h * effective_w
        for frame_idx in range(1, num_frames):
            start = frame_idx * tokens_per_frame + special_token_count
            idx_buffer[start : start + effective_grid_size] = block[frame_idx - 1].flatten()

    rand_idx = idx_buffer.reshape(1, -1, 1).argsort(dim=1)
    num_dst = int((idx_buffer == -1).sum())
    a_idx = rand_idx[:, num_dst:, :]
    b_idx = rand_idx[:, :num_dst, :]

    num_protected = int(N * 0.1)
    if protection == "decoupled_window":
        protected_indices = _decoupled_window_protected_indices(
            metric,
            patch_grid_size,
            num_frames,
            special_token_count,
            merge_ratio,
            dynamic_frame_mask,
        )
    elif protection == "adts" and num_protected > 0:
        protected_indices = torch.topk(metric.float().norm(dim=-1).mean(dim=0), k=num_protected).indices.sort().values
    else:
        step = max(1, N // max(num_protected, 1))
        protected_indices = torch.arange(0, N, step, device=device)[:num_protected]
    if protected_token_indices is not None and protected_token_indices.numel() > 0:
        protected_indices = torch.cat([protected_indices, protected_token_indices.to(device=device, dtype=torch.long)])
        protected_indices = torch.unique(protected_indices.clamp(0, N - 1), sorted=True)
    protected_idx = protected_indices.unsqueeze(0).unsqueeze(-1)
    num_protected_actual = protected_idx.shape[1]

    def split(x: Tensor):
        C = x.shape[-1]
        src = gather(x, dim=1, index=a_idx.expand(B, a_idx.shape[1], C))
        dst = gather(x, dim=1, index=b_idx.expand(B, b_idx.shape[1], C))
        protected = gather(x, dim=1, index=protected_idx.expand(B, num_protected_actual, C))
        return src, dst, protected

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        src_metric, dst_metric, _ = split(metric)
        r = min(src_metric.shape[1], r)
        node_max, node_idx = _fast_similarity_chunks(src_metric, dst_metric.transpose(-1, -2), 5000)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        src_indices = a_idx[0, :, 0]
        protected_mask_src = torch.isin(src_indices, protected_indices)
        edge_flat = edge_idx[0, :, 0]
        valid_edges = edge_flat[~protected_mask_src[edge_flat]]
        r_actual = min(r, valid_edges.shape[0])
        unm_idx = valid_edges[r_actual:].unsqueeze(0).unsqueeze(-1)
        src_idx = valid_edges[:r_actual].unsqueeze(0).unsqueeze(-1)
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)

    def merge(x: Tensor, mode: str = "mean", extra_tensors=None, extra_tensors_2=None):
        src, dst, protected = split(x)
        n, _, c = src.shape
        unm_len = unm_idx.shape[1]
        src_len = src_idx.shape[1]
        unm = gather(src, dim=-2, index=unm_idx.expand(n, unm_len, c))
        src = gather(src, dim=-2, index=src_idx.expand(n, src_len, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, src_len, c), src, reduce=mode)
        main = torch.cat([unm, dst, protected], dim=1)

        merged_extra_1 = None
        merged_extra_2 = None
        if extra_tensors is not None:
            src_e, dst_e, protected_e = split(extra_tensors)
            e_dim = extra_tensors.shape[-1]
            unm_e = gather(src_e, dim=-2, index=unm_idx.expand(n, unm_len, e_dim))
            src_e = gather(src_e, dim=-2, index=src_idx.expand(n, src_len, e_dim))
            dst_e = dst_e.scatter_reduce(-2, dst_idx.expand(n, src_len, e_dim), src_e, reduce=mode)
            merged_extra_1 = torch.cat([unm_e, dst_e, protected_e], dim=1)
        if extra_tensors_2 is not None:
            src_e, dst_e, protected_e = split(extra_tensors_2)
            e_dim = extra_tensors_2.shape[-1]
            unm_e = gather(src_e, dim=-2, index=unm_idx.expand(n, unm_len, e_dim))
            src_e = gather(src_e, dim=-2, index=src_idx.expand(n, src_len, e_dim))
            dst_e = dst_e.scatter_reduce(-2, dst_idx.expand(n, src_len, e_dim), src_e, reduce=mode)
            merged_extra_2 = torch.cat([unm_e, dst_e, protected_e], dim=1)

        if merged_extra_1 is not None and merged_extra_2 is not None:
            return main, merged_extra_1, merged_extra_2
        if merged_extra_1 is not None:
            return main, merged_extra_1
        return main

    def unmerge(x: Tensor) -> Tensor:
        unm_len = unm_idx.shape[1]
        dst_len = num_dst
        src_len = src_idx.shape[1]
        unm = x[..., :unm_len, :]
        dst = x[..., unm_len : unm_len + dst_len, :]
        protected = x[..., unm_len + dst_len : unm_len + dst_len + num_protected_actual, :]
        _, _, c = unm.shape
        src = gather(dst, dim=-2, index=dst_idx.expand(B, src_len, c))
        out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst)
        out.scatter_(
            dim=-2,
            index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx).expand(B, unm_len, c),
            src=unm,
        )
        out.scatter_(
            dim=-2,
            index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx).expand(B, src_len, c),
            src=src,
        )
        out.scatter_(dim=-2, index=protected_idx.expand(B, num_protected_actual, c), src=protected)
        return out

    return merge, unmerge


def _decoupled_window_protected_indices(
    metric: Tensor,
    patch_grid_size: tuple[int, int],
    num_frames: int,
    special_token_count: int,
    merge_ratio: float,
    dynamic_frame_mask: Tensor | None,
) -> Tensor:
    device = metric.device
    grid_h, grid_w = patch_grid_size
    num_patches = grid_h * grid_w
    tokens_per_frame = num_patches + special_token_count
    if dynamic_frame_mask is None:
        dynamic = torch.zeros(num_frames, dtype=torch.bool, device=device)
    else:
        dynamic = dynamic_frame_mask[0].to(device=device, dtype=torch.bool)
        if dynamic.numel() < num_frames:
            dynamic = F.pad(dynamic, (0, num_frames - dynamic.numel()), value=False)
        elif dynamic.numel() > num_frames:
            dynamic = dynamic[:num_frames]

    score = metric.float().norm(dim=-1).mean(dim=0)
    keep_per_static_frame = max(1, int(math.ceil(num_patches * max(0.0, 1.0 - merge_ratio))))
    window_rows = max(1, min(grid_h, int(round(math.sqrt(keep_per_static_frame * grid_h / max(grid_w, 1))))))
    window_cols = max(1, min(grid_w, int(math.ceil(keep_per_static_frame / window_rows))))
    row_bounds = torch.linspace(0, grid_h, window_rows + 1, device=device).round().long()
    col_bounds = torch.linspace(0, grid_w, window_cols + 1, device=device).round().long()

    protected: list[Tensor] = []
    for frame_idx in range(num_frames):
        frame_start = frame_idx * tokens_per_frame
        if bool(dynamic[frame_idx]):
            protected.append(torch.arange(frame_start, frame_start + tokens_per_frame, device=device))
            continue

        frame_candidates = []
        frame_scores = []
        patch_start = frame_start + special_token_count
        for row_idx in range(window_rows):
            row_start = int(row_bounds[row_idx].item())
            row_end = int(row_bounds[row_idx + 1].item())
            if row_end <= row_start:
                continue
            for col_idx in range(window_cols):
                col_start = int(col_bounds[col_idx].item())
                col_end = int(col_bounds[col_idx + 1].item())
                if col_end <= col_start:
                    continue
                rows = torch.arange(row_start, row_end, device=device)
                cols = torch.arange(col_start, col_end, device=device)
                local = (rows[:, None] * grid_w + cols[None, :]).flatten()
                token_indices = patch_start + local
                best = torch.argmax(score[token_indices])
                frame_candidates.append(token_indices[best])
                frame_scores.append(score[token_indices[best]])
        if frame_candidates:
            candidates = torch.stack(frame_candidates)
            scores = torch.stack(frame_scores)
            keep = min(keep_per_static_frame, candidates.numel())
            chosen = candidates[torch.topk(scores, k=keep).indices]
            protected.append(chosen)

    if not protected:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.unique(torch.cat(protected).long().clamp(0, metric.shape[1] - 1), sorted=True)


def _merge_do_nothing(x: Tensor, *args, **kwargs):
    if kwargs.get("extra_tensors") is not None and kwargs.get("extra_tensors_2") is not None:
        return x, kwargs["extra_tensors"], kwargs["extra_tensors_2"]
    if kwargs.get("extra_tensors") is not None:
        return x, kwargs["extra_tensors"]
    return x


@torch.jit.script
def _fast_similarity_chunks(a: Tensor, b_transposed: Tensor, chunk_size: int):
    B, num_src, C = a.shape
    original_dtype = a.dtype
    a_bf16 = a.to(torch.bfloat16)
    b_bf16 = b_transposed.to(torch.bfloat16)
    node_max = torch.empty(B, num_src, device=a.device, dtype=original_dtype)
    node_idx = torch.empty(B, num_src, device=a.device, dtype=torch.long)
    for i in range(0, num_src, chunk_size):
        end_i = min(i + chunk_size, num_src)
        scores = torch.bmm(a_bf16[:, i:end_i, :], b_bf16)
        chunk_max, chunk_idx = torch.max(scores, dim=2)
        node_max[:, i:end_i] = chunk_max.to(original_dtype)
        node_idx[:, i:end_i] = chunk_idx
    return node_max, node_idx


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def init_weights(
        self, init_attn_std: float | None = None, init_proj_std: float | None = None, factor: float = 1.0
    ) -> None:
        init_attn_std = init_attn_std or (self.dim**-0.5)
        init_proj_std = init_proj_std or init_attn_std * factor
        nn.init.normal_(self.qkv.weight, std=init_attn_std)
        nn.init.normal_(self.proj.weight, std=init_proj_std)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor, is_causal: bool = True) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.attn_drop if self.training else 0, is_causal=is_causal
        )
        x = x.transpose(1, 2).contiguous().view(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x
