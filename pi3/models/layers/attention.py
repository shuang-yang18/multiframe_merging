# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import math
import os
import warnings

from torch import Tensor
from torch import nn
import torch

from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend

XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
        # warnings.warn("xFormers is available (Attention)")
    else:
        # warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    # warnings.warn("xFormers is not available (Attention)")


class Attention(nn.Module):
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
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        # q, k, v = unbind(qkv, 2)
        q, k, v = [qkv[:,:,i] for i in range(3)]

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


    
class FlashAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1, 3)

        # q, k, v = unbind(qkv, 2)
        q, k, v = [qkv[:,:,i] for i in range(3)]

        if q.dtype == torch.bfloat16:
            with nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                x = scaled_dot_product_attention(q, k, v)
        else:
            with nn.attention.sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
                x = scaled_dot_product_attention(q, k, v)

        x = x.transpose(1, 2).reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


"""
Following is written by GPT-4o
"""
class CrossAttentionRope(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qk_norm: bool = False,
        norm_layer: nn.Module = nn.LayerNorm,
        rope=None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # Separate projection layers for query, key, and value
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.q_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(head_dim) if qk_norm else nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rope = rope

    def forward(self, query: Tensor, key: Tensor, value: Tensor, attn_bias=None, qpos=None, kpos=None) -> Tensor:
        """
        Args:
            query: Tensor of shape (B, N, C), input query
            key: Tensor of shape (B, M, C), input key
            value: Tensor of shape (B, M, C), input value
            attn_bias: Optional tensor for attention bias
        Returns:
            Tensor of shape (B, N, C), output of cross-attention
        """
        B, N, C = query.shape
        _, M, _ = key.shape

        # Project query, key, and value
        q = self.q_proj(query).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(key).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(value).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)

        if self.rope is not None:
            q = self.rope(q, qpos)
            k = self.rope(k, kpos)

        # Scale query
        q = q * self.scale

        # Compute attention scores
        attn = q @ k.transpose(-2, -1)  # (B, num_heads, N, M)
        if attn_bias is not None:
            attn = attn + attn_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Compute attention output
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, C)

        # Final projection
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffCrossAttentionRope(CrossAttentionRope):
    def forward(self, query: Tensor, key: Tensor, value: Tensor, attn_bias=None, qpos=None, kpos=None) -> Tensor:
        """
        Args:
            query: Tensor of shape (B, N, C), input query
            key: Tensor of shape (B, M, C), input key
            value: Tensor of shape (B, M, C), input value
            attn_bias: Optional tensor for attention bias
        Returns:
            Tensor of shape (B, N, C), output of cross-attention
        """
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(query, key, value, attn_bias)

        B, N, C = query.shape
        _, M, _ = key.shape

        # Project query, key, and value
        q = self.q_proj(query).reshape(B, N, self.num_heads, C // self.num_heads)
        k = self.k_proj(key).reshape(B, M, self.num_heads, C // self.num_heads)
        v = self.v_proj(value).reshape(B, M, self.num_heads, C // self.num_heads)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)

        if self.rope is not None:
            q = self.rope(q, qpos)
            k = self.rope(k, kpos)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        # Compute memory-efficient attention
        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape(B, N, C)

        # Final projection
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class AttentionRope(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qk_norm: bool = False,
        norm_layer: nn.Module = nn.LayerNorm,
        rope=None
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.head_dim = head_dim

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.q_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(head_dim) if qk_norm else nn.Identity()

        self.rope = rope

    def forward(self, x: Tensor, attn_bias=None, xpos=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)

        if self.rope is not None:
            q = self.rope(q, xpos)
            k = self.rope(k, xpos)
        
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttentionRope(AttentionRope):
    def forward(self, x: Tensor, attn_bias=None, xpos=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        
        qkv = qkv.transpose(1, 3)
        # q, k, v = unbind(qkv, 2)
        q, k, v = [qkv[:,:,i] for i in range(3)]
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)

        if self.rope is not None:
            q = self.rope(q, xpos)
            k = self.rope(k, xpos)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        # score_matrix = (q.permute(0, 2, 1, 3) * self.scale @ k.permute(0, 2, 1, 3).transpose(-2, -1)).sum(dim=1).reshape(frame_num, 261, frame_num, 261).mean(dim=[1, 3]).sum(1)         # for frame attention matrix
        # global_valid_id = torch.where(score_matrix > 0)
        # score_matrix = (q.permute(0, 2, 1, 3) * self.scale @ k.permute(0, 2, 1, 3).transpose(-2, -1)).sum(dim=1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    
class FlashAttentionRope(AttentionRope):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fastvggt_merge_ratio = None
        self.fastvggt_patch_grid_size = None
        self.fastvggt_num_frames = None
        self.fastvggt_special_token_count = None
        self.last_fastvggt_stats = None
        self.um_plan = None
        self.um_norm1 = None

    def forward(self, x: Tensor, attn_bias=None, xpos=None) -> Tensor:
        B, N, C = x.shape
        # U-M operates on normalized tokens before Q/K/V projection, exactly
        # as the Omega implementation does.  Its planner compresses only the
        # global-attention residual and restores the full sequence here.
        if self.um_plan is not None:
            from pi3.models.um import um_attention
            if self.um_norm1 is None:
                raise RuntimeError("Pi3 U-M requires its owning BlockRope norm1")
            return um_attention(self, x, self.um_plan, norm1=self.um_norm1, xpos=xpos)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1, 3)

        # q, k, v = unbind(qkv, 2)
        q, k, v = [qkv[:,:,i] for i in range(3)]
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)

        if self.rope is not None:
            q = self.rope(q, xpos)
            k = self.rope(k, xpos)

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

        if q.dtype == torch.bfloat16:
            with nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                x = scaled_dot_product_attention(q, k, v)
        else:
            with nn.attention.sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
                x = scaled_dot_product_attention(q, k, v)

        x = x.transpose(1, 2).reshape([B, N, C])
        if unmerge is not None:
            x = unmerge(x)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def _fastvggt_token_merge_bipartite2d(
    metric: Tensor,
    patch_grid_size: tuple[int, int],
    num_frames: int,
    special_token_count: int,
    merge_ratio: float,
):
    B, N, _ = metric.shape
    grid_h, grid_w = patch_grid_size
    tokens_per_frame = grid_h * grid_w + special_token_count
    if tokens_per_frame * num_frames != N:
        raise ValueError(
            f"Token count {N} does not match frames={num_frames}, tokens_per_frame={tokens_per_frame}"
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
        effective_grid_size = hsy * sy * wsx * sx
        for frame_idx in range(1, num_frames):
            start = frame_idx * tokens_per_frame + special_token_count
            idx_buffer[start : start + effective_grid_size] = block[frame_idx - 1].flatten()

    rand_idx = idx_buffer.reshape(1, -1, 1).argsort(dim=1)
    num_dst = int((idx_buffer == -1).sum())
    a_idx = rand_idx[:, num_dst:, :]
    b_idx = rand_idx[:, :num_dst, :]

    num_protected = int(N * 0.1)
    step = max(1, N // max(num_protected, 1))
    protected_indices = torch.arange(0, N, step, device=device)[:num_protected]
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

def get_attn_score(blk_class, x, frame_num, token_length, xpos=None):
    x = blk_class.norm1(x)
    
    B, N, C = x.shape
    qkv = blk_class.attn.qkv(x).reshape(B, N, 3, blk_class.attn.num_heads, C // blk_class.attn.num_heads)
    
    qkv = qkv.transpose(1, 3)
    # q, k, v = unbind(qkv, 2)
    q, k, v = [qkv[:,:,i] for i in range(3)]
    q, k = blk_class.attn.q_norm(q).to(v.dtype), blk_class.attn.k_norm(k).to(v.dtype)

    if blk_class.attn.rope is not None:
        q = blk_class.attn.rope(q, xpos)
        k = blk_class.attn.rope(k, xpos)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)

    score = (q.permute(0, 2, 1, 3) * blk_class.attn.scale @ k.permute(0, 2, 1, 3).transpose(-2, -1)).sum(dim=1).reshape(B, frame_num, token_length, frame_num, token_length).mean(dim=[2, 4]).sum(-1)

    return score


from .prope import _prepare_apply_fns, _prepare_apply_fns_query
class PRopeFlashAttention(AttentionRope):
    def forward(self, x: Tensor, extrinsics, H, W, patch_h, patch_w, K=None, attn_mask=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1, 3)

        # q, k, v = unbind(qkv, 2)
        q, k, v = [qkv[:,:,i] for i in range(3)]
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)

        apply_fn_q, apply_fn_kv, apply_fn_o = _prepare_apply_fns(
            head_dim=self.head_dim,
            viewmats=extrinsics,
            Ks=K,
            patches_x=patch_w,
            patches_y=patch_h,
            image_width=W,
            image_height=H,
        )
        q = apply_fn_q(q)
        k = apply_fn_kv(k)
        v = apply_fn_kv(v)

        if q.dtype == torch.bfloat16 and attn_mask is None:
            with nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                x = scaled_dot_product_attention(q, k, v)
        else:
            with nn.attention.sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
                x = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        
        x = apply_fn_o(x)

        x = x.transpose(1, 2).reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FlashCrossAttentionRope(CrossAttentionRope):
    def forward(self, query: Tensor, key: Tensor, value: Tensor, attn_bias=None, qpos=None, kpos=None) -> Tensor:
        """
        Args:
            query: Tensor of shape (B, N, C)
            key: Tensor of shape (B, M, C)
            value: Tensor of shape (B, M, C),
        Returns:
            Tensor of shape (B, N, C),
        """
        B, N, C = query.shape
        _, M, _ = key.shape

        q = self.q_proj(query).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(key).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(value).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)
        if self.rope is not None:
            q = self.rope(q, qpos)
            k = self.rope(k, kpos)
        
        dropout_p = self.attn_drop.p if self.training else 0.0
        
        if q.dtype == torch.bfloat16:
            with nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                x = scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_bias, dropout_p=dropout_p
                )
        else:
            with nn.attention.sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
                x = scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_bias, dropout_p=dropout_p
                )

        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
