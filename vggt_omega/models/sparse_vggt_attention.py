"""Sparse-VGGT block-sparse attention adapter for VGGT-Omega.

This module intentionally uses its own `sparse_vggt_*` names so it stays
separate from the existing FastVGGT/token-merging paths in this repository.
"""

from __future__ import annotations

import math
from collections import namedtuple

import torch
import torch.nn.functional as F
from einops import rearrange


SortResult = namedtuple("SortResult", ["values", "indices"])


def check_sparse_vggt_mode(sparse_ratio: float | None, cdf_threshold: float | None) -> None:
    use_ratio = sparse_ratio is not None
    use_cdf = cdf_threshold is not None
    if not (use_ratio or use_cdf):
        raise ValueError("Sparse-VGGT requires sparse_ratio and/or cdf_threshold.")


def sparse_vggt_predict_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    ks_q: int = 128,
    ks_k: int = 64,
    pool_mode: str = "avg",
) -> torch.Tensor:
    if pool_mode not in {"max", "avg"}:
        raise ValueError(f"Unknown Sparse-VGGT pool_mode={pool_mode!r}")
    pooling_fn = F.max_pool1d if pool_mode == "max" else F.avg_pool1d

    batch_size, num_heads, _, head_dim = query.shape
    query = rearrange(query, "B H T C -> (B H) C T")
    key = rearrange(key, "B H T C -> (B H) C T")
    pooled_query = pooling_fn(query, kernel_size=ks_q, ceil_mode=True)
    pooled_key = pooling_fn(key, kernel_size=ks_k, ceil_mode=True)
    pooled_query = rearrange(pooled_query, "(B H) C T -> B H T C", B=batch_size, H=num_heads)
    pooled_key = rearrange(pooled_key, "(B H) C T -> B H T C", B=batch_size, H=num_heads)
    pooled_score = pooled_query @ pooled_key.transpose(-1, -2) / math.sqrt(head_dim)
    return F.softmax(pooled_score, dim=-1)


def sparse_vggt_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    sparse_ratio: float | None,
    cdf_threshold: float | None,
    pool_mode: str = "avg",
) -> tuple[torch.Tensor, float]:
    try:
        from sparse_vggt.utils.sparse_wrapper import block_sparse_attn_cuda
    except ImportError:
        from vggt_omega.models.sparse_vggt_kernels import block_sparse_attn_cuda

    pooled_score = sparse_vggt_predict_attention(query, key, pool_mode=pool_mode)
    output, sparsity = block_sparse_attn_cuda(
        query=query,
        key=key,
        value=value,
        pooled_score=pooled_score,
        sparse_ratio=sparse_ratio,
        cdf_threshold=cdf_threshold,
        return_sparsity=True,
    )
    return output, float(sparsity.detach().float().item())

