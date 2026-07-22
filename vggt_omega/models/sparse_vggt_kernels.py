"""Minimal SpargeAttn kernel wrapper used by the Sparse-VGGT adapter."""

from __future__ import annotations

import math
from collections import namedtuple

import torch


SortResult = namedtuple("SortResult", ["values", "indices"])


def _int32_idx(sort_result):
    return SortResult(sort_result.values, sort_result.indices.to(torch.int32))


def _mem_eff_sort(t: torch.Tensor, chunks: int = 4, dim: int = 1):
    sorted_chunks = [_int32_idx(torch.sort(tt, dim=-1, descending=True)) for tt in torch.chunk(t, chunks, dim=dim)]
    return SortResult(
        torch.cat([chunk.values for chunk in sorted_chunks], dim=dim),
        torch.cat([chunk.indices for chunk in sorted_chunks], dim=dim),
    )


def _check_sparse_mode(topk: int | None, sparse_ratio: float | None, cdf_threshold: float | None) -> None:
    use_topk = topk is not None
    use_ratio = sparse_ratio is not None
    use_cdf = cdf_threshold is not None
    valid = (
        use_topk and not use_ratio and not use_cdf
        or use_ratio and not use_topk and not use_cdf
        or use_cdf and not use_topk and not use_ratio
        or use_ratio and use_cdf and not use_topk
    )
    if not valid:
        raise ValueError(f"Invalid Sparse-VGGT mode: {topk=}, {sparse_ratio=}, {cdf_threshold=}")


def get_block_mask(
    pooled_score: torch.Tensor,
    sink_blocks: int,
    topk: int | None = None,
    sparse_ratio: float | None = None,
    cdf_threshold: float | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    from spas_sage_attn.utils import fill_block_map_triton, hyperparameter_check

    _check_sparse_mode(topk, sparse_ratio, cdf_threshold)
    batch_size, num_heads, query_blocks, key_blocks = pooled_score.shape
    if not 0 <= sink_blocks <= key_blocks:
        raise ValueError(f"Invalid sink_blocks={sink_blocks} for key_blocks={key_blocks}")

    if sparse_ratio is not None:
        if not 0 <= sparse_ratio <= 1:
            raise ValueError(f"sparse_ratio must be in [0, 1], got {sparse_ratio}")
        topk = int(key_blocks * (1 - sparse_ratio))
    if topk is not None and not 0 <= topk <= key_blocks:
        raise ValueError(f"topk must be in [0, {key_blocks}], got {topk}")
    if cdf_threshold is not None and not 0 <= cdf_threshold <= 1:
        raise ValueError(f"cdf_threshold must be in [0, 1], got {cdf_threshold}")

    sorted_score = torch.sort(pooled_score, dim=-1, descending=True) if pooled_score.numel() < 2e8 else _mem_eff_sort(pooled_score)
    num_to_select = None
    if cdf_threshold is not None:
        cdf = torch.cumsum(sorted_score.values, dim=-1)
        cdf_thresh = hyperparameter_check(cdf_threshold, num_heads, pooled_score.device)
        cdf_thresh = cdf_thresh.view(1, num_heads, 1, 1) + eps
        cdf_thresh = cdf_thresh.expand(batch_size, -1, query_blocks, 1).contiguous()
        num_to_select = torch.searchsorted(cdf, cdf_thresh, right=True).squeeze(-1)
    if topk is not None:
        topk_tensor = torch.full((batch_size, num_heads, query_blocks), topk, device=pooled_score.device)
        num_to_select = topk_tensor if num_to_select is None else torch.clamp(num_to_select, min=topk)

    final_map = torch.zeros_like(pooled_score, dtype=torch.bool)
    final_map = fill_block_map_triton(final_map, num_to_select, sorted_score.indices)
    if sink_blocks > 0:
        ones_shape = list(final_map.shape)
        ones_shape[-1] = sink_blocks
        final_map = torch.cat([final_map, torch.ones(ones_shape, device=final_map.device, dtype=torch.bool)], dim=-1)
    return final_map


def block_sparse_attn_cuda(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    pooled_score: torch.Tensor,
    topk: int | None = None,
    sparse_ratio: float | None = None,
    cdf_threshold: float | None = None,
    return_sparsity: bool = False,
    dtype: torch.dtype = torch.float16,
):
    import spas_sage_attn._qattn as qattn
    from spas_sage_attn.quant_per_block import per_block_int8
    from spas_sage_attn.utils import block_map_lut_triton, hyperparameter_check

    out_dtype = query.dtype
    key_block_size = 64
    total_key_blocks = math.ceil(key.shape[-2] / key_block_size)
    original_key_blocks = pooled_score.shape[-1]
    final_map = get_block_mask(
        pooled_score,
        sink_blocks=total_key_blocks - original_key_blocks,
        topk=topk,
        sparse_ratio=sparse_ratio,
        cdf_threshold=cdf_threshold,
    )
    lut, valid_block_num = block_map_lut_triton(final_map)

    query = query.contiguous().to(dtype)
    key = key.contiguous().to(dtype)
    value = value.contiguous().to(dtype)

    key_mean = key.mean(dim=-2, keepdim=True)
    q_int8, q_scale, k_int8, k_scale = per_block_int8(query, key - key_mean)
    q_scale = q_scale.squeeze(-1)
    k_scale = k_scale.squeeze(-1)
    pv_threshold = hyperparameter_check(1e10, query.size(-3), query.device)
    output = torch.empty_like(query)
    qattn.qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold(
        q_int8,
        k_int8,
        value,
        output,
        lut,
        valid_block_num,
        pv_threshold,
        q_scale,
        k_scale,
        1,
        0,
        1,
        1.0 / math.sqrt(query.shape[-1]),
        0,
    )
    output = output.to(out_dtype)
    if return_sparsity:
        return output, 1 - final_map.float().mean()
    return output

