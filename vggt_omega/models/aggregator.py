# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from vggt_omega.models.layers import Mlp, RopePositionEmbedding, SelfAttentionBlock
from vggt_omega.models.layers.vision_transformer import DinoVisionTransformer


_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    """Alternating-attention encoder over video frames."""

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_register_tokens: int = 16,
        register_attention_block_indices: list[int] | None = None,
        cached_layer_indices: tuple[int, ...] = (4, 11, 17, 23),
        enable_token_merging: bool = False,
        token_merging_start: int = 0,
        token_merging_ratio: float = 0.9,
        token_merging_layer_ratios: str = "",
        token_merging_method: str = "spatial",
        token_merging_tstm_threshold: float = 0.8,
        token_merging_tstm_neighbor_size: int = 3,
        token_merging_flashvid_alpha: float = 0.7,
        token_merging_flashvid_expansion: float = 1.25,
        token_merging_flashvid_pool_stride: int = 2,
        token_merging_frame_restore_layer: int = 16,
        token_merging_frame_alpha: float = 0.9,
        token_merging_frame_segment_threshold: float = 0.8,
        token_merging_frame_merge_threshold: float = 0.8,
        token_merging_frame_max_window: int = 6,
        token_merging_frame_pool_stride: int = 2,
        token_merging_frame_multi_max_group_size: int = 2,
        token_merging_frame_multi_pair_threshold: float = 0.95,
        token_merging_frame_multi_span_threshold: float = 0.93,
        token_merging_frame_group_strategy: str = "local",
        token_merging_frame_protect_period: int = 0,
        token_merging_frame_protect_prefix: int = 0,
        enable_sparse_vggt: bool = False,
        sparse_vggt_sparse_ratio: float | None = 0.5,
        sparse_vggt_cdf_threshold: float | None = None,
        sparse_vggt_pool_mode: str = "avg",
        enable_da_vggt: bool = False,
        da_vggt_max_frames: int = 0,
        da_vggt_sampling_method: str = "fl_maxmin",
        da_vggt_n_anchors: int = 1,
        da_vggt_dino_batch_size: int = 256,
        da_vggt_lambda_div: float = 0.0,
    ) -> None:
        super().__init__()

        self.patch_embed = _build_patch_embed(patch_size=patch_size, embed_dim=embed_dim)
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=100,
            normalize_coords="max",
            dtype=torch.float32,
        )

        self.frame_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    ffn_ratio=mlp_ratio,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    ffn_layer=Mlp,
                    init_values=1e-5,
                    use_qk_norm=True,
                    mask_k_bias=True,
                )
                for _ in range(depth)
            ]
        )
        self.inter_frame_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    ffn_ratio=mlp_ratio,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    ffn_layer=Mlp,
                    init_values=1e-5,
                    use_qk_norm=True,
                    mask_k_bias=True,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.patch_size = patch_size
        self.cached_layer_indices = set(cached_layer_indices)
        self.camera_token = nn.Parameter(torch.empty(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.empty(1, 2, num_register_tokens, embed_dim))
        self.patch_token_start = 1 + num_register_tokens
        self.enable_token_merging = enable_token_merging
        self.token_merging_start = token_merging_start
        self.token_merging_ratio = token_merging_ratio
        self.token_merging_layer_ratios = token_merging_layer_ratios
        self._token_merging_layer_ratio_map = _parse_layer_ratio_schedule(token_merging_layer_ratios, depth)
        self.token_merging_method = token_merging_method
        self.token_merging_tstm_threshold = token_merging_tstm_threshold
        if token_merging_tstm_neighbor_size < 0 or (
            token_merging_tstm_neighbor_size > 0 and token_merging_tstm_neighbor_size % 2 == 0
        ):
            raise ValueError("token_merging_tstm_neighbor_size must be 0 for full search or a positive odd integer")
        self.token_merging_tstm_neighbor_size = token_merging_tstm_neighbor_size
        self.token_merging_flashvid_alpha = token_merging_flashvid_alpha
        self.token_merging_flashvid_expansion = token_merging_flashvid_expansion
        if token_merging_flashvid_pool_stride < 1:
            raise ValueError("token_merging_flashvid_pool_stride must be positive")
        self.token_merging_flashvid_pool_stride = token_merging_flashvid_pool_stride
        self.token_merging_frame_restore_layer = token_merging_frame_restore_layer
        self.token_merging_frame_alpha = token_merging_frame_alpha
        self.token_merging_frame_segment_threshold = token_merging_frame_segment_threshold
        self.token_merging_frame_merge_threshold = token_merging_frame_merge_threshold
        if token_merging_frame_max_window < 0 or token_merging_frame_max_window == 1:
            raise ValueError("token_merging_frame_max_window must be 0 for unlimited or at least 2")
        self.token_merging_frame_max_window = token_merging_frame_max_window
        if token_merging_frame_pool_stride < 1:
            raise ValueError("token_merging_frame_pool_stride must be positive")
        self.token_merging_frame_pool_stride = token_merging_frame_pool_stride
        if token_merging_frame_multi_max_group_size < 2:
            raise ValueError("token_merging_frame_multi_max_group_size must be at least 2")
        self.token_merging_frame_multi_max_group_size = token_merging_frame_multi_max_group_size
        self.token_merging_frame_multi_pair_threshold = token_merging_frame_multi_pair_threshold
        self.token_merging_frame_multi_span_threshold = token_merging_frame_multi_span_threshold
        if token_merging_frame_protect_period < 0 or token_merging_frame_protect_prefix < 0:
            raise ValueError("token_merging_frame_protect_period/prefix must be non-negative")
        self.token_merging_frame_protect_period = token_merging_frame_protect_period
        self.token_merging_frame_protect_prefix = token_merging_frame_protect_prefix
        if token_merging_frame_group_strategy not in {
            "local",
            "segment_middle",
            "global_cluster",
            "global_top_pairs",
        }:
            raise ValueError(
                "token_merging_frame_group_strategy must be 'local', 'segment_middle', "
                "'global_cluster', or 'global_top_pairs'"
            )
        self.token_merging_frame_group_strategy = token_merging_frame_group_strategy
        self.token_merging_protected_fraction = 0.5
        self.last_frame_merge_stats: list[dict[str, float | int | str]] = []
        self.last_token_merging_stats: list[dict[str, float | int | str]] = []
        self.enable_sparse_vggt = enable_sparse_vggt
        self.sparse_vggt_sparse_ratio = sparse_vggt_sparse_ratio
        self.sparse_vggt_cdf_threshold = sparse_vggt_cdf_threshold
        self.sparse_vggt_pool_mode = sparse_vggt_pool_mode
        self.last_sparse_vggt_stats: list[dict[str, float | int | str | None]] = []
        if enable_sparse_vggt:
            from vggt_omega.models.sparse_vggt_attention import check_sparse_vggt_mode

            check_sparse_vggt_mode(sparse_vggt_sparse_ratio, sparse_vggt_cdf_threshold)
            if sparse_vggt_pool_mode not in {"avg", "max"}:
                raise ValueError("sparse_vggt_pool_mode must be 'avg' or 'max'")

        self.enable_da_vggt = enable_da_vggt
        self.da_vggt_max_frames = da_vggt_max_frames
        self.da_vggt_sampling_method = da_vggt_sampling_method
        self.da_vggt_n_anchors = da_vggt_n_anchors
        self.da_vggt_dino_batch_size = da_vggt_dino_batch_size
        self.da_vggt_lambda_div = da_vggt_lambda_div

        self.inter_frame_attention_types = ["global"] * depth
        if register_attention_block_indices is None:
            register_attention_block_indices = [2, 6, 9, 14, 20]
        for idx in register_attention_block_indices:
            if idx < 0 or idx >= depth:
                raise ValueError(f"register_attention_block_indices contains invalid block index {idx}")
            self.inter_frame_attention_types[idx] = "register"

        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.init_weights()

    def init_weights(self) -> None:
        nn.init.normal_(self.camera_token, std=1e-3)
        nn.init.normal_(self.register_token, std=1e-3)

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[list[torch.Tensor | None], int]:
        batch_size, num_frames, num_channels, height, width = images.shape
        self.last_frame_merge_stats = []
        self.last_token_merging_stats = []
        self.last_sparse_vggt_stats = []
        if num_channels != 3:
            raise ValueError(f"Expected 3 input channels, got {num_channels}")

        images = (images - self._resnet_mean) / self._resnet_std
        images = images.view(batch_size * num_frames, num_channels, height, width)

        camera_token = slice_expand_and_flatten(self.camera_token, batch_size, num_frames)
        register_token = slice_expand_and_flatten(self.register_token, batch_size, num_frames)

        patch_tokens = self.patch_embed(images)
        cls_attention = None
        if isinstance(patch_tokens, dict):
            cls_attention = patch_tokens.get("x_cls_attention")
            if cls_attention is None:
                cls_attention = _flashvid_cls_attention_proxy(
                    patch_tokens["x_norm_patchtokens"],
                    patch_tokens.get("x_norm_clstoken"),
                )
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        patch_grid_size = (height // self.patch_size, width // self.patch_size)
        original_patch_tokens = None
        flashvid_inverse = None
        flashvid_merge_centers = None
        flashvid_rope_indices = None
        if self.enable_token_merging and self.token_merging_method == "flashvid_encoder":
            original_patch_tokens = patch_tokens.view(batch_size, num_frames, patch_tokens.shape[1], patch_tokens.shape[2])
            cls_attention = cls_attention.view(batch_size, num_frames, patch_tokens.shape[1])
            (
                compressed_patch_tokens,
                flashvid_inverse,
                flashvid_merge_centers,
                flashvid_rope_indices,
            ) = self._flashvid_encoder_compress(
                original_patch_tokens,
                cls_attention,
                patch_grid_size,
            )
            patch_tokens = compressed_patch_tokens.reshape(
                batch_size * num_frames,
                compressed_patch_tokens.shape[2],
                compressed_patch_tokens.shape[3],
            )

        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
        _, num_tokens, embed_dim = tokens.shape

        with torch.no_grad():
            rope_sin, rope_cos = self.rope_embed(H=patch_grid_size[0], W=patch_grid_size[1])
            if flashvid_rope_indices is not None:
                frame_rope = _gather_flashvid_rope(
                    rope_sin.to(device=patch_tokens.device, dtype=torch.float32),
                    rope_cos.to(device=patch_tokens.device, dtype=torch.float32),
                    flashvid_rope_indices.reshape(batch_size * num_frames, -1),
                )
            else:
                frame_rope = (
                    rope_sin.to(device=patch_tokens.device, dtype=torch.float32),
                    rope_cos.to(device=patch_tokens.device, dtype=torch.float32),
                )

        outputs = []
        frame_merge_state = None
        active_num_frames = num_frames
        for block_idx in range(self.depth):
            if (
                frame_merge_state is not None
                and self.token_merging_method
                in {
                    "frame_persistent",
                    "frame_persistent_spatial",
                    "frame_persistent_decoupled",
                    "frame_persistent_decoupled_window",
                }
                and block_idx == self.token_merging_frame_restore_layer
            ):
                tokens = _restore_frame_tokens(tokens, frame_merge_state)
                frame_merge_state = None
                active_num_frames = num_frames

            tokens, frame_tokens = self._run_frame_block(
                tokens,
                batch_size,
                active_num_frames,
                num_tokens,
                embed_dim,
                block_idx,
                frame_rope,
            )
            tokens = self._run_inter_frame_attention_block(
                tokens,
                batch_size,
                active_num_frames,
                num_frames,
                num_tokens,
                embed_dim,
                block_idx,
                self.inter_frame_attention_types[block_idx],
                patch_grid_size,
                _frame_dynamic_masks(frame_merge_state, active_num_frames, tokens.device)
                if frame_merge_state is not None
                else None,
            )
            if block_idx in self.cached_layer_indices:
                cache_frame_tokens = frame_tokens
                cache_tokens = tokens
                if frame_merge_state is not None:
                    cache_frame_tokens = _restore_frame_tokens(cache_frame_tokens, frame_merge_state)
                    cache_tokens = _restore_frame_tokens(cache_tokens, frame_merge_state)
                output_tokens = torch.cat([cache_frame_tokens, cache_tokens], dim=-1)
                if flashvid_inverse is not None:
                    output_tokens = self._restore_flashvid_encoder_tokens(
                        output_tokens,
                        flashvid_inverse,
                        original_patch_tokens,
                        flashvid_merge_centers,
                    )
                outputs.append(output_tokens)
            else:
                outputs.append(None)

            if (
                self.enable_token_merging
                and self.token_merging_method
                in {
                    "frame_persistent",
                    "frame_persistent_spatial",
                    "frame_persistent_decoupled",
                    "frame_persistent_decoupled_window",
                }
                and block_idx == self.token_merging_start
                and block_idx < self.token_merging_frame_restore_layer
                and frame_merge_state is None
            ):
                tokens, frame_merge_state = self._frame_merge(tokens, patch_grid_size)
                self._record_frame_merge_stats(block_idx, "persistent", num_frames, frame_merge_state.states)
                active_num_frames = tokens.shape[1]

        return outputs, self.patch_token_start

    def forward_dino(
        self,
        images: torch.Tensor,
        dino_batch_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if dino_batch_size is None:
            dino_batch_size = self.da_vggt_dino_batch_size
        batch_size, num_frames, num_channels, height, width = images.shape
        if batch_size != 1:
            raise ValueError("DA-VGGT chunked inference currently supports batch size 1.")
        if num_channels != 3:
            raise ValueError(f"Expected 3 input channels, got {num_channels}")

        images = (images - self._resnet_mean) / self._resnet_std
        images = images.view(batch_size * num_frames, num_channels, height, width)
        all_patch_tokens = []
        all_pooled_tokens = []
        for start in range(0, batch_size * num_frames, dino_batch_size):
            output = self.patch_embed(images[start : start + dino_batch_size])
            if isinstance(output, dict):
                patch_tokens = output["x_norm_patchtokens"]
                pooled_tokens = output.get("x_norm_clstoken")
                if pooled_tokens is None:
                    pooled_tokens = patch_tokens.float().mean(dim=1)
            else:
                patch_tokens = output
                pooled_tokens = patch_tokens.float().mean(dim=1)
            all_patch_tokens.append(patch_tokens.cpu())
            all_pooled_tokens.append(pooled_tokens.float().cpu())
        return torch.cat(all_patch_tokens, dim=0), torch.cat(all_pooled_tokens, dim=0)

    def forward_transformer(
        self,
        patch_tokens: torch.Tensor,
        *,
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> tuple[list[torch.Tensor | None], int]:
        camera_token = slice_expand_and_flatten(self.camera_token, batch_size, num_frames)
        register_token = slice_expand_and_flatten(self.register_token, batch_size, num_frames)
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
        _, num_tokens, embed_dim = tokens.shape
        patch_grid_size = (height // self.patch_size, width // self.patch_size)
        with torch.no_grad():
            rope_sin, rope_cos = self.rope_embed(H=patch_grid_size[0], W=patch_grid_size[1])
            frame_rope = (
                rope_sin.to(device=device, dtype=torch.float32),
                rope_cos.to(device=device, dtype=torch.float32),
            )

        outputs = []
        for block_idx in range(self.depth):
            tokens, frame_tokens = self._run_frame_block(
                tokens,
                batch_size,
                num_frames,
                num_tokens,
                embed_dim,
                block_idx,
                frame_rope,
            )
            tokens = self._run_inter_frame_attention_block(
                tokens,
                batch_size,
                num_frames,
                num_frames,
                num_tokens,
                embed_dim,
                block_idx,
                self.inter_frame_attention_types[block_idx],
                patch_grid_size,
            )
            if block_idx in self.cached_layer_indices:
                outputs.append(torch.cat([frame_tokens, tokens], dim=-1))
            else:
                outputs.append(None)
        return outputs, self.patch_token_start

    def _run_frame_block(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        num_frames: int,
        num_tokens: int,
        embed_dim: int,
        block_idx: int,
        rope_sincos: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = tokens.view(batch_size * num_frames, num_tokens, embed_dim)
        tokens = self.frame_blocks[block_idx](tokens, rope_sincos)
        return tokens, tokens.view(batch_size, num_frames, num_tokens, embed_dim)

    def _run_inter_frame_attention_block(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        num_frames: int,
        original_num_frames: int,
        num_tokens: int,
        embed_dim: int,
        block_idx: int,
        attention_type: str,
        patch_grid_size: tuple[int, int],
        dynamic_frame_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens = tokens.view(batch_size, num_frames, num_tokens, embed_dim)

        if attention_type == "global":
            if (
                self.enable_token_merging
                and block_idx >= self.token_merging_start
                and self.token_merging_method == "frame_temporary"
            ):
                merged_sequences = []
                frame_states = []
                for batch_idx in range(batch_size):
                    merged_tokens, state = _frame_merge_one(
                        tokens[batch_idx],
                        self.patch_token_start,
                        patch_grid_size,
                        self.token_merging_frame_alpha,
                        self.token_merging_frame_segment_threshold,
                        self.token_merging_frame_merge_threshold,
                        self.token_merging_frame_max_window,
                        self.token_merging_frame_pool_stride,
                        self.token_merging_frame_multi_max_group_size,
                        self.token_merging_frame_multi_pair_threshold,
                        self.token_merging_frame_multi_span_threshold,
                        self.token_merging_frame_group_strategy,
                    )
                    merged_sequences.append(merged_tokens.reshape(1, -1, embed_dim))
                    frame_states.append(state)
                self._record_frame_merge_stats(block_idx, "temporary", num_frames, frame_states)
                merged_outputs = self.inter_frame_blocks[block_idx](merged_sequences, None)
                outputs = []
                for merged_output, state in zip(merged_outputs, frame_states):
                    active_frames = merged_output.squeeze(0).view(state.active_frames, num_tokens, embed_dim)
                    outputs.append(active_frames[state.inverse])
                return torch.stack(outputs, dim=0)

            if (
                self.enable_token_merging
                and block_idx >= self.token_merging_start
                and self.token_merging_method
                in {
                    "spatial",
                    "protected_spatial",
                    "frame_persistent_spatial",
                    "frame_persistent_decoupled",
                    "frame_persistent_decoupled_window",
                }
                and (
                    self.token_merging_method
                    not in {"frame_persistent_decoupled", "frame_persistent_decoupled_window"}
                    or dynamic_frame_masks is not None
                )
            ):
                tokens = tokens.view(batch_size, num_frames * num_tokens, embed_dim)
                block = self.inter_frame_blocks[block_idx]
                layer_ratio = self._token_merging_ratio_for_block(block_idx)
                block.attn.fastvggt_merge_ratio = layer_ratio
                if self.token_merging_method == "protected_spatial":
                    block.attn.fastvggt_protection = "adts"
                elif self.token_merging_method == "frame_persistent_decoupled_window":
                    block.attn.fastvggt_protection = "decoupled_window"
                else:
                    block.attn.fastvggt_protection = "fastvggt"
                block.attn.fastvggt_patch_grid_size = patch_grid_size
                block.attn.fastvggt_num_frames = num_frames
                block.attn.fastvggt_special_token_count = self.patch_token_start
                if self.token_merging_method == "frame_persistent_decoupled":
                    block.attn.fastvggt_protected_token_indices = _dynamic_frame_token_indices(
                        dynamic_frame_masks,
                        num_tokens,
                        tokens.device,
                    )
                elif self.token_merging_method == "frame_persistent_decoupled_window":
                    block.attn.fastvggt_dynamic_frame_mask = dynamic_frame_masks
                try:
                    tokens = block(tokens, None)
                    stats = getattr(block.attn, "last_fastvggt_stats", None)
                    if stats:
                        active_tokens = int(stats["active_tokens"])
                        frame_merged_tokens = int(stats["original_tokens"])
                        frame_original_tokens = int(original_num_frames * num_tokens)
                        self.last_token_merging_stats.append(
                            {
                                "block": int(block_idx),
                                "mode": self.token_merging_method,
                                "merge_ratio": float(layer_ratio),
                                "frame_merged_tokens": frame_merged_tokens,
                                "frame_original_tokens": frame_original_tokens,
                                "active_over_frame_merged_token_ratio": float(
                                    active_tokens / frame_merged_tokens if frame_merged_tokens else 0.0
                                ),
                                "active_over_frame_original_token_ratio": float(
                                    active_tokens / frame_original_tokens if frame_original_tokens else 0.0
                                ),
                                **stats,
                            }
                        )
                finally:
                    block.attn.fastvggt_merge_ratio = None
                    block.attn.fastvggt_protection = None
                    block.attn.fastvggt_patch_grid_size = None
                    block.attn.fastvggt_num_frames = None
                    block.attn.fastvggt_special_token_count = None
                    block.attn.fastvggt_protected_token_indices = None
                    block.attn.fastvggt_dynamic_frame_mask = None
                return tokens.view(batch_size, num_frames, num_tokens, embed_dim)

            if (
                self.enable_token_merging
                and block_idx >= self.token_merging_start
                and self.token_merging_method == "tstm"
            ):
                return self._run_merged_global_attention(
                    tokens,
                    batch_size,
                    num_frames,
                    num_tokens,
                    embed_dim,
                    block_idx,
                    patch_grid_size,
                )
            tokens = tokens.view(batch_size, num_frames * num_tokens, embed_dim)
            if not self.enable_sparse_vggt:
                tokens = self.inter_frame_blocks[block_idx](tokens, None)
                return tokens.view(batch_size, num_frames, num_tokens, embed_dim)

            block = self.inter_frame_blocks[block_idx]
            block.attn.sparse_vggt_config = {
                "sparse_ratio": self.sparse_vggt_sparse_ratio,
                "cdf_threshold": self.sparse_vggt_cdf_threshold,
                "pool_mode": self.sparse_vggt_pool_mode,
                "num_frames": num_frames,
                "patch_grid_size": patch_grid_size,
                "special_token_count": self.patch_token_start,
            }
            try:
                tokens = block(tokens, None)
                sparse_stats = getattr(block.attn, "last_sparse_vggt_stats", None)
                if sparse_stats:
                    self.last_sparse_vggt_stats.append({"block": int(block_idx), **sparse_stats})
            finally:
                block.attn.sparse_vggt_config = None
            return tokens.view(batch_size, num_frames, num_tokens, embed_dim)

        if attention_type != "register":
            raise ValueError(f"Unknown inter-frame attention type: {attention_type}")

        patch_token_start = self.patch_token_start
        camera_and_register_tokens = tokens[:, :, :patch_token_start].reshape(
            batch_size,
            num_frames * patch_token_start,
            embed_dim,
        )
        patch_tokens = tokens[:, :, patch_token_start:].reshape(
            batch_size,
            num_frames * (num_tokens - patch_token_start),
            embed_dim,
        )

        camera_and_register_tokens = self.inter_frame_blocks[block_idx](camera_and_register_tokens, None)
        tokens = torch.cat([camera_and_register_tokens, patch_tokens], dim=1)

        camera_and_register_tokens = tokens[:, : num_frames * patch_token_start].view(
            batch_size,
            num_frames,
            patch_token_start,
            embed_dim,
        )
        patch_tokens = tokens[:, num_frames * patch_token_start :].view(
            batch_size,
            num_frames,
            num_tokens - patch_token_start,
            embed_dim,
        )
        return torch.cat([camera_and_register_tokens, patch_tokens], dim=2)

    def _run_merged_global_attention(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        num_frames: int,
        num_tokens: int,
        embed_dim: int,
        block_idx: int,
        patch_grid_size: tuple[int, int],
    ) -> torch.Tensor:
        special_tokens = tokens[:, :, : self.patch_token_start]
        patch_tokens = tokens[:, :, self.patch_token_start :]

        merged_sequences = []
        inverse_maps = []
        merge_centers = []
        special_count = num_frames * self.patch_token_start

        for batch_idx in range(batch_size):
            merged_patch, inverse = self._merge_patch_tokens(
                patch_tokens[batch_idx],
                patch_grid_size,
            )
            merged_sequences.append(
                torch.cat(
                    [
                        special_tokens[batch_idx].reshape(special_count, embed_dim),
                        merged_patch,
                    ],
                    dim=0,
                ).unsqueeze(0)
            )
            inverse_maps.append(inverse)
            merge_centers.append(merged_patch[inverse])

        merged_outputs = self.inter_frame_blocks[block_idx](merged_sequences, None)
        outputs = []
        for batch_idx, merged_output in enumerate(merged_outputs):
            merged_output = merged_output.squeeze(0)
            special_output = merged_output[:special_count].view(num_frames, self.patch_token_start, embed_dim)
            patch_output = merged_output[special_count:][inverse_maps[batch_idx]]
            patch_flat = patch_tokens[batch_idx].reshape(num_frames * (num_tokens - self.patch_token_start), embed_dim)
            patch_output = patch_output + (patch_flat - merge_centers[batch_idx])
            patch_output = patch_output.view(num_frames, num_tokens - self.patch_token_start, embed_dim)
            outputs.append(torch.cat([special_output, patch_output], dim=1))
        return torch.stack(outputs, dim=0)

    def _merge_patch_tokens(
        self,
        patch_tokens: torch.Tensor,
        patch_grid_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        retention_ratio = _retention_from_merge_ratio(self.token_merging_ratio)
        if self.token_merging_method == "spatial":
            raise AssertionError("spatial merging is applied inside attention like FastVGGT")
        if self.token_merging_method == "protected_spatial":
            raise AssertionError("protected_spatial merging is applied inside attention like FastVGGT")
        if self.token_merging_method == "tstm":
            return _tstm_merge(
                patch_tokens,
                patch_grid_size,
                retention_ratio,
                self.token_merging_protected_fraction,
                self.token_merging_tstm_threshold,
                self.token_merging_tstm_neighbor_size,
            )
        raise ValueError(
            f"Unknown token_merging_method={self.token_merging_method!r}; "
            "expected 'spatial', 'protected_spatial', 'tstm', 'flashvid_encoder', "
            "'frame_temporary', 'frame_persistent', 'frame_persistent_spatial', "
            "'frame_persistent_decoupled', or 'frame_persistent_decoupled_window'"
        )

    def _token_merging_ratio_for_block(self, block_idx: int) -> float:
        return self._token_merging_layer_ratio_map.get(block_idx, self.token_merging_ratio)

    def _frame_merge(
        self,
        tokens: torch.Tensor,
        patch_grid_size: tuple[int, int],
    ) -> tuple[torch.Tensor, "_FrameMergeState"]:
        merged_batches = []
        states = []
        for batch_idx in range(tokens.shape[0]):
            merged_tokens, state = _frame_merge_one(
                tokens[batch_idx],
                self.patch_token_start,
                patch_grid_size,
                self.token_merging_frame_alpha,
                self.token_merging_frame_segment_threshold,
                self.token_merging_frame_merge_threshold,
                self.token_merging_frame_max_window,
                self.token_merging_frame_pool_stride,
                self.token_merging_frame_multi_max_group_size,
                self.token_merging_frame_multi_pair_threshold,
                self.token_merging_frame_multi_span_threshold,
                self.token_merging_frame_group_strategy,
                self.token_merging_frame_protect_period,
                self.token_merging_frame_protect_prefix,
            )
            merged_batches.append(merged_tokens)
            states.append(state)
        active_counts = {state.active_frames for state in states}
        if len(active_counts) != 1:
            raise ValueError(f"Frame merging produced different active frame counts across batch: {sorted(active_counts)}")
        return torch.stack(merged_batches, dim=0), _BatchFrameMergeState(states)

    def _record_frame_merge_stats(
        self,
        block_idx: int,
        mode: str,
        original_frames: int,
        states: list["_FrameMergeState"],
    ) -> None:
        if not states:
            return
        active_frames = [state.active_frames for state in states]
        active_mean = sum(active_frames) / len(active_frames)
        retention_ratio = active_mean / original_frames if original_frames else 0.0
        segment_counts = [len(state.segments) for state in states]
        segments = [[(int(start), int(end)) for start, end in state.segments] for state in states]
        merge_group_sizes = [size for state in states for size in state.merge_group_sizes]
        merge_groups = [
            [[int(frame_idx) for frame_idx in group] for group in state.merge_groups]
            for state in states
        ]
        frame_to_active = [
            [int(active_idx) for active_idx in state.inverse.detach().cpu().tolist()]
            for state in states
        ]
        similarity_matrices = [
            state.similarity_matrix.detach().cpu().tolist()
            for state in states
            if state.similarity_matrix is not None
        ]
        multi_group_sizes = [size for size in merge_group_sizes if size > 2]
        stat = {
            "block": int(block_idx),
            "mode": mode,
            "original_frames": int(original_frames),
            "active_frames_min": int(min(active_frames)),
            "active_frames_mean": float(active_mean),
            "active_frames_max": int(max(active_frames)),
            "retention_ratio_mean": float(retention_ratio),
            "merge_ratio_mean": float(1.0 - retention_ratio),
            "segments_min": int(min(segment_counts)),
            "segments_mean": float(sum(segment_counts) / len(segment_counts)),
            "segments_max": int(max(segment_counts)),
            "segments": segments,
            "merge_groups": merge_groups,
            "frame_to_active": frame_to_active,
            "merge_groups_count": int(len(merge_group_sizes)),
            "merge_group_size_mean": float(sum(merge_group_sizes) / len(merge_group_sizes))
            if merge_group_sizes
            else 0.0,
            "merge_group_size_max": int(max(merge_group_sizes)) if merge_group_sizes else 0,
            "multi_frame_groups_count": int(len(multi_group_sizes)),
            "multi_frame_group_size_mean": float(sum(multi_group_sizes) / len(multi_group_sizes))
            if multi_group_sizes
            else 0.0,
                "frame_group_strategy": self.token_merging_frame_group_strategy,
                "protect_period": int(self.token_merging_frame_protect_period),
                "protect_prefix": int(self.token_merging_frame_protect_prefix),
            }
        if similarity_matrices:
            stat["similarity_matrices"] = similarity_matrices
        self.last_frame_merge_stats.append(stat)

    def _flashvid_encoder_compress(
        self,
        patch_tokens: torch.Tensor,
        cls_attention: torch.Tensor,
        patch_grid_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        retention_ratio = _retention_from_merge_ratio(self.token_merging_ratio)
        return _flashvid_encoder_compress(
            patch_tokens,
            cls_attention,
            patch_grid_size,
            retention_ratio,
            self.token_merging_flashvid_alpha,
            self.token_merging_flashvid_expansion,
            self.token_merging_flashvid_pool_stride,
            self.token_merging_tstm_threshold,
        )

    def _restore_flashvid_encoder_tokens(
        self,
        tokens: torch.Tensor,
        inverse: torch.Tensor,
        original_patch_tokens: torch.Tensor,
        merge_centers: torch.Tensor,
    ) -> torch.Tensor:
        special_tokens = tokens[:, :, : self.patch_token_start]
        patch_tokens = tokens[:, :, self.patch_token_start :]
        restored = torch.gather(
            patch_tokens,
            dim=2,
            index=inverse.unsqueeze(-1).expand(-1, -1, -1, patch_tokens.shape[-1]),
        )
        return torch.cat([special_tokens, restored], dim=2)


class _FrameMergeState:
    def __init__(
        self,
        inverse: torch.Tensor,
        active_mask: torch.Tensor,
        segments: list[tuple[int, int]],
        merge_group_sizes: list[int] | None = None,
        merge_groups: list[list[int]] | None = None,
        similarity_matrix: torch.Tensor | None = None,
    ) -> None:
        self.inverse = inverse
        self.active_mask = active_mask
        self.segments = segments
        self.merge_group_sizes = merge_group_sizes or []
        self.merge_groups = merge_groups or []
        self.similarity_matrix = similarity_matrix
        self.active_frames = int(inverse.max().item()) + 1 if inverse.numel() else 0


class _BatchFrameMergeState:
    def __init__(self, states: list[_FrameMergeState]) -> None:
        self.states = states
        self.active_frames = states[0].active_frames if states else 0


def _restore_frame_tokens(tokens: torch.Tensor, state: _FrameMergeState | _BatchFrameMergeState) -> torch.Tensor:
    if isinstance(state, _BatchFrameMergeState):
        restored = [tokens[batch_idx, frame_state.inverse] for batch_idx, frame_state in enumerate(state.states)]
        return torch.stack(restored, dim=0)
    return tokens[:, state.inverse]


def _frame_dynamic_masks(
    state: _FrameMergeState | _BatchFrameMergeState,
    active_num_frames: int,
    device: torch.device,
) -> torch.Tensor:
    states = state.states if isinstance(state, _BatchFrameMergeState) else [state]
    masks = []
    for frame_state in states:
        mask = torch.zeros(active_num_frames, dtype=torch.bool, device=device)
        for start, end in frame_state.segments:
            mask[frame_state.inverse[start].to(device=device)] = True
            mask[frame_state.inverse[end].to(device=device)] = True
        masks.append(mask)
    return torch.stack(masks, dim=0)


def _dynamic_frame_token_indices(
    dynamic_frame_masks: torch.Tensor,
    num_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    if dynamic_frame_masks.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    dynamic_frames = torch.nonzero(dynamic_frame_masks[0].to(device=device), as_tuple=False).flatten()
    if dynamic_frames.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    token_offsets = torch.arange(num_tokens, device=device, dtype=torch.long)
    return (dynamic_frames[:, None] * num_tokens + token_offsets[None, :]).flatten()


def _frame_merge_one(
    tokens: torch.Tensor,
    patch_token_start: int,
    patch_grid_size: tuple[int, int],
    alpha: float,
    segment_threshold: float,
    merge_threshold: float,
    max_window: int,
    pool_stride: int,
    multi_max_group_size: int,
    multi_pair_threshold: float,
    multi_span_threshold: float,
    group_strategy: str = "local",
    protect_period: int = 0,
    protect_prefix: int = 0,
) -> tuple[torch.Tensor, _FrameMergeState]:
    num_frames = tokens.shape[0]
    if num_frames <= 1:
        inverse = torch.arange(num_frames, device=tokens.device)
        active_mask = torch.ones(num_frames, dtype=torch.bool, device=tokens.device)
        return tokens, _FrameMergeState(inverse, active_mask, [(0, num_frames - 1)], [])

    patch_tokens = tokens[:, patch_token_start:]
    pooled = _pool_frame_similarity_tokens(patch_tokens, patch_grid_size, pool_stride)
    if group_strategy == "global_cluster":
        return _frame_merge_global_cluster(
            tokens,
            pooled,
            sim_threshold=multi_pair_threshold,
            max_group_size=multi_max_group_size,
            protect_period=protect_period,
            protect_prefix=protect_prefix,
        )
    if group_strategy == "global_top_pairs":
        return _frame_merge_global_top_pairs(
            tokens,
            pooled,
            target_merge_ratio=multi_span_threshold,
        )
    segments = _streaming_frame_segments(pooled, alpha, segment_threshold, max_window)

    active_tokens = []
    inverse = torch.empty(num_frames, dtype=torch.long, device=tokens.device)
    active_mask = torch.zeros(num_frames, dtype=torch.bool, device=tokens.device)
    assigned = [False] * num_frames
    merge_group_sizes: list[int] = []
    merge_groups: list[list[int]] = []

    def append_frame(frame_idx: int) -> None:
        inverse[frame_idx] = len(active_tokens)
        active_mask[frame_idx] = True
        assigned[frame_idx] = True
        active_tokens.append(tokens[frame_idx])

    def append_group(frame_indices: list[int]) -> None:
        active_idx = len(active_tokens)
        for offset, frame_idx in enumerate(frame_indices):
            inverse[frame_idx] = active_idx
            active_mask[frame_idx] = offset == 0
            assigned[frame_idx] = True
        merged = tokens[frame_indices].float().mean(dim=0)
        active_tokens.append(merged.to(tokens.dtype))
        merge_group_sizes.append(len(frame_indices))
        merge_groups.append([int(frame_idx) for frame_idx in frame_indices])

    def append_merge(left_idx: int, right_idx: int, cur_sim: torch.Tensor, next_sim: torch.Tensor) -> None:
        weight_left = cur_sim.float().clamp_min(1e-6)
        weight_right = next_sim.float().clamp_min(1e-6)
        merged = (weight_left * tokens[left_idx] + weight_right * tokens[right_idx]) / (weight_left + weight_right)
        active_idx = len(active_tokens)
        inverse[left_idx] = active_idx
        inverse[right_idx] = active_idx
        active_mask[left_idx] = True
        assigned[left_idx] = True
        assigned[right_idx] = True
        active_tokens.append(merged.to(tokens.dtype))
        merge_group_sizes.append(2)
        merge_groups.append([int(left_idx), int(right_idx)])

    def can_merge_group(start_idx: int, group_size: int) -> bool:
        end_idx = start_idx + group_size - 1
        if end_idx > end:
            return False
        pair_sims = [
            _frame_pair_similarity(pooled[idx], pooled[idx + 1]).float()
            for idx in range(start_idx, end_idx)
        ]
        span_sim = _frame_pair_similarity(pooled[start_idx], pooled[end_idx]).float()
        return bool(
            all(sim > multi_pair_threshold for sim in pair_sims)
            and span_sim > multi_span_threshold
        )

    for start, end in segments:
        if start == end:
            append_frame(start)
            continue
        append_frame(start)
        if group_strategy == "segment_middle":
            if end - start > 1:
                append_group(list(range(start + 1, end)))
            if not assigned[end]:
                append_frame(end)
            continue
        cursor = start + 1
        while cursor < end:
            group_size = 0
            for candidate_size in range(min(multi_max_group_size, 4), 2, -1):
                if can_merge_group(cursor, candidate_size):
                    group_size = candidate_size
                    break
            if group_size:
                append_group(list(range(cursor, cursor + group_size)))
                cursor += group_size
                continue

            cur_sim = _frame_pair_similarity(pooled[cursor], pooled[cursor + 1])
            next_sim = (
                _frame_pair_similarity(pooled[cursor + 1], pooled[cursor + 2])
                if cursor + 1 < end
                else cur_sim
            )
            if cur_sim > merge_threshold and cur_sim > next_sim:
                append_merge(cursor, cursor + 1, cur_sim, next_sim)
                cursor += 2
            else:
                append_frame(cursor)
                cursor += 1
        if not assigned[end]:
            append_frame(end)

    merged_tokens = torch.stack(active_tokens, dim=0)
    return merged_tokens, _FrameMergeState(inverse, active_mask, segments, merge_group_sizes, merge_groups)


def _frame_merge_global_cluster(
    tokens: torch.Tensor,
    pooled: torch.Tensor,
    sim_threshold: float,
    max_group_size: int,
    protect_period: int = 0,
    protect_prefix: int = 0,
) -> tuple[torch.Tensor, _FrameMergeState]:
    num_frames = tokens.shape[0]
    sim_matrix = _frame_similarity_matrix(pooled)
    pair_scores = []
    for left_idx in range(num_frames):
        for right_idx in range(left_idx + 1, num_frames):
            score = float(sim_matrix[left_idx, right_idx].item())
            if score >= sim_threshold:
                pair_scores.append((score, left_idx, right_idx))
    pair_scores.sort(reverse=True, key=lambda item: item[0])

    clusters: list[list[int]] = []
    frame_to_cluster: list[int | None] = [None] * num_frames

    def is_protected(frame_idx: int) -> bool:
        return protect_period > 0 and protect_prefix > 0 and frame_idx % protect_period < protect_prefix

    def can_join(cluster: list[int], frame_idx: int) -> bool:
        if len(cluster) >= max_group_size:
            return False
        if is_protected(frame_idx) or any(is_protected(idx) for idx in cluster):
            return False
        sims = sim_matrix[frame_idx, torch.tensor(cluster, device=tokens.device, dtype=torch.long)]
        return bool(torch.all(sims >= sim_threshold).item())

    for _, left_idx, right_idx in pair_scores:
        if is_protected(left_idx) or is_protected(right_idx):
            continue
        left_cluster = frame_to_cluster[left_idx]
        right_cluster = frame_to_cluster[right_idx]
        if left_cluster is None and right_cluster is None:
            cluster_idx = len(clusters)
            clusters.append([left_idx, right_idx])
            frame_to_cluster[left_idx] = cluster_idx
            frame_to_cluster[right_idx] = cluster_idx
        elif left_cluster is not None and right_cluster is None:
            cluster = clusters[left_cluster]
            if can_join(cluster, right_idx):
                cluster.append(right_idx)
                frame_to_cluster[right_idx] = left_cluster
        elif left_cluster is None and right_cluster is not None:
            cluster = clusters[right_cluster]
            if can_join(cluster, left_idx):
                cluster.append(left_idx)
                frame_to_cluster[left_idx] = right_cluster

    for frame_idx, cluster_idx in enumerate(frame_to_cluster):
        if cluster_idx is None:
            frame_to_cluster[frame_idx] = len(clusters)
            clusters.append([frame_idx])

    clusters = [sorted(cluster) for cluster in clusters]
    clusters.sort(key=lambda cluster: cluster[0])
    inverse = torch.empty(num_frames, dtype=torch.long, device=tokens.device)
    active_mask = torch.zeros(num_frames, dtype=torch.bool, device=tokens.device)
    active_tokens = []
    merge_group_sizes: list[int] = []
    merge_groups: list[list[int]] = []
    for active_idx, cluster in enumerate(clusters):
        frame_indices = torch.tensor(cluster, device=tokens.device, dtype=torch.long)
        inverse[frame_indices] = active_idx
        active_mask[cluster[0]] = True
        merged = tokens[frame_indices].float().mean(dim=0).to(tokens.dtype)
        active_tokens.append(merged)
        if len(cluster) > 1:
            merge_group_sizes.append(len(cluster))
            merge_groups.append([int(frame_idx) for frame_idx in cluster])

    merged_tokens = torch.stack(active_tokens, dim=0)
    segments = [(idx, idx) for idx in range(num_frames)]
    return merged_tokens, _FrameMergeState(
        inverse,
        active_mask,
        segments,
        merge_group_sizes,
        merge_groups,
        sim_matrix,
    )


def _frame_merge_global_top_pairs(
    tokens: torch.Tensor,
    pooled: torch.Tensor,
    target_merge_ratio: float,
) -> tuple[torch.Tensor, _FrameMergeState]:
    num_frames = tokens.shape[0]
    if target_merge_ratio > 1.0:
        target_pairs = int(round(target_merge_ratio))
    else:
        target_pairs = int(round(num_frames * max(0.0, min(0.5, target_merge_ratio))))
    target_pairs = min(target_pairs, num_frames // 2)
    if target_pairs <= 0:
        inverse = torch.arange(num_frames, device=tokens.device)
        active_mask = torch.ones(num_frames, dtype=torch.bool, device=tokens.device)
        return tokens, _FrameMergeState(inverse, active_mask, [(idx, idx) for idx in range(num_frames)], [])

    sim_matrix = _frame_similarity_matrix(pooled)
    pair_scores = []
    for left_idx in range(num_frames):
        for right_idx in range(left_idx + 1, num_frames):
            pair_scores.append((float(sim_matrix[left_idx, right_idx].item()), left_idx, right_idx))
    pair_scores.sort(reverse=True, key=lambda item: item[0])

    used = [False] * num_frames
    selected_pairs: list[list[int]] = []
    for _, left_idx, right_idx in pair_scores:
        if used[left_idx] or used[right_idx]:
            continue
        selected_pairs.append([left_idx, right_idx])
        used[left_idx] = True
        used[right_idx] = True
        if len(selected_pairs) >= target_pairs:
            break

    pair_by_first = {pair[0]: pair for pair in selected_pairs}
    inverse = torch.empty(num_frames, dtype=torch.long, device=tokens.device)
    active_mask = torch.zeros(num_frames, dtype=torch.bool, device=tokens.device)
    active_tokens = []
    merge_group_sizes: list[int] = []
    merge_groups: list[list[int]] = []
    consumed = [False] * num_frames
    for frame_idx in range(num_frames):
        if consumed[frame_idx]:
            continue
        if frame_idx in pair_by_first:
            pair = pair_by_first[frame_idx]
            active_idx = len(active_tokens)
            frame_indices = torch.tensor(pair, device=tokens.device, dtype=torch.long)
            inverse[frame_indices] = active_idx
            active_mask[pair[0]] = True
            active_tokens.append(tokens[frame_indices].float().mean(dim=0).to(tokens.dtype))
            merge_group_sizes.append(2)
            merge_groups.append([int(idx) for idx in pair])
            consumed[pair[0]] = True
            consumed[pair[1]] = True
        else:
            active_idx = len(active_tokens)
            inverse[frame_idx] = active_idx
            active_mask[frame_idx] = True
            active_tokens.append(tokens[frame_idx])
            consumed[frame_idx] = True

    merged_tokens = torch.stack(active_tokens, dim=0)
    segments = [(idx, idx) for idx in range(num_frames)]
    return merged_tokens, _FrameMergeState(
        inverse,
        active_mask,
        segments,
        merge_group_sizes,
        merge_groups,
        sim_matrix,
    )


def _frame_similarity_matrix(pooled: torch.Tensor) -> torch.Tensor:
    device_type = pooled.device.type if pooled.device.type in {"cuda", "cpu"} else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        descriptors = F.normalize(pooled.float().mean(dim=1), dim=-1)
        return (descriptors @ descriptors.transpose(0, 1)).clamp_(-1.0, 1.0)


def _pool_frame_similarity_tokens(
    patch_tokens: torch.Tensor,
    patch_grid_size: tuple[int, int],
    pool_stride: int,
) -> torch.Tensor:
    num_frames, num_patches, embed_dim = patch_tokens.shape
    grid_h, grid_w = patch_grid_size
    if pool_stride <= 1:
        return patch_tokens
    if num_patches != grid_h * grid_w:
        raise ValueError(f"Patch token count {num_patches} does not match grid {patch_grid_size}")
    pooled_h = math.ceil(grid_h / pool_stride)
    pooled_w = math.ceil(grid_w / pool_stride)
    grid = patch_tokens.view(num_frames, grid_h, grid_w, embed_dim).permute(0, 3, 1, 2)
    pooled = F.avg_pool2d(grid.float(), kernel_size=pool_stride, stride=pool_stride, ceil_mode=True)
    return pooled.permute(0, 2, 3, 1).reshape(num_frames, pooled_h * pooled_w, embed_dim).to(patch_tokens.dtype)


def _streaming_frame_segments(
    pooled_tokens: torch.Tensor,
    alpha: float,
    threshold: float,
    max_window: int,
) -> list[tuple[int, int]]:
    num_frames = pooled_tokens.shape[0]
    segments = []
    start = 0
    ref = pooled_tokens[0]
    ema = pooled_tokens.new_tensor(1.0, dtype=torch.float32)
    for frame_idx in range(1, num_frames):
        sim = _frame_pair_similarity(ref, pooled_tokens[frame_idx])
        ema = alpha * sim.float() + (1.0 - alpha) * ema
        if ema < threshold or (max_window > 0 and (frame_idx - start + 1) >= max_window):
            segments.append((start, frame_idx - 1))
            start = frame_idx
            ref = pooled_tokens[frame_idx]
            ema = pooled_tokens.new_tensor(1.0, dtype=torch.float32)
    segments.append((start, num_frames - 1))
    return segments


def _frame_pair_similarity(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = F.normalize(left.float(), dim=-1)
    right = F.normalize(right.float(), dim=-1)
    return (left * right).sum(dim=-1).mean()


def _spatial_merge(
    patch_tokens: torch.Tensor,
    patch_grid_size: tuple[int, int],
    retention_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_frames, num_patches, embed_dim = patch_tokens.shape
    grid_h, grid_w = patch_grid_size
    if num_patches != grid_h * grid_w:
        raise ValueError(f"Patch token count {num_patches} does not match grid {patch_grid_size}")

    target_per_frame = _target_tokens(num_patches, retention_ratio)
    merged_parts = []
    inverse_parts = []
    offset = 0
    for frame_idx in range(num_frames):
        frame_merged, frame_inverse = _feature_merge_frame(
            patch_tokens[frame_idx],
            patch_grid_size,
            target_per_frame,
        )
        merged_parts.append(frame_merged)
        inverse_parts.append(frame_inverse + offset)
        offset += frame_merged.shape[0]
    return torch.cat(merged_parts, dim=0), torch.cat(inverse_parts, dim=0)


def _flashvid_cls_attention_proxy(
    patch_tokens: torch.Tensor,
    cls_token: torch.Tensor | None,
) -> torch.Tensor:
    if cls_token is None:
        return patch_tokens.float().norm(dim=-1)
    return torch.einsum("bnc,bc->bn", F.normalize(patch_tokens.float(), dim=-1), F.normalize(cls_token.float(), dim=-1))


def _gather_flashvid_rope(
    rope_sin: torch.Tensor,
    rope_cos: torch.Tensor,
    patch_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    def gather_one(rope: torch.Tensor) -> torch.Tensor:
        if rope.ndim == 2:
            return rope[patch_indices].unsqueeze(1)
        if rope.ndim == 3:
            gathered = rope[:, patch_indices.reshape(-1), :]
            return gathered.view(rope.shape[0], *patch_indices.shape, rope.shape[-1]).permute(1, 0, 2, 3)
        raise ValueError(f"Unsupported RoPE shape for FlashVID encoder compression: {tuple(rope.shape)}")

    return gather_one(rope_sin), gather_one(rope_cos)


def _flashvid_encoder_compress(
    patch_tokens: torch.Tensor,
    cls_attention: torch.Tensor,
    patch_grid_size: tuple[int, int],
    retention_ratio: float,
    alpha: float,
    expansion: float,
    pool_stride: int,
    temporal_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, num_frames, original_num_patches, embed_dim = patch_tokens.shape
    if original_num_patches != patch_grid_size[0] * patch_grid_size[1]:
        raise ValueError(f"Patch token count {original_num_patches} does not match grid {patch_grid_size}")

    pooled_tokens, pooled_attention, original_to_pool, pool_to_patch = _flashvid_pool_grid_tokens(
        patch_tokens,
        cls_attention,
        patch_grid_size,
        pool_stride,
    )
    _, _, num_patches, _ = pooled_tokens.shape
    if num_patches < 1:
        raise ValueError(f"Patch token count {num_patches} does not match grid {patch_grid_size}")

    token_budget = max(1, min(num_patches, int(math.ceil(num_patches * retention_ratio * expansion))))
    num_adts_tokens = max(0, min(token_budget, int(math.ceil(token_budget * alpha))))
    num_tstm_tokens = token_budget - num_adts_tokens

    compressed_batches = []
    inverse_batches = []
    center_batches = []
    rope_index_batches = []
    for batch_idx in range(batch_size):
        compressed, pooled_inverse, _, pooled_rope_indices = _flashvid_encoder_compress_one(
            pooled_tokens[batch_idx],
            pooled_attention[batch_idx],
            num_adts_tokens,
            num_tstm_tokens,
            temporal_threshold,
        )
        inverse = pooled_inverse[:, original_to_pool]
        centers = torch.gather(
            compressed,
            dim=1,
            index=inverse.unsqueeze(-1).expand(-1, -1, embed_dim),
        )
        rope_indices = pool_to_patch[pooled_rope_indices]
        compressed_batches.append(compressed)
        inverse_batches.append(inverse)
        center_batches.append(centers)
        rope_index_batches.append(rope_indices)

    return (
        torch.stack(compressed_batches, dim=0),
        torch.stack(inverse_batches, dim=0),
        torch.stack(center_batches, dim=0),
        torch.stack(rope_index_batches, dim=0),
    )


def _flashvid_pool_grid_tokens(
    patch_tokens: torch.Tensor,
    cls_attention: torch.Tensor,
    patch_grid_size: tuple[int, int],
    pool_stride: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, num_frames, num_patches, embed_dim = patch_tokens.shape
    grid_h, grid_w = patch_grid_size
    if pool_stride <= 1:
        identity = torch.arange(num_patches, device=patch_tokens.device)
        return patch_tokens, cls_attention, identity, identity
    pooled_h = math.ceil(grid_h / pool_stride)
    pooled_w = math.ceil(grid_w / pool_stride)
    flat = patch_tokens.view(batch_size * num_frames, grid_h, grid_w, embed_dim).permute(0, 3, 1, 2)
    pooled = F.avg_pool2d(flat, kernel_size=pool_stride, stride=pool_stride, ceil_mode=True, count_include_pad=False)
    pooled = pooled.permute(0, 2, 3, 1).reshape(batch_size, num_frames, pooled_h * pooled_w, embed_dim)

    attention = cls_attention.view(batch_size * num_frames, 1, grid_h, grid_w)
    pooled_attention = F.avg_pool2d(
        attention,
        kernel_size=pool_stride,
        stride=pool_stride,
        ceil_mode=True,
        count_include_pad=False,
    )
    pooled_attention = pooled_attention.reshape(batch_size, num_frames, pooled_h * pooled_w)

    y = torch.arange(grid_h, device=patch_tokens.device)
    x = torch.arange(grid_w, device=patch_tokens.device)
    original_to_pool = (
        torch.div(y[:, None], pool_stride, rounding_mode="floor") * pooled_w
        + torch.div(x[None, :], pool_stride, rounding_mode="floor")
    ).reshape(-1)

    pooled_y = torch.arange(pooled_h, device=patch_tokens.device)
    pooled_x = torch.arange(pooled_w, device=patch_tokens.device)
    center_y = (pooled_y * pool_stride + pool_stride // 2).clamp_max(grid_h - 1)
    center_x = (pooled_x * pool_stride + pool_stride // 2).clamp_max(grid_w - 1)
    pool_to_patch = (center_y[:, None] * grid_w + center_x[None, :]).reshape(-1)
    return pooled, pooled_attention, original_to_pool, pool_to_patch


def _flashvid_encoder_compress_one(
    patch_tokens: torch.Tensor,
    cls_attention: torch.Tensor,
    num_adts_tokens: int,
    num_tstm_tokens: int,
    temporal_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_frames, num_patches, embed_dim = patch_tokens.shape
    device = patch_tokens.device
    token_budget = num_adts_tokens + num_tstm_tokens

    if token_budget >= num_patches:
        inverse = torch.arange(num_patches, device=device).expand(num_frames, -1)
        return patch_tokens, inverse, patch_tokens, inverse

    selected_features, selected_indices = _flashvid_adts_v2_select(
        patch_tokens,
        cls_attention,
        num_adts_tokens,
    )
    selected_mask = torch.zeros(num_frames, num_patches, dtype=torch.bool, device=device)
    if num_adts_tokens > 0:
        selected_mask.scatter_(1, selected_indices, True)

    cluster_inverse = _flashvid_tstm_cluster_inverse(
        patch_tokens,
        selected_mask,
        temporal_threshold,
    )
    unique_clusters, compact_inverse = torch.unique(cluster_inverse.reshape(-1), sorted=True, return_inverse=True)
    merged_clusters = _average_by_inverse(patch_tokens.reshape(-1, embed_dim), compact_inverse, unique_clusters.numel())
    compact_inverse = compact_inverse.view(num_frames, num_patches)

    compressed_frames = []
    inverse_frames = []
    center_frames = []
    rope_frames = []
    for frame_idx in range(num_frames):
        frame_selected = selected_indices[frame_idx] if num_adts_tokens > 0 else torch.empty(0, dtype=torch.long, device=device)
        frame_selected_tokens = selected_features[frame_idx] if num_adts_tokens > 0 else patch_tokens.new_empty(0, embed_dim)
        selected_inverse = torch.full((num_patches,), -1, dtype=torch.long, device=device)
        if num_adts_tokens > 0:
            selected_inverse[frame_selected] = torch.arange(num_adts_tokens, device=device)

        other_mask = ~selected_mask[frame_idx]
        other_clusters = compact_inverse[frame_idx, other_mask]
        unique_other_clusters, cluster_to_unique = torch.unique(other_clusters, sorted=True, return_inverse=True)
        if unique_other_clusters.numel() == 0 or num_tstm_tokens == 0:
            other_tokens = patch_tokens.new_empty(0, embed_dim)
            other_inverse = torch.empty(0, dtype=torch.long, device=device)
            other_rope = torch.empty(0, dtype=torch.long, device=device)
        else:
            unique_features = merged_clusters[unique_other_clusters]
            unique_rope = _first_index_for_clusters(
                torch.where(other_mask)[0],
                cluster_to_unique,
                unique_other_clusters.numel(),
            )
            if unique_features.shape[0] > num_tstm_tokens:
                other_tokens, unique_to_kept, other_rope = _flashvid_dpc_reduce(
                    unique_features,
                    unique_rope,
                    num_tstm_tokens,
                )
                other_inverse = unique_to_kept[cluster_to_unique]
            else:
                other_tokens = unique_features
                other_inverse = cluster_to_unique
                other_rope = unique_rope

        frame_tokens = torch.cat([frame_selected_tokens, other_tokens], dim=0)
        frame_rope = torch.cat([frame_selected, other_rope], dim=0)
        if frame_tokens.shape[0] < token_budget:
            pad_count = token_budget - frame_tokens.shape[0]
            pad_token = frame_tokens[-1:] if frame_tokens.numel() else patch_tokens[frame_idx, :1]
            pad_rope = frame_rope[-1:] if frame_rope.numel() else torch.zeros(1, dtype=torch.long, device=device)
            frame_tokens = torch.cat([frame_tokens, pad_token.expand(pad_count, -1)], dim=0)
            frame_rope = torch.cat([frame_rope, pad_rope.expand(pad_count)], dim=0)
        elif frame_tokens.shape[0] > token_budget:
            frame_tokens = frame_tokens[:token_budget]
            frame_rope = frame_rope[:token_budget]

        frame_inverse = selected_inverse
        if other_mask.any() and other_inverse.numel() > 0:
            frame_inverse[other_mask] = num_adts_tokens + other_inverse
        frame_inverse = frame_inverse.clamp_min(0).clamp_max(token_budget - 1)
        compressed_frames.append(frame_tokens)
        inverse_frames.append(frame_inverse)
        center_frames.append(frame_tokens[frame_inverse])
        rope_frames.append(frame_rope)

    return (
        torch.stack(compressed_frames, dim=0),
        torch.stack(inverse_frames, dim=0),
        torch.stack(center_frames, dim=0),
        torch.stack(rope_frames, dim=0),
    )


def _flashvid_adts_v2_select(
    features: torch.Tensor,
    cls_attention: torch.Tensor,
    num_retained_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_frames, num_visual_tokens, feat_dim = features.shape
    if num_retained_tokens <= 0:
        empty_idx = torch.empty(num_frames, 0, dtype=torch.long, device=features.device)
        return features.new_empty(num_frames, 0, feat_dim), empty_idx
    if num_retained_tokens >= num_visual_tokens:
        keep_indices = torch.arange(num_visual_tokens, device=features.device).expand(num_frames, -1)
        return features, keep_indices

    original_features = features
    features_f = features.float()
    pooled_features = features_f.mean(1)
    global_cls_attention = cls_attention.float() * 1e6
    normed = F.normalize(features_f, dim=-1)
    dist_matrix = 1.0 - torch.bmm(normed, normed.transpose(-1, -2))
    calibration_term1 = global_cls_attention.unsqueeze(1)
    local_cls_attention = torch.einsum("b n d, c d -> b c n", features_f, pooled_features).mean(1)
    calibration_term2 = local_cls_attention.unsqueeze(1)
    dist_matrix = dist_matrix * calibration_term1 * calibration_term2

    keep_indices = torch.zeros(num_frames, num_retained_tokens, dtype=torch.long, device=features.device)
    min_dist = torch.topk(dist_matrix, k=2, dim=1, largest=False).values[:, 1, :]
    keep_indices[:, 0] = torch.argmax(min_dist, dim=-1)
    for idx in range(1, num_retained_tokens):
        dist_sub_matrix = torch.gather(
            dist_matrix,
            dim=1,
            index=keep_indices[:, :idx].unsqueeze(-1).expand(-1, -1, num_visual_tokens),
        )
        min_dist = torch.min(dist_sub_matrix, dim=1).values
        min_dist.scatter_(1, keep_indices[:, :idx], -torch.inf)
        keep_indices[:, idx] = torch.argmax(min_dist, dim=-1)

    keep_indices = keep_indices.sort().values
    selected_features = torch.gather(
        original_features,
        dim=1,
        index=keep_indices.unsqueeze(-1).expand(-1, -1, feat_dim),
    )
    return selected_features, keep_indices


def _flashvid_tstm_cluster_inverse(
    patch_tokens: torch.Tensor,
    selected_mask: torch.Tensor,
    temporal_threshold: float,
) -> torch.Tensor:
    num_frames, num_patches, embed_dim = patch_tokens.shape
    device = patch_tokens.device
    cluster_ids = torch.arange(num_frames * num_patches, device=device).view(num_frames, num_patches)
    if num_frames <= 1 or temporal_threshold >= 1.0:
        return cluster_ids

    normed = F.normalize(patch_tokens.float(), dim=-1)
    similarities = torch.bmm(normed[1:], normed[:-1].transpose(1, 2))
    token_mask = ~selected_mask
    similarities[~token_mask[1:].unsqueeze(-1).expand(-1, -1, num_patches)] = -1.0
    similarities[~token_mask[:-1].unsqueeze(1).expand(-1, num_patches, -1)] = -1.0
    best_sim, best_prev = similarities.max(dim=-1)
    for frame_idx in range(1, num_frames):
        merge_mask = best_sim[frame_idx - 1] > temporal_threshold
        if merge_mask.any():
            cluster_ids[frame_idx, merge_mask] = cluster_ids[frame_idx - 1, best_prev[frame_idx - 1, merge_mask]]
    return cluster_ids


def _first_index_for_clusters(indices: torch.Tensor, cluster_inverse: torch.Tensor, num_clusters: int) -> torch.Tensor:
    first = torch.full((num_clusters,), indices.max() + 1 if indices.numel() else 0, dtype=torch.long, device=indices.device)
    first.scatter_reduce_(0, cluster_inverse, indices, reduce="amin", include_self=True)
    return first.clamp_max(indices.max() if indices.numel() else 0)


def _flashvid_dpc_reduce(
    features: torch.Tensor,
    rope_indices: torch.Tensor,
    num_clusters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    token_count, embed_dim = features.shape
    if token_count <= num_clusters:
        inverse = torch.arange(token_count, device=features.device)
        return features, inverse, rope_indices

    distances = torch.cdist(features.float().unsqueeze(0), features.float().unsqueeze(0)).squeeze(0) / math.sqrt(embed_dim)
    knn = min(7, token_count)
    nearest_dist = torch.topk(distances, k=knn, dim=-1, largest=False).values
    density = torch.mean(-(nearest_dist**2), dim=-1).exp()
    higher_density = density[None, :] > density[:, None]
    max_dist = distances.max()
    min_higher_dist = torch.where(higher_density, distances, max_dist).min(dim=-1).values
    score = min_higher_dist * density
    centers = torch.topk(score, k=num_clusters).indices.sort().values
    center_distances = distances[:, centers]
    inverse = torch.argmin(center_distances, dim=-1)
    inverse[centers] = torch.arange(num_clusters, device=features.device)
    reduced = _average_by_inverse(features, inverse, num_clusters)
    return reduced, inverse, rope_indices[centers]


def _protected_spatial_merge(
    patch_tokens: torch.Tensor,
    patch_grid_size: tuple[int, int],
    retention_ratio: float,
    protected_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_frames, num_patches, embed_dim = patch_tokens.shape
    target_per_frame = _target_tokens(num_patches, retention_ratio)
    protected_per_frame = min(num_patches, max(1, int(round(target_per_frame * protected_fraction))))
    spatial_budget = max(target_per_frame - protected_per_frame, 1)
    scores = _token_importance_scores(patch_tokens)

    merged_parts = []
    inverse_parts = []
    offset = 0
    for frame_idx in range(num_frames):
        frame_tokens = patch_tokens[frame_idx]
        protected = torch.topk(scores[frame_idx], k=protected_per_frame, largest=True).indices
        protected_mask = torch.zeros(num_patches, dtype=torch.bool, device=patch_tokens.device)
        protected_mask[protected] = True

        inverse_frame = torch.empty(num_patches, dtype=torch.long, device=patch_tokens.device)
        sorted_protected = protected.sort().values
        inverse_frame[sorted_protected] = torch.arange(sorted_protected.numel(), device=patch_tokens.device)
        protected_tokens = frame_tokens[sorted_protected]

        unprotected = torch.where(~protected_mask)[0]
        if unprotected.numel() > 0:
            keep_h, keep_w = _target_grid_for_budget(*patch_grid_size, spatial_budget)
            coarse = _coarse_grid_inverse(*patch_grid_size, keep_h, keep_w, patch_tokens.device)[unprotected]
            unique_coarse, compact = torch.unique(coarse, sorted=True, return_inverse=True)
            spatial_tokens = _average_by_inverse(frame_tokens[unprotected], compact, unique_coarse.numel())
            inverse_frame[unprotected] = compact + protected_tokens.shape[0]
            frame_merged = torch.cat([protected_tokens, spatial_tokens], dim=0)
        else:
            frame_merged = protected_tokens

        merged_parts.append(frame_merged)
        inverse_parts.append(inverse_frame + offset)
        offset += frame_merged.shape[0]

    return torch.cat(merged_parts, dim=0), torch.cat(inverse_parts, dim=0)


def _tstm_merge(
    patch_tokens: torch.Tensor,
    patch_grid_size: tuple[int, int],
    retention_ratio: float,
    protected_fraction: float,
    temporal_threshold: float,
    neighbor_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_frames, num_patches, embed_dim = patch_tokens.shape
    grid_h, grid_w = patch_grid_size
    target_per_frame = _target_tokens(num_patches, retention_ratio)
    protected_per_frame = min(num_patches, max(1, int(round(target_per_frame * protected_fraction))))
    scores = _token_importance_scores(patch_tokens)

    cluster_ids = torch.empty(num_frames, num_patches, dtype=torch.long, device=patch_tokens.device)
    protected_masks = []
    for frame_idx in range(num_frames):
        protected = torch.topk(scores[frame_idx], k=protected_per_frame, largest=True).indices
        protected_mask = torch.zeros(num_patches, dtype=torch.bool, device=patch_tokens.device)
        protected_mask[protected] = True
        protected_masks.append(protected_mask)

    cluster_ids[0] = torch.arange(num_patches, device=patch_tokens.device)
    next_cluster = num_patches
    normalized = F.normalize(patch_tokens.float(), dim=-1)

    for frame_idx in range(1, num_frames):
        if neighbor_size == 0:
            similarities = normalized[frame_idx] @ normalized[frame_idx - 1].transpose(0, 1)
            best_sim, best_prev = similarities.max(dim=-1)
        else:
            prev_candidates = _local_previous_candidates(normalized[frame_idx - 1], grid_h, grid_w, neighbor_size)
            current = normalized[frame_idx].view(num_patches, 1, embed_dim)
            similarities = (current * prev_candidates).sum(dim=-1)
            best_sim, best_local = similarities.max(dim=-1)
            local_indices = _local_candidate_indices(grid_h, grid_w, neighbor_size, patch_tokens.device)
            best_prev = local_indices[torch.arange(num_patches, device=patch_tokens.device), best_local]
        merge_mask = (
            (best_sim > temporal_threshold)
            & ~protected_masks[frame_idx]
            & ~protected_masks[frame_idx - 1][best_prev]
        )
        new_ids = torch.arange(next_cluster, next_cluster + num_patches, device=patch_tokens.device)
        cluster_ids[frame_idx] = new_ids
        cluster_ids[frame_idx, merge_mask] = cluster_ids[frame_idx - 1, best_prev[merge_mask]]
        next_cluster += num_patches

    inverse = cluster_ids.reshape(-1)
    unique_ids, compact_inverse = torch.unique(inverse, sorted=True, return_inverse=True)
    merged = _average_by_inverse(patch_tokens.reshape(-1, embed_dim), compact_inverse, unique_ids.numel())
    target_total = max(1, num_frames * _target_tokens(num_patches, retention_ratio))
    if merged.shape[0] > target_total:
        merged, compact_inverse = _coarsen_temporal_clusters(
            patch_tokens,
            compact_inverse,
            merged,
            patch_grid_size,
            retention_ratio,
        )
    return merged, compact_inverse


def _target_tokens(num_tokens: int, retention_ratio: float) -> int:
    return max(1, min(num_tokens, int(round(num_tokens * retention_ratio))))


def _retention_from_merge_ratio(merge_ratio: float) -> float:
    return max(1e-6, min(1.0, 1.0 - merge_ratio))


def _parse_layer_ratio_schedule(schedule: str, depth: int) -> dict[int, float]:
    if not schedule:
        return {}
    ratios: dict[int, float] = {}
    for raw_part in schedule.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid token merging layer ratio segment {part!r}; expected layers:ratio")
        layer_spec, ratio_spec = part.split(":", 1)
        ratio = float(ratio_spec)
        if ratio < 0.0 or ratio >= 1.0:
            raise ValueError("Layer-wise token merging ratios must be in [0, 1)")
        layer_spec = layer_spec.strip()
        if "-" in layer_spec:
            start_s, end_s = layer_spec.split("-", 1)
            start_layer = int(start_s)
            end_layer = int(end_s)
        else:
            start_layer = end_layer = int(layer_spec)
        if start_layer < 1 or end_layer > depth or start_layer > end_layer:
            raise ValueError(f"Layer range {layer_spec!r} is outside valid 1-{depth}")
        for layer in range(start_layer, end_layer + 1):
            ratios[layer - 1] = ratio
    return ratios


def _target_grid(grid_h: int, grid_w: int, retention_ratio: float) -> tuple[int, int]:
    return _target_grid_for_budget(grid_h, grid_w, _target_tokens(grid_h * grid_w, retention_ratio))


def _target_grid_for_budget(grid_h: int, grid_w: int, budget: int) -> tuple[int, int]:
    ratio = max(1.0 / (grid_h * grid_w), min(1.0, budget / (grid_h * grid_w)))
    scale = ratio**0.5
    return max(1, min(grid_h, int(round(grid_h * scale)))), max(1, min(grid_w, int(round(grid_w * scale))))


def _coarse_grid_inverse(
    grid_h: int,
    grid_w: int,
    keep_h: int,
    keep_w: int,
    device: torch.device,
) -> torch.Tensor:
    y = torch.arange(grid_h, device=device)
    x = torch.arange(grid_w, device=device)
    coarse_y = torch.div(y * keep_h, grid_h, rounding_mode="floor").clamp_max(keep_h - 1)
    coarse_x = torch.div(x * keep_w, grid_w, rounding_mode="floor").clamp_max(keep_w - 1)
    return (coarse_y[:, None] * keep_w + coarse_x[None, :]).reshape(-1)


def _average_by_inverse(tokens: torch.Tensor, inverse: torch.Tensor, num_clusters: int) -> torch.Tensor:
    merged = torch.zeros(num_clusters, tokens.shape[-1], dtype=tokens.dtype, device=tokens.device)
    merged.scatter_add_(0, inverse[:, None].expand(-1, tokens.shape[-1]), tokens)
    counts = torch.bincount(inverse, minlength=num_clusters).clamp_min(1).to(tokens.dtype)
    return merged / counts[:, None]


def _feature_merge_frame(
    frame_tokens: torch.Tensor,
    patch_grid_size: tuple[int, int],
    target_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_patches, embed_dim = frame_tokens.shape
    grid_h, grid_w = patch_grid_size
    keep_h, keep_w = _target_grid_for_budget(grid_h, grid_w, target_tokens)
    coarse_ids = _coarse_grid_inverse(grid_h, grid_w, keep_h, keep_w, frame_tokens.device)
    scores = frame_tokens.float().norm(dim=-1)

    center_indices = []
    for coarse_id in range(keep_h * keep_w):
        candidates = torch.where(coarse_ids == coarse_id)[0]
        if candidates.numel() == 0:
            continue
        best = candidates[torch.argmax(scores[candidates])]
        center_indices.append(best)

    centers = torch.stack(center_indices)
    if centers.numel() > target_tokens:
        center_scores = scores[centers]
        centers = centers[torch.topk(center_scores, k=target_tokens, largest=True).indices]
    elif centers.numel() < target_tokens:
        selected = torch.zeros(num_patches, dtype=torch.bool, device=frame_tokens.device)
        selected[centers] = True
        extra = torch.topk(scores.masked_fill(selected, -torch.inf), k=target_tokens - centers.numel()).indices
        centers = torch.cat([centers, extra], dim=0)

    centers = centers.sort().values
    normed_tokens = F.normalize(frame_tokens.float(), dim=-1)
    normed_centers = normed_tokens[centers]
    inverse = torch.argmax(normed_tokens @ normed_centers.transpose(0, 1), dim=-1)
    inverse[centers] = torch.arange(centers.numel(), device=frame_tokens.device)
    return _average_by_inverse(frame_tokens, inverse, centers.numel()), inverse


def _token_importance_scores(patch_tokens: torch.Tensor) -> torch.Tensor:
    scores = patch_tokens.float().norm(dim=-1)
    if patch_tokens.shape[0] > 1:
        temporal_change = torch.zeros_like(scores)
        temporal_change[1:] = (patch_tokens[1:].float() - patch_tokens[:-1].float()).norm(dim=-1)
        scores = scores + temporal_change
    return scores


def _coarsen_temporal_clusters(
    patch_tokens: torch.Tensor,
    inverse: torch.Tensor,
    merged_tokens: torch.Tensor,
    patch_grid_size: tuple[int, int],
    retention_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_frames, num_patches, embed_dim = patch_tokens.shape
    grid_h, grid_w = patch_grid_size
    keep_h, keep_w = _target_grid(grid_h, grid_w, retention_ratio)
    num_clusters = merged_tokens.shape[0]
    device = patch_tokens.device

    frame_ids = torch.arange(num_frames, device=device).repeat_interleave(num_patches).to(merged_tokens.dtype)
    patch_ids = torch.arange(num_patches, device=device).repeat(num_frames)
    y_ids = torch.div(patch_ids, grid_w, rounding_mode="floor").to(merged_tokens.dtype)
    x_ids = (patch_ids % grid_w).to(merged_tokens.dtype)

    cluster_counts = torch.bincount(inverse, minlength=num_clusters).clamp_min(1).to(merged_tokens.dtype)
    cluster_frames = torch.zeros(num_clusters, dtype=merged_tokens.dtype, device=device)
    cluster_y = torch.zeros_like(cluster_frames)
    cluster_x = torch.zeros_like(cluster_frames)
    cluster_frames.scatter_add_(0, inverse, frame_ids)
    cluster_y.scatter_add_(0, inverse, y_ids)
    cluster_x.scatter_add_(0, inverse, x_ids)
    cluster_frames = torch.round(cluster_frames / cluster_counts).long().clamp_(0, num_frames - 1)
    cluster_y = torch.div((cluster_y / cluster_counts).long() * keep_h, grid_h, rounding_mode="floor").clamp_(0, keep_h - 1)
    cluster_x = torch.div((cluster_x / cluster_counts).long() * keep_w, grid_w, rounding_mode="floor").clamp_(0, keep_w - 1)

    coarse_ids = cluster_frames * (keep_h * keep_w) + cluster_y * keep_w + cluster_x
    _, cluster_to_coarse = torch.unique(coarse_ids, sorted=True, return_inverse=True)
    coarsened = _average_by_inverse(merged_tokens, cluster_to_coarse, int(cluster_to_coarse.max().item()) + 1)
    return coarsened, cluster_to_coarse[inverse]


def _local_previous_candidates(
    prev_tokens: torch.Tensor,
    grid_h: int,
    grid_w: int,
    neighbor_size: int,
) -> torch.Tensor:
    embed_dim = prev_tokens.shape[-1]
    prev_grid = prev_tokens.transpose(0, 1).view(1, embed_dim, grid_h, grid_w)
    unfolded = F.unfold(prev_grid, kernel_size=neighbor_size, padding=neighbor_size // 2)
    neighbor_count = neighbor_size * neighbor_size
    return unfolded.view(embed_dim, neighbor_count, grid_h * grid_w).permute(2, 1, 0)


def _local_candidate_indices(
    grid_h: int,
    grid_w: int,
    neighbor_size: int,
    device: torch.device,
) -> torch.Tensor:
    index_grid = torch.arange(grid_h * grid_w, device=device, dtype=torch.float32).view(1, 1, grid_h, grid_w)
    neighbor_count = neighbor_size * neighbor_size
    unfolded = F.unfold(index_grid, kernel_size=neighbor_size, padding=neighbor_size // 2)
    unfolded = unfolded.view(neighbor_count, grid_h * grid_w).transpose(0, 1)
    return unfolded.long().clamp_(0, grid_h * grid_w - 1)


def _build_patch_embed(patch_size: int, embed_dim: int) -> DinoVisionTransformer:
    model = DinoVisionTransformer(
        img_size=224,
        patch_size=patch_size,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="max",
        pos_embed_rope_dtype="fp32",
        embed_dim=embed_dim,
        depth=24,
        num_heads=16,
        ffn_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-5,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
    )
    model.init_weights()
    return model


def slice_expand_and_flatten(token_tensor: torch.Tensor, batch_size: int, num_frames: int) -> torch.Tensor:
    first_frame_token = token_tensor[:, 0:1].expand(batch_size, 1, *token_tensor.shape[2:])
    other_frame_tokens = token_tensor[:, 1:].expand(batch_size, num_frames - 1, *token_tensor.shape[2:])
    tokens = torch.cat([first_frame_token, other_frame_tokens], dim=1)
    return tokens.view(batch_size * num_frames, *tokens.shape[2:])
