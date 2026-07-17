# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import torch
import torch.nn as nn

from vggt_omega.models.aggregator import Aggregator
from vggt_omega.models.heads import CameraHead, DenseHead, TextAlignmentHead


class VGGTOmega(nn.Module):
    """Minimal VGGT-Omega inference model for camera and depth prediction."""

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 1024,
        enable_camera: bool = True,
        enable_depth: bool = True,
        enable_alignment: bool = False,
        register_attention_block_indices: list[int] | None = None,
        enable_token_merging: bool = False,
        token_merging_start: int = 0,
        token_merging_ratio: float = 0.9,
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
        **unused_kwargs,
    ) -> None:
        super().__init__()
        if unused_kwargs:
            warnings.warn(
                f"Ignoring unsupported VGGTOmega options: {sorted(unused_kwargs)}",
                stacklevel=2,
            )

        self.aggregator = Aggregator(
            patch_size=patch_size,
            embed_dim=embed_dim,
            register_attention_block_indices=register_attention_block_indices,
            enable_token_merging=enable_token_merging,
            token_merging_start=token_merging_start,
            token_merging_ratio=token_merging_ratio,
            token_merging_method=token_merging_method,
            token_merging_tstm_threshold=token_merging_tstm_threshold,
            token_merging_tstm_neighbor_size=token_merging_tstm_neighbor_size,
            token_merging_flashvid_alpha=token_merging_flashvid_alpha,
            token_merging_flashvid_expansion=token_merging_flashvid_expansion,
            token_merging_flashvid_pool_stride=token_merging_flashvid_pool_stride,
            token_merging_frame_restore_layer=token_merging_frame_restore_layer,
            token_merging_frame_alpha=token_merging_frame_alpha,
            token_merging_frame_segment_threshold=token_merging_frame_segment_threshold,
            token_merging_frame_merge_threshold=token_merging_frame_merge_threshold,
            token_merging_frame_max_window=token_merging_frame_max_window,
            token_merging_frame_pool_stride=token_merging_frame_pool_stride,
            token_merging_frame_multi_max_group_size=token_merging_frame_multi_max_group_size,
            token_merging_frame_multi_pair_threshold=token_merging_frame_multi_pair_threshold,
            token_merging_frame_multi_span_threshold=token_merging_frame_multi_span_threshold,
            token_merging_frame_group_strategy=token_merging_frame_group_strategy,
        )
        _warn_if_rope_not_max(self.aggregator)
        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.dense_head = DenseHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_depth else None
        self.text_alignment_head = TextAlignmentHead(dim_in=2 * embed_dim) if enable_alignment else None

    def forward(self, images: torch.Tensor, use_amp: bool = True) -> dict[str, torch.Tensor]:
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp and images.is_cuda):
            aggregated_tokens_list, patch_token_start = self.aggregator(images)

        final_tokens = aggregated_tokens_list[-1]
        if final_tokens is None:
            raise ValueError("Aggregator did not cache the final layer, which VGGTOmega needs.")

        predictions = {
            "camera_and_register_tokens": final_tokens[:, :, :patch_token_start].contiguous(),
        }
        with torch.autocast(device_type="cuda", enabled=False):
            if self.camera_head is not None:
                predictions["pose_enc"] = self.camera_head(
                    aggregated_tokens_list,
                    patch_token_start=patch_token_start,
                )

            if self.dense_head is not None:
                depth, depth_conf = self.dense_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_token_start=patch_token_start,
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.text_alignment_head is not None:
                predictions.update(
                    self.text_alignment_head(
                        aggregated_tokens_list,
                        patch_token_start=patch_token_start,
                    )
                )

        frame_merge_stats = getattr(self.aggregator, "last_frame_merge_stats", None)
        if frame_merge_stats:
            predictions["frame_merge_stats"] = frame_merge_stats
        token_merging_stats = getattr(self.aggregator, "last_token_merging_stats", None)
        if token_merging_stats:
            predictions["token_merging_stats"] = token_merging_stats

        if not self.training:
            predictions["images"] = images
        return predictions


def _warn_if_rope_not_max(aggregator: nn.Module) -> None:
    for name, module in (("aggregator.patch_embed", aggregator.patch_embed), ("aggregator", aggregator)):
        rope_embed = getattr(module, "rope_embed", None)
        normalize_coords = getattr(rope_embed, "normalize_coords", None)
        if normalize_coords != "max":
            warnings.warn(
                f"{name} RoPE normalize_coords is {normalize_coords!r}; "
                "the released VGGT-Omega checkpoint was trained with 'max'.",
                stacklevel=2,
            )
