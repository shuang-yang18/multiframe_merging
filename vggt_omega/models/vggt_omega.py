# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from vggt_omega.models.aggregator import Aggregator
from vggt_omega.models.heads import CameraHead, DenseHead, TextAlignmentHead
from vggt_omega.utils.rotation import mat_to_quat, quat_to_mat


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
        token_merging_layer_ratios: str = "",
        token_merging_method: str = "spatial",
        token_merging_flashvid_alpha: float = 0.7,
        token_merging_flashvid_expansion: float = 1.25,
        token_merging_flashvid_pool_stride: int = 2,
        token_merging_flashvid_tstm_threshold: float = 0.8,
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
        token_merging_frame_anchor_count: int = 4,
        token_merging_frame_anchor_selection: str = "uniform",
        token_merging_frame_adaptive_boundary_z: float = 2.5,
        token_merging_frame_adaptive_medoid_z: float = 1.5,
        token_merging_frame_patch_fusion_quantile: float = 0.75,
        token_merging_frame_special_cross_attention: bool = False,
        token_merging_frame_special_cross_attention_alpha: float = 0.1,
        token_merging_segment_bank_pair_threshold: float = 0.986,
        token_merging_segment_bank_span_threshold: float = 0.948,
        token_merging_segment_bank_max_group_size: int = 4,
        omega_accelerator: str = "none",
        sparse_vggt_sparse_ratio: float | None = 0.5,
        sparse_vggt_cdf_threshold: float | None = None,
        sparse_vggt_pool_mode: str = "avg",
        da_vggt_max_frames: int = 0,
        da_vggt_sampling_method: str = "fl_maxmin",
        da_vggt_n_anchors: int = 1,
        da_vggt_dino_batch_size: int = 256,
        da_vggt_lambda_div: float = 0.0,
        dynamic_fastvggt_schedule: str = "all",
        skip_global_attention_blocks: str = "",
        skip_inter_frame_attention_blocks: str = "",
        frame_only_inter_frame_blocks: str = "",
        enable_adaptive_frame_token_fusion: bool = False,
        adaptive_frame_representation: str = "global_pool",
        adaptive_representation_pca_dim: int = 512,
        adaptive_representation_clusters: int = 3,
        adaptive_spatial_grid: int = 4,
        adaptive_grouping: str = "serial",
        adaptive_reference_selection: str = "first",
        adaptive_reference_participates: bool = True,
        adaptive_group_similarity_threshold: float = 0.98,
        adaptive_group_max_size: int = 4,
        adaptive_parallel_window: int = 10,
        adaptive_update_policy: str = "initial_only",
        adaptive_update_after_blocks: str = "9,17",
        adaptive_frame_fusion: str = "direct",
        adaptive_frame_fusion_weighting: str = "similarity",
        adaptive_frame_token_similarity_threshold: float = 0.95,
        adaptive_token_merging: str = "fast_bipartite",
        adaptive_token_keep_ratio: float = 0.1,
        adaptive_token_clusters: int = 4,
        adaptive_token_cluster_budget: str = "proportional",
        adaptive_token_kmeans_iterations: int = 12,
        **unused_kwargs,
    ) -> None:
        super().__init__()
        if omega_accelerator not in {"none", "da_vggt", "sparse_vggt"}:
            raise ValueError("omega_accelerator must be 'none', 'da_vggt', or 'sparse_vggt'")
        if unused_kwargs:
            warnings.warn(
                f"Ignoring unsupported VGGTOmega options: {sorted(unused_kwargs)}",
                stacklevel=2,
            )
        self.omega_accelerator = omega_accelerator

        self.aggregator = Aggregator(
            patch_size=patch_size,
            embed_dim=embed_dim,
            register_attention_block_indices=register_attention_block_indices,
            enable_token_merging=enable_token_merging,
            token_merging_start=token_merging_start,
            token_merging_ratio=token_merging_ratio,
            token_merging_layer_ratios=token_merging_layer_ratios,
            token_merging_method=token_merging_method,
            token_merging_flashvid_alpha=token_merging_flashvid_alpha,
            token_merging_flashvid_expansion=token_merging_flashvid_expansion,
            token_merging_flashvid_pool_stride=token_merging_flashvid_pool_stride,
            token_merging_flashvid_tstm_threshold=token_merging_flashvid_tstm_threshold,
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
            token_merging_frame_protect_period=token_merging_frame_protect_period,
            token_merging_frame_protect_prefix=token_merging_frame_protect_prefix,
            token_merging_frame_anchor_count=token_merging_frame_anchor_count,
            token_merging_frame_anchor_selection=token_merging_frame_anchor_selection,
            token_merging_frame_adaptive_boundary_z=token_merging_frame_adaptive_boundary_z,
            token_merging_frame_adaptive_medoid_z=token_merging_frame_adaptive_medoid_z,
            token_merging_frame_patch_fusion_quantile=token_merging_frame_patch_fusion_quantile,
            token_merging_frame_special_cross_attention=token_merging_frame_special_cross_attention,
            token_merging_frame_special_cross_attention_alpha=token_merging_frame_special_cross_attention_alpha,
            token_merging_segment_bank_pair_threshold=token_merging_segment_bank_pair_threshold,
            token_merging_segment_bank_span_threshold=token_merging_segment_bank_span_threshold,
            token_merging_segment_bank_max_group_size=token_merging_segment_bank_max_group_size,
            enable_sparse_vggt=omega_accelerator == "sparse_vggt",
            sparse_vggt_sparse_ratio=sparse_vggt_sparse_ratio,
            sparse_vggt_cdf_threshold=sparse_vggt_cdf_threshold,
            sparse_vggt_pool_mode=sparse_vggt_pool_mode,
            enable_da_vggt=omega_accelerator == "da_vggt",
            da_vggt_max_frames=da_vggt_max_frames,
            da_vggt_sampling_method=da_vggt_sampling_method,
            da_vggt_n_anchors=da_vggt_n_anchors,
            da_vggt_dino_batch_size=da_vggt_dino_batch_size,
            da_vggt_lambda_div=da_vggt_lambda_div,
            dynamic_fastvggt_schedule=dynamic_fastvggt_schedule,
            skip_global_attention_blocks=skip_global_attention_blocks,
            skip_inter_frame_attention_blocks=skip_inter_frame_attention_blocks,
            frame_only_inter_frame_blocks=frame_only_inter_frame_blocks,
            enable_adaptive_frame_token_fusion=enable_adaptive_frame_token_fusion,
            adaptive_frame_representation=adaptive_frame_representation,
            adaptive_representation_pca_dim=adaptive_representation_pca_dim,
            adaptive_representation_clusters=adaptive_representation_clusters,
            adaptive_spatial_grid=adaptive_spatial_grid,
            adaptive_grouping=adaptive_grouping,
            adaptive_reference_selection=adaptive_reference_selection,
            adaptive_reference_participates=adaptive_reference_participates,
            adaptive_group_similarity_threshold=adaptive_group_similarity_threshold,
            adaptive_group_max_size=adaptive_group_max_size,
            adaptive_parallel_window=adaptive_parallel_window,
            adaptive_update_policy=adaptive_update_policy,
            adaptive_update_after_blocks=adaptive_update_after_blocks,
            adaptive_frame_fusion=adaptive_frame_fusion,
            adaptive_frame_fusion_weighting=adaptive_frame_fusion_weighting,
            adaptive_frame_token_similarity_threshold=adaptive_frame_token_similarity_threshold,
            adaptive_token_merging=adaptive_token_merging,
            adaptive_token_keep_ratio=adaptive_token_keep_ratio,
            adaptive_token_clusters=adaptive_token_clusters,
            adaptive_token_cluster_budget=adaptive_token_cluster_budget,
            adaptive_token_kmeans_iterations=adaptive_token_kmeans_iterations,
        )
        _warn_if_rope_not_max(self.aggregator)
        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.dense_head = DenseHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_depth else None
        self.text_alignment_head = TextAlignmentHead(dim_in=2 * embed_dim) if enable_alignment else None

    def forward(self, images: torch.Tensor, use_amp: bool = True, **unused_kwargs) -> dict[str, torch.Tensor]:
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        if self.omega_accelerator == "da_vggt" and self.aggregator.da_vggt_max_frames > 0:
            if images.shape[1] > self.aggregator.da_vggt_max_frames:
                return self._forward_da_vggt(images, use_amp=use_amp)

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
        adaptive_fusion_stats = getattr(self.aggregator, "last_adaptive_frame_token_fusion_stats", None)
        if adaptive_fusion_stats:
            predictions["adaptive_frame_token_fusion_stats"] = adaptive_fusion_stats
        segment_patch_bank_stats = getattr(self.aggregator, "last_segment_patch_bank_stats", None)
        if segment_patch_bank_stats:
            predictions["segment_patch_bank_stats"] = segment_patch_bank_stats
        frame_special_cross_attention_stats = getattr(
            self.aggregator, "last_frame_special_cross_attention_stats", None
        )
        if frame_special_cross_attention_stats:
            predictions["frame_special_cross_attention_stats"] = frame_special_cross_attention_stats
        if self.aggregator.enable_sparse_vggt:
            sparse_vggt_stats = getattr(self.aggregator, "last_sparse_vggt_stats", None)
            if sparse_vggt_stats:
                predictions["sparse_vggt_stats"] = sparse_vggt_stats

        if not self.training:
            predictions["images"] = images
        return predictions

    def _forward_da_vggt(self, images: torch.Tensor, use_amp: bool = True) -> dict[str, torch.Tensor]:
        import time

        batch_size, num_frames, _, height, width = images.shape
        if batch_size != 1:
            raise ValueError("DA-VGGT accelerator currently supports batch size 1.")
        if self.camera_head is None:
            raise ValueError("DA-VGGT accelerator needs camera_head for chunk alignment.")

        device = images.device
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        timing = {"accelerator": "da_vggt"}
        self.aggregator.last_frame_merge_stats = []
        self.aggregator.last_token_merging_stats = []
        self.aggregator.last_sparse_vggt_stats = []

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp and images.is_cuda):
            patch_tokens_cpu, pooled_tokens = self.aggregator.forward_dino(images)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timing["da_vggt_dino"] = time.perf_counter() - start

        start = time.perf_counter()
        chunks, anchors = _da_vggt_make_chunks(
            pooled_tokens,
            self.aggregator.da_vggt_max_frames,
            self.aggregator.da_vggt_sampling_method,
            self.aggregator.da_vggt_n_anchors,
            self.aggregator.da_vggt_lambda_div,
        )
        timing["da_vggt_sampling"] = time.perf_counter() - start
        timing["da_vggt_num_chunks"] = len(chunks)
        timing["da_vggt_chunk_size"] = self.aggregator.da_vggt_max_frames
        timing["da_vggt_sampling_method"] = self.aggregator.da_vggt_sampling_method

        chunk_pose_encs = []
        chunk_frame_indices = []
        chunk_depths = [] if self.dense_head is not None else None
        chunk_depth_confs = [] if self.dense_head is not None else None
        chunk_times = []
        patch_token_start = None

        start_all_chunks = time.perf_counter()
        for chunk_indices in chunks:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start_chunk = time.perf_counter()
            index = torch.tensor(chunk_indices, dtype=torch.long)
            patch_tokens = patch_tokens_cpu[index].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp and images.is_cuda):
                aggregated_tokens_list, patch_token_start = self.aggregator.forward_transformer(
                    patch_tokens,
                    batch_size=1,
                    num_frames=len(chunk_indices),
                    height=height,
                    width=width,
                    device=device,
                )
            with torch.autocast(device_type="cuda", enabled=False):
                pose_enc = self.camera_head(aggregated_tokens_list, patch_token_start=patch_token_start)
                if self.dense_head is not None:
                    chunk_images = images[:, chunk_indices]
                    depth, depth_conf = self.dense_head(
                        aggregated_tokens_list,
                        images=chunk_images,
                        patch_token_start=patch_token_start,
                    )
                    chunk_depths.append(depth.detach().cpu())
                    chunk_depth_confs.append(depth_conf.detach().cpu())
            chunk_pose_encs.append(pose_enc.detach().cpu())
            chunk_frame_indices.append(list(chunk_indices))
            del patch_tokens, aggregated_tokens_list, pose_enc
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize(device)
            chunk_times.append(time.perf_counter() - start_chunk)
        timing["da_vggt_transformer_total"] = time.perf_counter() - start_all_chunks
        timing["da_vggt_transformer_per_chunk"] = chunk_times

        pose_enc = _da_vggt_align_poses(chunk_pose_encs, chunk_frame_indices, anchors, num_frames, device)
        predictions = {
            "pose_enc": pose_enc.unsqueeze(0),
            "da_vggt_stats": {
                **timing,
                "chunk_frame_indices": chunk_frame_indices,
                "anchors": anchors,
            },
        }
        token_merging_stats = getattr(self.aggregator, "last_token_merging_stats", None)
        if token_merging_stats:
            predictions["token_merging_stats"] = token_merging_stats
        frame_merge_stats = getattr(self.aggregator, "last_frame_merge_stats", None)
        if frame_merge_stats:
            predictions["frame_merge_stats"] = frame_merge_stats

        if self.dense_head is not None and chunk_depths is not None and chunk_depth_confs is not None:
            depth, depth_conf, depth_scales = _da_vggt_align_depths(
                chunk_depths,
                chunk_depth_confs,
                chunk_frame_indices,
                anchors,
                num_frames,
                device,
            )
            predictions["depth"] = depth.unsqueeze(0)
            predictions["depth_conf"] = depth_conf.unsqueeze(0)
            predictions["da_vggt_stats"]["depth_chunk_scales"] = depth_scales

        if not self.training:
            predictions["images"] = images
        return predictions


def _pose_enc_to_se3(pose_enc: torch.Tensor) -> torch.Tensor:
    translation = pose_enc[..., :3]
    quat = F.normalize(pose_enc[..., 3:7], dim=-1)
    rotation = quat_to_mat(quat)
    transform = torch.zeros(*pose_enc.shape[:-1], 4, 4, device=pose_enc.device, dtype=pose_enc.dtype)
    transform[..., :3, :3] = rotation
    transform[..., :3, 3] = translation
    transform[..., 3, 3] = 1.0
    return transform


def _se3_to_pose_enc(transform: torch.Tensor, focal: torch.Tensor) -> torch.Tensor:
    translation = transform[..., :3, 3]
    quat = mat_to_quat(transform[..., :3, :3])
    return torch.cat([translation, quat, focal], dim=-1)


def _procrustes_se3(src_se3_list: list[torch.Tensor], dst_se3_list: list[torch.Tensor]) -> torch.Tensor:
    src_points = []
    dst_points = []
    for src, dst in zip(src_se3_list, dst_se3_list):
        src_points.extend([src[:3, 3], src[:3, 3] + src[:3, 2]])
        dst_points.extend([dst[:3, 3], dst[:3, 3] + dst[:3, 2]])
    src = torch.stack(src_points).float()
    dst = torch.stack(dst_points).float()
    src_center = src.mean(dim=0)
    dst_center = dst.mean(dim=0)
    u, _, vt = torch.linalg.svd((src - src_center).T @ (dst - dst_center))
    sign = torch.ones(3, device=src.device, dtype=src.dtype)
    sign[2] = 1.0 if torch.det(vt.T @ u.T) >= 0 else -1.0
    rotation = vt.T @ torch.diag(sign) @ u.T
    transform = torch.eye(4, device=src.device, dtype=src.dtype)
    transform[:3, :3] = rotation
    transform[:3, 3] = dst_center - rotation @ src_center
    return transform


def _da_vggt_align_poses(
    chunk_pose_encs: list[torch.Tensor],
    chunk_frame_indices: list[list[int]],
    anchors: list[int],
    num_frames: int,
    device: torch.device,
) -> torch.Tensor:
    all_pose_encs = torch.zeros(num_frames, 9, device=device)
    anchor_set = set(anchors)
    if len(anchors) <= 1:
        ref_pose = chunk_pose_encs[0][0, 0].to(device).float()
        t_ref = _pose_enc_to_se3(ref_pose)
        for chunk_idx, pose_chunk_cpu in enumerate(chunk_pose_encs):
            pose_chunk = pose_chunk_cpu[0].to(device).float()
            indices = chunk_frame_indices[chunk_idx]
            if chunk_idx == 0:
                transforms = _pose_enc_to_se3(pose_chunk)
                all_pose_encs[indices] = _se3_to_pose_enc(transforms, pose_chunk[:, 7:9])
            else:
                transform = t_ref @ torch.linalg.inv(_pose_enc_to_se3(pose_chunk[0]))
                rest = pose_chunk[1:]
                if len(rest) > 0:
                    all_pose_encs[indices[1:]] = _se3_to_pose_enc(transform @ _pose_enc_to_se3(rest), rest[:, 7:9])
        return all_pose_encs

    ref_anchor_se3 = None
    for chunk_idx, pose_chunk_cpu in enumerate(chunk_pose_encs):
        pose_chunk = pose_chunk_cpu[0].to(device).float()
        indices = chunk_frame_indices[chunk_idx]
        anchor_positions = [indices.index(anchor) for anchor in anchors]
        chunk_anchor_se3 = list(_pose_enc_to_se3(pose_chunk[anchor_positions]).unbind(0))
        if chunk_idx == 0:
            ref_anchor_se3 = chunk_anchor_se3
            transforms = _pose_enc_to_se3(pose_chunk)
            all_pose_encs[indices] = _se3_to_pose_enc(transforms, pose_chunk[:, 7:9])
            continue
        transform = _procrustes_se3(chunk_anchor_se3, ref_anchor_se3)
        non_anchor_positions = [idx for idx, frame_idx in enumerate(indices) if frame_idx not in anchor_set]
        if non_anchor_positions:
            rest = pose_chunk[non_anchor_positions]
            rest_indices = [indices[idx] for idx in non_anchor_positions]
            all_pose_encs[rest_indices] = _se3_to_pose_enc(transform @ _pose_enc_to_se3(rest), rest[:, 7:9])
    return all_pose_encs


def _da_vggt_align_depths(
    chunk_depths: list[torch.Tensor],
    chunk_depth_confs: list[torch.Tensor],
    chunk_frame_indices: list[list[int]],
    anchors: list[int],
    num_frames: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    ref_depth = chunk_depths[0][0, 0].to(device)
    height, width = ref_depth.shape[:2]
    all_depth = torch.zeros(num_frames, height, width, 1, device=device)
    all_depth_conf = torch.zeros(num_frames, height, width, device=device)
    anchor_set = set(anchors)
    primary_anchor = anchors[0]
    depth_scales = []
    for chunk_idx, depth_cpu in enumerate(chunk_depths):
        depth = depth_cpu[0].to(device)
        depth_conf = chunk_depth_confs[chunk_idx][0].to(device)
        indices = chunk_frame_indices[chunk_idx]
        if chunk_idx == 0:
            scale = 1.0
            mask_positions = list(range(len(indices)))
        else:
            anchor_pos = indices.index(primary_anchor)
            anchor_depth = depth[anchor_pos]
            valid = (ref_depth > 1e-6) & (anchor_depth > 1e-6)
            scale = float(torch.median(ref_depth[valid] / anchor_depth[valid])) if valid.any() else 1.0
            mask_positions = [idx for idx, frame_idx in enumerate(indices) if frame_idx not in anchor_set]
        depth_scales.append(scale)
        if mask_positions:
            output_indices = torch.tensor([indices[idx] for idx in mask_positions], device=device, dtype=torch.long)
            input_indices = torch.tensor(mask_positions, device=device, dtype=torch.long)
            all_depth[output_indices] = depth[input_indices] * scale
            all_depth_conf[output_indices] = depth_conf[input_indices]
    return all_depth, all_depth_conf, depth_scales


def _da_vggt_make_chunks(
    pooled_tokens: torch.Tensor,
    chunk_size: int,
    method: str,
    n_anchors: int,
    lambda_div: float,
) -> tuple[list[list[int]], list[int]]:
    if chunk_size <= 0:
        raise ValueError("da_vggt_max_frames must be positive.")
    num_frames = pooled_tokens.shape[0]
    if method == "step":
        chunks, anchors = _da_vggt_step_sampling_split(num_frames, chunk_size, n_anchors)
        return chunks, anchors
    if method != "fl_maxmin":
        raise ValueError("da_vggt_sampling_method currently supports 'fl_maxmin' or 'step'.")
    feats = F.normalize(pooled_tokens.float(), dim=-1)
    sim = (feats @ feats.T).numpy()
    chunks, anchors = _da_vggt_fl_maxmin_split(sim, chunk_size, lambda_div, n_anchors)
    return chunks, anchors


def _da_vggt_step_sampling_split(num_frames: int, chunk_size: int, n_anchors: int) -> tuple[list[list[int]], list[int]]:
    stride = max(1, num_frames // chunk_size)
    anchors = _da_vggt_uniform_anchors(num_frames, n_anchors)
    anchor_set = set(anchors)
    chunks = []
    for batch_idx in range(stride):
        non_anchor = []
        frame_idx = 1 + batch_idx
        while frame_idx < num_frames:
            if frame_idx not in anchor_set:
                non_anchor.append(frame_idx)
            frame_idx += stride
        chunks.append(_da_vggt_insert_anchors(non_anchor, anchors))
    return chunks, anchors


def _da_vggt_uniform_anchors(num_frames: int, n_anchors: int) -> list[int]:
    if n_anchors <= 1:
        return [0]
    return list(dict.fromkeys(round(idx * (num_frames - 1) / (n_anchors - 1)) for idx in range(n_anchors)))


def _da_vggt_insert_anchors(non_anchor: list[int], anchors: list[int]) -> list[int]:
    result = [anchors[0]]
    original_len = len(non_anchor)
    n_eff = max(1, len(anchors))
    for anchor_idx, anchor in enumerate(anchors[1:], 1):
        insert_pos = original_len * anchor_idx // n_eff + (anchor_idx - 1)
        non_anchor.insert(insert_pos, anchor)
    result.extend(non_anchor)
    return result


def _da_vggt_fl_maxmin_split(
    sim_matrix,
    chunk_size: int,
    lambda_div: float,
    n_anchors: int,
) -> tuple[list[list[int]], list[int]]:
    import numpy as np

    num_frames = sim_matrix.shape[0]
    num_chunks = max(1, num_frames // chunk_size)
    sim = np.clip(sim_matrix, 0, None).astype(np.float32)
    chunks = [[] for _ in range(num_chunks)]
    coverage = np.zeros((num_chunks, num_frames), dtype=np.float32)
    chunk_scores = np.zeros(num_chunks, dtype=np.float64)
    chunk_counts = np.zeros(num_chunks, dtype=np.int32)
    assigned = np.zeros(num_frames, dtype=bool)

    for _ in range(min(num_chunks * chunk_size, num_frames)):
        eligible = chunk_counts < chunk_size
        target = int(np.argmin(np.where(eligible, chunk_scores, np.inf)))
        diff = sim - coverage[target][None, :]
        np.clip(diff, 0, None, out=diff)
        gains = diff.sum(axis=1)
        gains[assigned] = -1.0
        if lambda_div > 0 and chunks[target]:
            penalty = sim[:, chunks[target]].mean(axis=1)
            mask = ~assigned & (gains > 0)
            gains[mask] *= 1.0 - lambda_div * penalty[mask]
        best = int(np.argmax(gains))
        chunks[target].append(best)
        np.maximum(coverage[target], sim[best], out=coverage[target])
        chunk_scores[target] = float(coverage[target].sum())
        chunk_counts[target] += 1
        assigned[best] = True

    for frame_idx in np.where(~assigned)[0]:
        target = int(np.argmin(chunk_scores))
        chunks[target].append(int(frame_idx))
        np.maximum(coverage[target], sim[frame_idx], out=coverage[target])
        chunk_scores[target] = float(coverage[target].sum())

    anchors = _da_vggt_uniform_anchors(num_frames, n_anchors)
    anchor_set = set(anchors)
    for chunk_idx, chunk in enumerate(chunks):
        chunks[chunk_idx] = _da_vggt_insert_anchors([frame for frame in chunk if frame not in anchor_set], anchors)
    return chunks, anchors


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
