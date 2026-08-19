import torch
import torch.nn as nn
from pi3.models.da_vggt import cosine_similarity, diversity_partition, pseudo_positions, pose_weighted_similarity
import torch.nn.functional as F
from functools import partial
from copy import deepcopy

from .dinov2.layers import Mlp
from ..utils.geometry import homogenize_points
from .layers.pos_embed import RoPE2D, PositionGetter
from .layers.block import BlockRope
from .layers.attention import FlashAttentionRope
from .layers.transformer_head import TransformerDecoder, LinearPts3d
from .layers.camera_head import CameraHead
from .dinov2.hub.backbones import dinov2_vitl14, dinov2_vitl14_reg
from huggingface_hub import PyTorchModelHubMixin

class Pi3(nn.Module, PyTorchModelHubMixin):
    def __init__(
            self,
            pos_type='rope100',
            decoder_size='large',
            enable_token_merging=False,
            token_merging_method='none',
            token_merging_ratio=0.9,
            token_merging_frame_alpha=0.1,
            token_merging_frame_segment_threshold=0.9,
            token_merging_frame_merge_threshold=0.1,
            token_merging_frame_max_window=20,
            token_merging_frame_pool_stride=2,
            token_merging_frame_multi_max_group_size=4,
            token_merging_frame_multi_pair_threshold=0.95,
            token_merging_frame_multi_span_threshold=0.93,
            um_lambda_cost=None,
            um_spatial_radius=2,
            um_temporal_window=4,
            um_refresh_layers="0,9",
        ):
        super().__init__()

        # ----------------------
        #        Encoder
        # ----------------------
        self.encoder = dinov2_vitl14_reg(pretrained=False)
        self.patch_size = 14
        del self.encoder.mask_token
        self.enable_token_merging = enable_token_merging
        self.token_merging_method = token_merging_method
        self.token_merging_ratio = token_merging_ratio
        self.token_merging_frame_alpha = token_merging_frame_alpha
        self.token_merging_frame_segment_threshold = token_merging_frame_segment_threshold
        self.token_merging_frame_merge_threshold = token_merging_frame_merge_threshold
        self.token_merging_frame_max_window = token_merging_frame_max_window
        self.token_merging_frame_pool_stride = token_merging_frame_pool_stride
        self.token_merging_frame_multi_max_group_size = token_merging_frame_multi_max_group_size
        self.token_merging_frame_multi_pair_threshold = token_merging_frame_multi_pair_threshold
        self.token_merging_frame_multi_span_threshold = token_merging_frame_multi_span_threshold
        self.last_frame_merge_stats = []
        self.last_token_merging_stats = []
        self.um_lambda_cost = um_lambda_cost
        self.um_spatial_radius = int(um_spatial_radius)
        self.um_temporal_window = int(um_temporal_window)
        self.um_refresh_layers = {int(v) for v in um_refresh_layers.split(",") if v.strip()} if um_lambda_cost is not None else set()
        self._um_plan = None
        # Populated once per U-M refresh during the most recent forward.  The
        # evaluator consumes this to report token retention on the same
        # full-patch-token basis as VGGT-Omega.
        self.last_um_layer_retention = []

        # ----------------------
        #  Positonal Encoding
        # ----------------------
        self.pos_type = pos_type if pos_type is not None else 'none'
        self.rope=None
        if self.pos_type.startswith('rope'): # eg rope100 
            if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            freq = float(self.pos_type[len('rope'):])
            self.rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()
        else:
            raise NotImplementedError
        

        # ----------------------
        #        Decoder
        # ----------------------
        enc_embed_dim = self.encoder.blocks[0].attn.qkv.in_features        # 1024
        if decoder_size == 'small':
            dec_embed_dim = 384
            dec_num_heads = 6
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'base':
            dec_embed_dim = 768
            dec_num_heads = 12
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'large':
            dec_embed_dim = 1024
            dec_num_heads = 16
            mlp_ratio = 4
            dec_depth = 36
        else:
            raise NotImplementedError
        self.decoder = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=0.01,
                qk_norm=True,
                attn_class=FlashAttentionRope,
                rope=self.rope
            ) for _ in range(dec_depth)])
        self.dec_embed_dim = dec_embed_dim

        # ----------------------
        #     Register_token
        # ----------------------
        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, self.dec_embed_dim))
        nn.init.normal_(self.register_token, std=1e-6)

        # ----------------------
        #  Local Points Decoder
        # ----------------------
        self.point_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=1024,
            rope=self.rope,
        )
        self.point_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)

        # ----------------------
        #     Conf Decoder
        # ----------------------
        self.conf_decoder = deepcopy(self.point_decoder)
        self.conf_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=1)

        # ----------------------
        #  Camera Pose Decoder
        # ----------------------
        self.camera_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,                # 8
            out_dim=512,
            rope=self.rope,
            use_checkpoint=False
        )
        self.camera_head = CameraHead(dim=512)

        # For ImageNet Normalize
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)


    def decode(self, hidden, N, H, W):
        BN, hw, _ = hidden.shape
        B = BN // N
        original_N = N

        final_output = []
        
        hidden = hidden.reshape(B*N, hw, -1)

        register_token = self.register_token.repeat(B, N, 1, 1).reshape(B*N, *self.register_token.shape[-2:])

        # Concatenate special tokens with patch tokens
        hidden = torch.cat([register_token, hidden], dim=1)
        hw = hidden.shape[1]
        patch_grid_size = (H // self.patch_size, W // self.patch_size)
        frame_merge_state = None
        active_N = N
        self._um_plan = None
        self.last_um_layer_retention = []

        if self.pos_type.startswith('rope'):
            pos = self.position_getter(B * N, H//self.patch_size, W//self.patch_size, hidden.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * N, self.patch_start_idx, 2).to(hidden.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)
       
        for i in range(len(self.decoder)):
            blk = self.decoder[i]

            if i % 2 == 0:
                pos = pos.reshape(B*active_N, hw, -1)
                hidden = hidden.reshape(B*active_N, hw, -1)
            else:
                if (
                    self.enable_token_merging
                    and self.token_merging_method == "frame_persistent_spatial"
                    and frame_merge_state is None
                ):
                    hidden = hidden.reshape(B, active_N, hw, -1)
                    pos = pos.reshape(B, active_N, hw, -1)
                    hidden, pos, frame_merge_state = self._frame_merge(hidden, pos, patch_grid_size)
                    self._record_frame_merge_stats(i, original_N, frame_merge_state.states)
                    active_N = hidden.shape[1]
                pos = pos.reshape(B, active_N*hw, -1)
                hidden = hidden.reshape(B, active_N*hw, -1)
                if self.enable_token_merging and self.token_merging_method in {"fastvggt", "frame_persistent_spatial"}:
                    blk.attn.fastvggt_merge_ratio = self.token_merging_ratio
                    blk.attn.fastvggt_patch_grid_size = patch_grid_size
                    blk.attn.fastvggt_num_frames = active_N
                    blk.attn.fastvggt_special_token_count = self.patch_start_idx
                if self.um_lambda_cost is not None:
                    from .um import build_um_plan
                    global_block_idx = i // 2
                    if self._um_plan is None or global_block_idx in self.um_refresh_layers:
                        self._um_plan = build_um_plan(
                            hidden, active_N, self.patch_start_idx, patch_grid_size,
                            self.um_spatial_radius, self.um_temporal_window, self.um_lambda_cost,
                        )
                        full_patch_tokens = int(active_N * self._um_plan.patch_count)
                        representative_count = int(self._um_plan.representative_source_indices.numel())
                        self.last_um_layer_retention.append({
                            "source_layer": int(global_block_idx),
                            "representative_count": representative_count,
                            "full_patch_tokens": full_patch_tokens,
                            "patch_token_retention_percent": 100.0 * representative_count / full_patch_tokens,
                        })
                    blk.attn.um_plan = self._um_plan

            try:
                hidden = blk(hidden, xpos=pos)
                stats = getattr(blk.attn, "last_fastvggt_stats", None)
                if stats:
                    self.last_token_merging_stats.append({"block": int(i), "mode": self.token_merging_method, **stats})
            finally:
                if i % 2 == 1:
                    blk.attn.fastvggt_merge_ratio = None
                    blk.attn.fastvggt_patch_grid_size = None
                    blk.attn.fastvggt_num_frames = None
                    blk.attn.fastvggt_special_token_count = None
                    blk.attn.um_plan = None
                    blk.attn.um_norm1 = None
                if i % 2 == 0:
                    blk.attn.um_plan = None
                    blk.attn.um_norm1 = None

            if i+1 in [len(self.decoder)-1, len(self.decoder)]:
                output = hidden.reshape(B, active_N, hw, -1)
                if frame_merge_state is not None:
                    output = _restore_frame_tokens(output, frame_merge_state)
                final_output.append(output.reshape(B*original_N, hw, -1))

        if frame_merge_state is not None:
            pos = pos.reshape(B, active_N, hw, -1)
            pos = _restore_frame_tokens(pos, frame_merge_state)
            pos = pos.reshape(B*original_N, hw, -1)
        else:
            pos = pos.reshape(B*original_N, hw, -1)
        return torch.cat([final_output[0], final_output[1]], dim=-1), pos
    
    def forward(self, imgs):
        self.last_frame_merge_stats = []
        self.last_token_merging_stats = []
        imgs = (imgs - self.image_mean) / self.image_std

        B, N, _, H, W = imgs.shape
        patch_h, patch_w = H // 14, W // 14
        
        # encode by dinov2
        imgs = imgs.reshape(B*N, _, H, W)
        hidden = self.encoder(imgs, is_training=True)

        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]

        return self.forward_from_encoder_tokens(hidden, imgs, B, N, H, W)

    def forward_from_encoder_tokens(self, hidden, imgs, B, N, H, W):
        """Run Pi3 decoder/heads from cached DINO patch tokens (DA-VGGT entry)."""
        patch_h, patch_w = H // 14, W // 14

        hidden, pos = self.decode(hidden, N, H, W)

        point_hidden = self.point_decoder(hidden, xpos=pos)
        conf_hidden = self.conf_decoder(hidden, xpos=pos)
        camera_hidden = self.camera_decoder(hidden, xpos=pos)

        with torch.amp.autocast(device_type='cuda', enabled=False):
            # local points
            point_hidden = point_hidden.float()
            ret = self.point_head([point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
            xy, z = ret.split([2, 1], dim=-1)
            z = torch.exp(z)
            local_points = torch.cat([xy * z, z], dim=-1)

            # confidence
            conf_hidden = conf_hidden.float()
            conf = self.conf_head([conf_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)

            # camera
            camera_hidden = camera_hidden.float()
            camera_poses = self.camera_head(camera_hidden[:, self.patch_start_idx:], patch_h, patch_w).reshape(B, N, 4, 4)

            # unproject local points using camera poses
            points = torch.einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize_points(local_points))[..., :3]

        return dict(
            points=points,
            local_points=local_points,
            conf=conf,
            camera_poses=camera_poses,
        )

    @torch.inference_mode()
    def forward_dino(self, imgs, batch_size=32):
        """Cache normalized DINO patch tokens on CPU for DA-VGGT scheduling."""
        B, N, channels, H, W = imgs.shape
        if B != 1 or channels != 3:
            raise ValueError("DA-VGGT token caching supports B=1 RGB videos")
        normalized = (imgs - self.image_mean) / self.image_std
        flat = normalized.reshape(B * N, channels, H, W)
        patches, pooled = [], []
        for start in range(0, len(flat), batch_size):
            token = self.encoder(flat[start:start + batch_size], is_training=True)
            if isinstance(token, dict):
                token = token["x_norm_patchtokens"]
            patches.append(token.cpu()); pooled.append(token.float().mean(1).cpu())
        return torch.cat(patches), torch.cat(pooled)

    @torch.inference_mode()
    def forward_da_vggt(self, imgs, chunk_size=50, dino_batch_size=32, local_search_iters=5, pseudo_pose_gamma=1e-3, pose_tau=None):
        """Full DA-VGGT scheduling with cached Pi3 encoder tokens."""
        if imgs.ndim == 4: imgs = imgs.unsqueeze(0)
        B, count, _, H, W = imgs.shape
        if B != 1: raise ValueError("DA-VGGT requires B=1")
        if count <= chunk_size: return self(imgs)
        cached, pooled = self.forward_dino(imgs, dino_batch_size)
        sim = cosine_similarity(pooled); initial = diversity_partition(sim, chunk_size, iters=local_search_iters)
        def run(indices):
            ids = torch.tensor(indices, device=imgs.device)
            normalized = (imgs[:, ids] - self.image_mean) / self.image_std
            output = self.forward_from_encoder_tokens(cached[ids.cpu()].to(imgs.device), normalized.reshape(len(indices), 3, H, W), 1, len(indices), H, W)
            return output
        first = run(initial[0]); pseudo = pseudo_positions(sim, initial[0], first['camera_poses'][0, :, :3, 3].float().cpu().numpy(), pseudo_pose_gamma)
        refined = diversity_partition(pose_weighted_similarity(sim, pseudo, pose_tau), chunk_size, iters=local_search_iters)
        local_points, confs, poses, reference = [None]*count, [None]*count, [None]*count, None
        for indices in refined:
            out = run(indices); c2w = out['camera_poses'][0].float()
            reference = c2w[0].clone() if reference is None else reference
            aligned = reference @ torch.linalg.inv(c2w[0]) @ c2w
            for local, original in enumerate(indices):
                if original == 0 and poses[0] is not None: continue
                poses[original] = aligned[local]; local_points[original] = out['local_points'][0, local]; confs[original] = out['conf'][0, local]
        pose_stack, local_stack, conf_stack = torch.stack(poses).unsqueeze(0), torch.stack(local_points).unsqueeze(0), torch.stack(confs).unsqueeze(0)
        points = torch.einsum('bnij,bnhwj->bnhwi', pose_stack, homogenize_points(local_stack))[..., :3]
        return {'camera_poses': pose_stack, 'local_points': local_stack, 'conf': conf_stack, 'points': points, 'images': imgs, 'chunk_frame_indices': refined, 'initial_chunk_frame_indices': initial}

    def _frame_merge(self, tokens, pos, patch_grid_size):
        merged_tokens = []
        merged_pos = []
        states = []
        for batch_idx in range(tokens.shape[0]):
            one_tokens, one_pos, state = _frame_merge_one(
                tokens[batch_idx],
                pos[batch_idx],
                self.patch_start_idx,
                patch_grid_size,
                self.token_merging_frame_alpha,
                self.token_merging_frame_segment_threshold,
                self.token_merging_frame_merge_threshold,
                self.token_merging_frame_max_window,
                self.token_merging_frame_pool_stride,
                self.token_merging_frame_multi_max_group_size,
                self.token_merging_frame_multi_pair_threshold,
                self.token_merging_frame_multi_span_threshold,
            )
            merged_tokens.append(one_tokens)
            merged_pos.append(one_pos)
            states.append(state)
        active_counts = {state.active_frames for state in states}
        if len(active_counts) != 1:
            raise ValueError(f"Frame merging produced different active frame counts across batch: {sorted(active_counts)}")
        return torch.stack(merged_tokens, dim=0), torch.stack(merged_pos, dim=0), _BatchFrameMergeState(states)

    def _record_frame_merge_stats(self, block_idx, original_frames, states):
        active_frames = [state.active_frames for state in states]
        active_mean = sum(active_frames) / len(active_frames)
        retention_ratio = active_mean / original_frames if original_frames else 0.0
        segment_counts = [len(state.segments) for state in states]
        merge_group_sizes = [size for state in states for size in state.merge_group_sizes]
        multi_group_sizes = [size for size in merge_group_sizes if size > 2]
        self.last_frame_merge_stats.append(
            {
                "block": int(block_idx),
                "mode": "persistent",
                "original_frames": int(original_frames),
                "active_frames_min": int(min(active_frames)),
                "active_frames_mean": float(active_mean),
                "active_frames_max": int(max(active_frames)),
                "retention_ratio_mean": float(retention_ratio),
                "merge_ratio_mean": float(1.0 - retention_ratio),
                "segments_min": int(min(segment_counts)),
                "segments_mean": float(sum(segment_counts) / len(segment_counts)),
                "segments_max": int(max(segment_counts)),
                "segments": [[(int(start), int(end)) for start, end in state.segments] for state in states],
                "merge_groups_count": int(len(merge_group_sizes)),
                "merge_group_size_mean": float(sum(merge_group_sizes) / len(merge_group_sizes)) if merge_group_sizes else 0.0,
                "merge_group_size_max": int(max(merge_group_sizes)) if merge_group_sizes else 0,
                "multi_frame_groups_count": int(len(multi_group_sizes)),
                "multi_frame_group_size_mean": float(sum(multi_group_sizes) / len(multi_group_sizes)) if multi_group_sizes else 0.0,
            }
        )


class _FrameMergeState:
    def __init__(self, inverse, active_mask, segments, merge_group_sizes=None):
        self.inverse = inverse
        self.active_mask = active_mask
        self.segments = segments
        self.merge_group_sizes = merge_group_sizes or []
        self.active_frames = int(inverse.max().item()) + 1 if inverse.numel() else 0


class _BatchFrameMergeState:
    def __init__(self, states):
        self.states = states
        self.active_frames = states[0].active_frames if states else 0


def _restore_frame_tokens(tokens, state):
    restored = [tokens[batch_idx, frame_state.inverse] for batch_idx, frame_state in enumerate(state.states)]
    return torch.stack(restored, dim=0)


def _frame_merge_one(
    tokens,
    pos,
    patch_token_start,
    patch_grid_size,
    alpha,
    segment_threshold,
    merge_threshold,
    max_window,
    pool_stride,
    multi_max_group_size,
    multi_pair_threshold,
    multi_span_threshold,
):
    num_frames = tokens.shape[0]
    if num_frames <= 1:
        inverse = torch.arange(num_frames, device=tokens.device)
        active_mask = torch.ones(num_frames, dtype=torch.bool, device=tokens.device)
        return tokens, pos, _FrameMergeState(inverse, active_mask, [(0, num_frames - 1)], [])

    patch_tokens = tokens[:, patch_token_start:]
    pooled = _pool_frame_similarity_tokens(patch_tokens, patch_grid_size, pool_stride)
    segments = _streaming_frame_segments(pooled, alpha, segment_threshold, max_window)

    active_tokens = []
    active_pos = []
    inverse = torch.empty(num_frames, dtype=torch.long, device=tokens.device)
    active_mask = torch.zeros(num_frames, dtype=torch.bool, device=tokens.device)
    assigned = [False] * num_frames
    merge_group_sizes = []

    def append_frame(frame_idx):
        inverse[frame_idx] = len(active_tokens)
        active_mask[frame_idx] = True
        assigned[frame_idx] = True
        active_tokens.append(tokens[frame_idx])
        active_pos.append(pos[frame_idx])

    def append_group(frame_indices):
        active_idx = len(active_tokens)
        for offset, frame_idx in enumerate(frame_indices):
            inverse[frame_idx] = active_idx
            active_mask[frame_idx] = offset == 0
            assigned[frame_idx] = True
        active_tokens.append(tokens[frame_indices].float().mean(dim=0).to(tokens.dtype))
        active_pos.append(pos[frame_indices[0]])
        merge_group_sizes.append(len(frame_indices))

    def append_merge(left_idx, right_idx, cur_sim, next_sim):
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
        active_pos.append(pos[left_idx])
        merge_group_sizes.append(2)

    def can_merge_group(start_idx, group_size, end):
        end_idx = start_idx + group_size - 1
        if end_idx > end:
            return False
        pair_sims = [_frame_pair_similarity(pooled[idx], pooled[idx + 1]).float() for idx in range(start_idx, end_idx)]
        span_sim = _frame_pair_similarity(pooled[start_idx], pooled[end_idx]).float()
        return bool(all(sim > multi_pair_threshold for sim in pair_sims) and span_sim > multi_span_threshold)

    for start, end in segments:
        if start == end:
            append_frame(start)
            continue
        append_frame(start)
        cursor = start + 1
        while cursor < end:
            group_size = 0
            for candidate_size in range(min(multi_max_group_size, 4), 2, -1):
                if can_merge_group(cursor, candidate_size, end):
                    group_size = candidate_size
                    break
            if group_size:
                append_group(list(range(cursor, cursor + group_size)))
                cursor += group_size
                continue

            cur_sim = _frame_pair_similarity(pooled[cursor], pooled[cursor + 1])
            next_sim = _frame_pair_similarity(pooled[cursor + 1], pooled[cursor + 2]) if cursor + 1 < end else cur_sim
            if cur_sim > merge_threshold and cur_sim > next_sim:
                append_merge(cursor, cursor + 1, cur_sim, next_sim)
                cursor += 2
            else:
                append_frame(cursor)
                cursor += 1
        if not assigned[end]:
            append_frame(end)

    return torch.stack(active_tokens, dim=0), torch.stack(active_pos, dim=0), _FrameMergeState(
        inverse, active_mask, segments, merge_group_sizes
    )


def _pool_frame_similarity_tokens(patch_tokens, patch_grid_size, pool_stride):
    num_frames, _, channels = patch_tokens.shape
    grid_h, grid_w = patch_grid_size
    patches = patch_tokens.reshape(num_frames, grid_h, grid_w, channels).permute(0, 3, 1, 2).float()
    if pool_stride > 1:
        patches = F.avg_pool2d(patches, kernel_size=pool_stride, stride=pool_stride, ceil_mode=False)
    return F.normalize(patches.flatten(2).transpose(1, 2), dim=-1)


def _frame_pair_similarity(left, right):
    return (left * right).sum(dim=-1).mean()


def _streaming_frame_segments(pooled, alpha, segment_threshold, max_window):
    num_frames = pooled.shape[0]
    segments = []
    start = 0
    anchor = pooled[0]
    for frame_idx in range(1, num_frames):
        sim = _frame_pair_similarity(anchor, pooled[frame_idx])
        window_full = max_window > 0 and (frame_idx - start + 1) > max_window
        if sim < segment_threshold or window_full:
            segments.append((start, frame_idx - 1))
            start = frame_idx
            anchor = pooled[frame_idx]
        else:
            anchor = F.normalize(alpha * anchor + (1.0 - alpha) * pooled[frame_idx], dim=-1)
    segments.append((start, num_frames - 1))
    return segments
