#!/usr/bin/env python3
"""Sequence-level camera-attention dynamic-region diagnostic for long videos.

The full 300-frame global-attention matrix is prohibitively large.  This tool
extracts block-4 camera attention in fixed temporal chunks, then fits one
PCA/K-means/CRF partition over every patch from the complete sequence.  The
dynamic decision is made once per sequence: the class with the highest mean
camera attention over all frames is selected as dynamic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import MiniBatchKMeans

from visualize_layer4_token_clusters import (
    edge_aware_crf_labels,
    list_images,
    load_patch_rgb_features,
    render_camera_attention_overlay,
    render_dynamic_overlay,
    render_overlay,
)
from vggt_omega.evaluation import load_model
from vggt_omega.utils.load_fn import load_and_preprocess_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("tum_dynamic", "7scenes"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/vggt_omega_1b_512.pt"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--input-mode", choices=("balanced", "max_size"), default="balanced")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--num-frames", type=int, default=0, help="Uniform samples from max-frames; 0 uses every available frame.")
    parser.add_argument("--chunk-frames", type=int, default=10)
    parser.add_argument("--pca-fit-samples", type=int, default=8192)
    parser.add_argument("--pca-dim", type=int, default=128)
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--crf-spatial-weight", type=float, default=0.9)
    parser.add_argument("--crf-temporal-weight", type=float, default=0.08)
    parser.add_argument("--crf-color-sigma", type=float, default=0.18)
    parser.add_argument("--crf-unary-temperature", type=float, default=1.0)
    parser.add_argument("--crf-iterations", type=int, default=10)
    parser.add_argument("--render-frames", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def batches(total: int, size: int):
    for start in range(0, total, size):
        yield start, min(start + size, total)


def main() -> None:
    args = parse_args()
    if args.clusters != 3:
        raise ValueError("This diagnostic is defined for exactly K=3")
    if args.max_frames < 1 or args.num_frames < 0 or args.chunk_frames < 1 or args.pca_fit_samples < 128 or args.render_frames < 1:
        raise ValueError("max-frames, num-frames, chunk-frames, pca-fit-samples, and render-frames are invalid")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    source_paths = list_images(args.dataset, args.dataset_root, args.sequence)[: args.max_frames]
    if args.num_frames:
        if args.num_frames > len(source_paths):
            raise ValueError(f"num-frames={args.num_frames} exceeds available source frames={len(source_paths)}")
        source_indices = np.linspace(0, len(source_paths) - 1, args.num_frames).round().astype(int)
        paths = [source_paths[index] for index in source_indices]
    else:
        source_indices = np.arange(len(source_paths))
        paths = source_paths
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = args.output_dir / "_temporary"
    temp_dir.mkdir(exist_ok=True)
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device, enable_camera=False, enable_depth=False)
    model.eval()

    raw_chunks: list[Path] = []
    attention_chunks: list[np.ndarray] = []
    grid_height = grid_width = patch_count = 0
    for start, end in batches(len(paths), args.chunk_frames):
        images = load_and_preprocess_images(
            [str(path) for path in paths[start:end]],
            mode=args.input_mode,
            image_resolution=args.image_resolution,
            patch_size=model.aggregator.patch_size,
        ).unsqueeze(0).to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16,
            enabled=device.type == "cuda",
        ):
            cached, patch_start = model.aggregator(images, stop_after_block=4, capture_camera_attention_block=4)
        layer4 = cached[4]
        if layer4 is None or model.aggregator.last_camera_patch_attention is None:
            raise RuntimeError("Block-4 camera-attention capture failed")
        tokens = layer4[0, :, patch_start:, :].float().cpu().numpy()
        attention = model.aggregator.last_camera_patch_attention[0].float().cpu().numpy()
        if not patch_count:
            grid_height = images.shape[-2] // model.aggregator.patch_size
            grid_width = images.shape[-1] // model.aggregator.patch_size
            patch_count = grid_height * grid_width
        if tokens.shape[1] != patch_count or attention.shape != tokens.shape[:2]:
            raise RuntimeError("Inconsistent patch-token or camera-attention layout across chunks")
        chunk_path = temp_dir / f"tokens_{start:04d}_{end:04d}.npy"
        np.save(chunk_path, tokens.astype(np.float16))
        raw_chunks.append(chunk_path)
        attention_chunks.append(attention.astype(np.float32))
        print(f"extracted frames {start}:{end}", flush=True)

    attention_maps = np.concatenate(attention_chunks, axis=0).reshape(len(paths), grid_height, grid_width)
    channel_sum = np.zeros((patch_count, tokens.shape[-1]), dtype=np.float64)
    channel_sumsq = np.zeros_like(channel_sum)
    for chunk_path in raw_chunks:
        chunk = np.load(chunk_path, mmap_mode="r").astype(np.float32)
        channel_sum += chunk.sum(axis=0)
        channel_sumsq += np.square(chunk).sum(axis=0)
    total_frames = len(paths)
    temporal_mean = channel_sum / total_frames
    temporal_std = np.sqrt(np.maximum(channel_sumsq / total_frames - np.square(temporal_mean), 1e-6))

    pca_dim = min(args.pca_dim, total_frames * patch_count, tokens.shape[-1])
    sampled_features: list[np.ndarray] = []
    remaining_samples = args.pca_fit_samples
    sample_quota = int(np.ceil(args.pca_fit_samples / len(raw_chunks)))
    for chunk_index, chunk_path in enumerate(raw_chunks):
        chunk = np.load(chunk_path, mmap_mode="r").astype(np.float32)
        temporal = (chunk - temporal_mean[None]) / temporal_std[None]
        spatial = (chunk - chunk.mean(axis=1, keepdims=True)) / (chunk.std(axis=1, keepdims=True) + 1e-6)
        normalized = 0.5 * (temporal + spatial)
        normalized /= np.linalg.norm(normalized, axis=-1, keepdims=True) + 1e-6
        chunks_left = len(raw_chunks) - chunk_index
        take = min(max(sample_quota, remaining_samples // chunks_left), remaining_samples, len(normalized) * patch_count)
        if take:
            indices = np.random.default_rng(args.seed + len(sampled_features)).choice(
                len(normalized) * patch_count, size=take, replace=False
            )
            sampled_features.append(normalized.reshape(-1, normalized.shape[-1])[indices])
            remaining_samples -= take
        if not remaining_samples:
            break
    pca_sample = np.concatenate(sampled_features, axis=0)
    pca_sample_tensor = torch.from_numpy(pca_sample).to(device=device, dtype=torch.float32)
    pca_mean = pca_sample_tensor.mean(dim=0)
    _, _, pca_components = torch.pca_lowrank(pca_sample_tensor - pca_mean, q=pca_dim, center=False, niter=3)
    pca_mean = pca_mean.cpu().numpy()
    pca_components = pca_components.cpu().numpy()
    del pca_sample_tensor

    reduced_path = temp_dir / "reduced.npy"
    reduced = np.lib.format.open_memmap(reduced_path, mode="w+", dtype=np.float32, shape=(total_frames, patch_count, pca_dim))
    cursor = 0
    for chunk_path in raw_chunks:
        chunk = np.load(chunk_path, mmap_mode="r").astype(np.float32)
        temporal = (chunk - temporal_mean[None]) / temporal_std[None]
        spatial = (chunk - chunk.mean(axis=1, keepdims=True)) / (chunk.std(axis=1, keepdims=True) + 1e-6)
        normalized = 0.5 * (temporal + spatial)
        normalized /= np.linalg.norm(normalized, axis=-1, keepdims=True) + 1e-6
        transformed = (normalized.reshape(-1, normalized.shape[-1]) - pca_mean) @ pca_components
        reduced[cursor : cursor + len(chunk)] = transformed.reshape(len(chunk), patch_count, pca_dim)
        cursor += len(chunk)

    kmeans = MiniBatchKMeans(n_clusters=3, batch_size=args.chunk_frames * patch_count, n_init=10, random_state=args.seed)
    for start, end in batches(total_frames, args.chunk_frames):
        kmeans.partial_fit(reduced[start:end].reshape(-1, pca_dim))
    raw_labels = np.empty((total_frames, grid_height, grid_width), dtype=np.int64)
    unary = np.empty((total_frames, grid_height, grid_width, 3), dtype=np.float32)
    for start, end in batches(total_frames, args.chunk_frames):
        features = reduced[start:end].reshape(-1, pca_dim)
        distances = kmeans.transform(features).astype(np.float32) ** 2
        distances -= distances.min(axis=1, keepdims=True)
        raw_labels[start:end] = kmeans.predict(features).reshape(end - start, grid_height, grid_width)
        unary[start:end] = distances.reshape(end - start, grid_height, grid_width, 3)
    margin = float(np.partition(unary.reshape(-1, 3), 1, axis=1)[:, 1].mean())
    unary /= max(margin, 1e-6)
    patch_rgb = load_patch_rgb_features(paths, grid_height, grid_width)
    labels = edge_aware_crf_labels(
        unary,
        patch_rgb,
        args.crf_spatial_weight,
        args.crf_temporal_weight,
        args.crf_color_sigma,
        args.crf_unary_temperature,
        args.crf_iterations,
    )
    global_means = [float(attention_maps[labels == class_id].mean()) if np.any(labels == class_id) else 0.0 for class_id in range(3)]
    dynamic_class = int(np.argmax(global_means))
    sequence_dynamic_mask = labels == dynamic_class
    if not np.array_equal(sequence_dynamic_mask, labels == dynamic_class):
        raise RuntimeError("Dynamic mask must exactly match the selected cluster label")

    np.save(args.output_dir / "labels.npy", labels)
    np.save(args.output_dir / "same_frame_patch_attention.npy", attention_maps)
    np.save(args.output_dir / "dynamic_mask.npy", sequence_dynamic_mask)
    summary = {
        "dataset": args.dataset,
        "sequence": args.sequence,
        "frames": total_frames,
        "source_frame_count": len(source_paths),
        "selected_source_indices": source_indices.tolist(),
        "chunk_frames": args.chunk_frames,
        "pca_dim": pca_dim,
        "pca_fit_samples": int(len(pca_sample)),
        "clusters": 3,
        "camera_attention_block": 4,
        "rule": "dynamic_class = argmax_i mean(camera_attention | class=i, over all frames)",
        "global_class_mean_attention": global_means,
        "dynamic_class": dynamic_class,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

    render_indices = np.linspace(0, total_frames - 1, min(args.render_frames, total_frames)).round().astype(int)
    attention_dir = args.output_dir / "camera_attention"
    dynamic_dir = args.output_dir / "camera_attention_dynamic"
    attention_dir.mkdir(exist_ok=True)
    dynamic_dir.mkdir(exist_ok=True)
    cluster_dir = args.output_dir / "clusters"
    cluster_dir.mkdir(exist_ok=True)
    for frame_index in render_indices:
        source_index = int(source_indices[frame_index])
        cluster_image = render_overlay(
            paths[frame_index], labels[frame_index], 0.46,
            f"source frame {source_index} | PCA {pca_dim} | K 3",
        )
        cluster_image.save(cluster_dir / f"frame_{frame_index:04d}_source_{source_index:04d}_overlay.png")
        attention_image, _ = render_camera_attention_overlay(
            paths[frame_index], attention_maps[frame_index], labels[frame_index],
            f"source frame {source_index} | camera token self-frame attention",
        )
        attention_image.save(attention_dir / f"frame_{frame_index:04d}_source_{source_index:04d}_overlay.png")
        label = dynamic_class if dynamic_class is not None else "none"
        dynamic_image = render_dynamic_overlay(
            paths[frame_index], sequence_dynamic_mask[frame_index], 0.46,
            f"source frame {source_index} | sequence dynamic class {label}",
        )
        dynamic_image.save(dynamic_dir / f"frame_{frame_index:04d}_source_{source_index:04d}_overlay.png")

    for chunk_path in raw_chunks:
        chunk_path.unlink(missing_ok=True)
    reduced_path.unlink(missing_ok=True)
    temp_dir.rmdir()
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
