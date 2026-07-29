#!/usr/bin/env python3
"""Add camera-attention class labels to an existing PCA/K-means/CRF partition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from visualize_layer4_token_clusters import (
    list_images,
    render_camera_attention_overlay,
    render_dynamic_overlay,
    render_overlay,
)
from vggt_omega.evaluation import load_model
from vggt_omega.utils.load_fn import load_and_preprocess_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("tum_dynamic", "7scenes", "nrgbd"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--reference-labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/vggt_omega_1b_512.pt"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--input-mode", choices=("balanced", "max_size"), default="balanced")
    parser.add_argument("--save-visualizations", action="store_true", help="Write PNG overlays (disabled by default).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = np.load(args.reference_labels)
    if labels.ndim != 3 or int(labels.max()) != 2:
        raise ValueError("reference-labels must contain a K=3 [frames, height, width] partition")
    config_path = args.reference_labels.with_name("config.json")
    config = json.loads(config_path.read_text())
    source_indices = np.asarray(config["selected_indices"], dtype=np.int64)
    if len(source_indices) != len(labels):
        raise ValueError("Reference labels and selected_indices have different frame counts")
    all_paths = list_images(args.dataset, args.dataset_root, args.sequence)
    image_paths = [all_paths[index] for index in source_indices]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device, enable_camera=False, enable_depth=False)
    model.eval()
    images = load_and_preprocess_images(
        [str(path) for path in image_paths],
        mode=args.input_mode,
        image_resolution=args.image_resolution,
        patch_size=model.aggregator.patch_size,
    ).unsqueeze(0).to(device, non_blocking=True)
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16,
        enabled=device.type == "cuda",
    ):
        model.aggregator(images, stop_after_block=4, capture_camera_attention_block=4)
    captured = model.aggregator.last_camera_patch_attention
    if captured is None:
        raise RuntimeError("Camera attention capture failed")
    attention = captured[0].float().cpu().numpy().reshape(labels.shape)
    # Give every sampled frame equal weight.  A whole-sequence token mean would
    # otherwise let spatially larger K-means classes dominate the decision.
    per_frame_class_means = np.full((len(labels), 3), np.nan, dtype=np.float64)
    for frame_index, (frame_attention, frame_labels) in enumerate(zip(attention, labels)):
        for class_id in range(3):
            class_attention = frame_attention[frame_labels == class_id]
            if class_attention.size:
                per_frame_class_means[frame_index, class_id] = float(class_attention.mean())
    means = np.nanmean(per_frame_class_means, axis=0).tolist()
    dynamic_class = int(np.argmax(means))
    dynamic_mask = labels == dynamic_class

    np.save(args.output_dir / "labels.npy", labels)
    np.save(args.output_dir / "same_frame_patch_attention.npy", attention)
    np.save(args.output_dir / "dynamic_mask.npy", dynamic_mask)
    summary = {
        "dataset": args.dataset,
        "sequence": args.sequence,
        "reference_labels": str(args.reference_labels),
        "selected_source_indices": source_indices.tolist(),
        "camera_attention_block": 4,
        "rule": "dynamic_class = argmax_i mean_t(mean(camera_attention[t] | reference class[t]=i))",
        "per_frame_class_mean_attention": [[float(value) if np.isfinite(value) else None for value in row] for row in per_frame_class_means],
        "equal_frame_class_mean_attention": means,
        "dynamic_class": dynamic_class,
        "save_visualizations": args.save_visualizations,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.save_visualizations:
        cluster_dir = args.output_dir / "clusters"
        attention_dir = args.output_dir / "camera_attention"
        dynamic_dir = args.output_dir / "camera_attention_global_dynamic"
        for directory in (cluster_dir, attention_dir, dynamic_dir):
            directory.mkdir(exist_ok=True)
        for frame_index, (source_index, image_path, frame_labels, frame_attention, frame_mask) in enumerate(
            zip(source_indices, image_paths, labels, attention, dynamic_mask)
        ):
            cluster = render_overlay(image_path, frame_labels, 0.46, f"source frame {int(source_index)} | reference PCA 128 | K 3")
            cluster.save(cluster_dir / f"frame_{frame_index:02d}_source_{int(source_index):04d}_overlay.png")
            attention_image, _ = render_camera_attention_overlay(
                image_path, frame_attention, frame_labels, f"source frame {int(source_index)} | camera token self-frame attention"
            )
            attention_image.save(attention_dir / f"frame_{frame_index:02d}_source_{int(source_index):04d}_overlay.png")
            dynamic = render_dynamic_overlay(
                image_path, frame_mask, 0.46, f"source frame {int(source_index)} | global camera-attention dynamic class {dynamic_class}"
            )
            dynamic.save(dynamic_dir / f"frame_{frame_index:02d}_source_{int(source_index):04d}_overlay.png")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
