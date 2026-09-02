#!/usr/bin/env python3
"""Forward-only ScanNet VRAM benchmark for original VGGT acceleration modes."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from vggt.evaluation import load_model
from vggt.utils.load_fn import load_and_preprocess_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--num-frames", type=int, default=300)
    parser.add_argument("--sampling-stride", type=int, default=3)
    parser.add_argument("--image-resolution", type=int, default=518)
    parser.add_argument("--method", choices=("baseline", "fastvggt", "u-m", "avggt"), required=True)
    parser.add_argument("--avggt-subsample-factor", type=int, choices=(2, 4, 6, 9), default=4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def select_images(args: argparse.Namespace) -> list[str]:
    sequence_dir = args.data_root / args.sequence
    records: list[Path] = []
    for rgb in sorted((sequence_dir / "color").glob("*.jpg"), key=lambda path: int(path.stem)):
        if (sequence_dir / "depth" / f"{rgb.stem}.png").is_file() and (
            sequence_dir / "pose" / f"{rgb.stem}.txt"
        ).is_file():
            records.append(rgb)
    stop = args.sampling_stride * args.num_frames
    selected = records[:stop:args.sampling_stride]
    if len(selected) != args.num_frames:
        raise RuntimeError(
            f"Sequence provides only {len(selected)} frames with stride {args.sampling_stride}; "
            f"need {args.num_frames}"
        )
    return [str(path) for path in selected]


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    paths = select_images(args)
    model = load_model(
        args.checkpoint,
        device,
        enable_camera=True,
        enable_depth=True,
        inter_frame_attention="global",
        enable_token_merging=args.method == "fastvggt",
        token_merging_ratio=0.9,
        token_merging_method="spatial",
        um_lambda_cost=0.04 if args.method == "u-m" else None,
        um_spatial_radius=2,
        um_temporal_window=4,
        um_refresh_layers="0,10,17",
        avggt_subsample_factor=args.avggt_subsample_factor if args.method == "avggt" else None,
        model_bfloat16=True,
    )
    images = load_and_preprocess_images(paths, mode="crop", target_size=args.image_resolution).to(
        device=device, dtype=torch.bfloat16
    )
    result: dict[str, object] = {
        "model": "vggt",
        "method": args.method,
        "sequence": args.sequence,
        "num_frames": args.num_frames,
        "sampling_stride": args.sampling_stride,
        "image_resolution": args.image_resolution,
        "cached_layers": sorted(model.aggregator.cached_layer_indices),
    }
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictions = model(images)
        torch.cuda.synchronize(device)
        result.update(
            success=True,
            forward_seconds=time.perf_counter() - started,
            peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30,
            output_keys=sorted(predictions),
        )
        del predictions
    except torch.cuda.OutOfMemoryError as exc:
        result.update(
            success=False,
            error="CUDA out of memory",
            detail=str(exc),
            peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30,
        )
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result.get("success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
