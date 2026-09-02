#!/usr/bin/env python3
"""One-video 7Scenes benchmark for VGGT and all enabled accelerators.

The four methods intentionally run in separate environments.  LiteVGGT and
the local VGGT project both expose a top-level ``vggt`` package, while Co-Me
requires its own compiled CUDA extensions.  This launcher imports exactly one
backend per process and writes an identically shaped JSON timing/VRAM/accuracy record.

All backends use the same centre-crop-to-square 504 px preprocessing.  504 is
divisible by both the 14 px VGGT patch size and Co-Me's released 3 x 3 token
groups (36 x 36 patch grid).
"""
from __future__ import annotations

import argparse
import gc
import importlib
import inspect
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


VGGT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = VGGT_ROOT.parent
LITE_ROOT = WORKSPACE_ROOT / "LiteVGGT-repo"
COME_ROOT = WORKSPACE_ROOT / "CoMe"

# Geometry evaluation is shared by all three model repositories rather than
# installed as a package.  Keep it importable even when the launcher is called
# with only ``vggt`` in PYTHONPATH.
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=("baseline", "fastvggt", "da-vggt", "sparse-vggt", "u-m", "avggt", "litevggt", "come"),
        required=True,
    )
    parser.add_argument("--dataset", choices=("7scenes", "nrgbd", "scannet"), default="7scenes")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sequence", default="chess/seq-03")
    parser.add_argument("--num-frames", type=int, default=300)
    parser.add_argument("--sampling-stride", type=int, default=3)
    parser.add_argument(
        "--sampling-mode", choices=("fixed_stride", "uniform_first_last", "all_stride"),
        default="fixed_stride",
        help="fixed_stride selects from frame 0; uniform_first_last retains endpoints; all_stride keeps every stride-th valid frame.",
    )
    parser.add_argument("--image-size", type=int, default=504)
    parser.add_argument("--timing-repeats", type=int, default=1)
    parser.add_argument("--avggt-subsample-factor", choices=(2, 4, 6, 9), type=int, default=4)
    parser.add_argument("--merge-ratio", type=float, default=0.9)
    parser.add_argument("--um-lambda", type=float, default=0.04)
    parser.add_argument("--sparse-vggt-sparse-ratio", type=float, default=0.5)
    parser.add_argument(
        "--sparse-vggt-cdf-threshold",
        type=float,
        default=None,
        help="Optional cumulative-attention coverage for Sparse-VGGT, e.g. 0.97.",
    )
    parser.add_argument("--da-chunk-size", type=int, default=50)
    parser.add_argument("--checkpoint", type=Path, default=VGGT_ROOT / "ckpts" / "model.pt")
    parser.add_argument(
        "--litevggt-checkpoint", type=Path,
        default=LITE_ROOT / "checkpoints" / "te_dict.pt",
        help="Released TE-remapped LiteVGGT checkpoint, relative to this workspace by default.",
    )
    parser.add_argument(
        "--come-confidence-checkpoint", type=Path,
        default=COME_ROOT / "output" / "confidence_distill" / "vggt" / "20260324_172742" / "step_02000",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def selected_images(args: argparse.Namespace) -> list[Path]:
    sequence_root = args.dataset_root / args.sequence
    if args.dataset == "scannet":
        paths = sorted(
            (sequence_root / "color").glob("*.jpg"), key=lambda path: int(path.stem)
        )
        valid_paths: list[Path] = []
        for path in paths:
            depth_path = sequence_root / "depth" / f"{path.stem}.png"
            pose_path = sequence_root / "pose" / f"{path.stem}.txt"
            if not depth_path.is_file() or not pose_path.is_file():
                continue
            try:
                pose = np.loadtxt(pose_path, dtype=np.float64)
            except (OSError, ValueError):
                continue
            # ScanNet contains occasional tracking-loss poses encoded with
            # inf/nan.  They cannot serve as geometry ground truth and must
            # be removed before fixed-stride frame selection, exactly as the
            # Omega ScanNet loader does.
            if pose.shape == (4, 4) and np.isfinite(pose).all():
                valid_paths.append(path)
        paths = valid_paths
    elif args.dataset == "nrgbd":
        pose_path = sequence_root / "poses.txt"
        poses = np.loadtxt(pose_path, dtype=np.float64)
        if poses.ndim != 2 or poses.shape[1] != 4 or poses.shape[0] % 4:
            raise ValueError(f"Invalid NRGBD poses: {pose_path}")
        poses = poses.reshape(-1, 4, 4)
        paths = []
        for path in sorted(
            (sequence_root / "images").glob("img*.png"),
            key=lambda item: int(item.stem.removeprefix("img")),
        ):
            index = int(path.stem.removeprefix("img"))
            depth = sequence_root / "depth" / f"depth{index}.png"
            if depth.is_file() and index < len(poses) and np.isfinite(poses[index]).all():
                paths.append(path)
    else:
        candidates = sorted(
            sequence_root.glob("*.color.png"),
            key=lambda path: int(path.name.split("-")[1].split(".")[0]),
        )
        paths = []
        for path in candidates:
            pose_path = path.with_name(path.name.replace(".color.png", ".pose.txt"))
            projected = path.with_name(path.name.replace(".color.png", ".depth.proj.png"))
            raw_depth = path.with_name(path.name.replace(".color.png", ".depth.png"))
            try:
                pose = np.loadtxt(pose_path, dtype=np.float64)
            except (OSError, ValueError):
                continue
            if pose.shape == (4, 4) and np.isfinite(pose).all() and (projected.is_file() or raw_depth.is_file()):
                paths.append(path)
    if args.sampling_mode == "all_stride":
        sampled = paths[::args.sampling_stride]
        if args.method == "litevggt":
            sampled = sampled[: len(sampled) // 8 * 8]
            if len(sampled) < 8:
                raise ValueError("LiteVGGT full sampling requires at least 8 valid sampled frames")
        return sampled
    if args.sampling_mode == "uniform_first_last":
        if len(paths) < args.num_frames:
            raise ValueError(f"{args.sequence} has only {len(paths)} valid RGB frames")
        indices = np.linspace(0, len(paths) - 1, args.num_frames, dtype=np.int64)
        return [paths[int(index)] for index in indices]
    required = 1 + (args.num_frames - 1) * args.sampling_stride
    if len(paths) < required:
        raise ValueError(
            f"{args.sequence} has {len(paths)} RGB frames, but {args.num_frames} frames at "
            f"stride {args.sampling_stride} require {required}."
        )
    return paths[:required:args.sampling_stride]


def load_square(paths: list[Path], image_size: int, device: torch.device) -> torch.Tensor:
    """Aspect-preserving resize to cover, then centre crop, matching Co-Me resize semantics."""
    images: list[torch.Tensor] = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        width, height = image.size
        scale = max(image_size / width, image_size / height)
        resized = image.resize((round(width * scale), round(height * scale)), Image.Resampling.BICUBIC)
        left = (resized.width - image_size) // 2
        top = (resized.height - image_size) // 2
        resized = resized.crop((left, top, left + image_size, top + image_size))
        pixels = torch.from_numpy(np.asarray(resized).copy()).permute(2, 0, 1)
        images.append(pixels.float().div_(255.0))
    return torch.stack(images).unsqueeze(0).to(device=device, dtype=torch.bfloat16)


def load_native(method: str, checkpoint: Path, device: torch.device, args: argparse.Namespace):
    # CoMe ships a second, incompatible top-level ``vggt`` package below its
    # thirdparty directory.  Some editable installs leave that package cached
    # in sys.modules, even when VGGT_ROOT is first in PYTHONPATH.  Native VGGT
    # accelerators must never instantiate that CoMe adapter: it expects a
    # MultiViewInput and does not implement forward_da_vggt.
    sys.path.insert(0, str(VGGT_ROOT))
    for module_name in tuple(sys.modules):
        if module_name == "vggt" or module_name.startswith("vggt."):
            module_file = getattr(sys.modules[module_name], "__file__", None)
            if module_file is not None and not Path(module_file).resolve().is_relative_to(VGGT_ROOT):
                del sys.modules[module_name]
    importlib.invalidate_caches()
    from vggt.evaluation import load_model

    model = load_model(
        checkpoint,
        device,
        enable_camera=True,
        enable_depth=True,
        inter_frame_attention="global",
        enable_token_merging=method == "fastvggt",
        token_merging_ratio=args.merge_ratio,
        token_merging_method="spatial",
        um_lambda_cost=args.um_lambda if method == "u-m" else None,
        avggt_subsample_factor=args.avggt_subsample_factor if method == "avggt" else None,
        model_bfloat16=True,
    )
    model_source = Path(inspect.getfile(type(model))).resolve()
    if not model_source.is_relative_to(VGGT_ROOT):
        raise RuntimeError(f"Native VGGT loader resolved the wrong class: {model_source}")
    if method == "sparse-vggt":
        sparse_checkout = Path(os.environ.get("SPARSE_VGGT_ROOT", str(WORKSPACE_ROOT / "sparse-vggt")))
        sparse_root = sparse_checkout / "src"
        # The original Sparse-VGGT checkout carries an older SpargeAttn build
        # which only has Ada/Hopper cubins.  Prefer a separately built
        # Blackwell-compatible checkout when SPARGEATTN_ROOT is supplied.
        sparge_root = Path(os.environ.get(
            "SPARGEATTN_ROOT", str(sparse_checkout / "external" / "SpargeAttn")
        ))
        if not sparse_root.is_dir() or not sparge_root.is_dir():
            raise FileNotFoundError(f"Sparse-VGGT checkout is incomplete: {sparse_checkout}")
        sys.path.insert(0, str(sparse_root))
        sys.path.insert(0, str(sparge_root))
        # Avoid retaining an already imported extension from the old editable
        # package when this process happens to share interpreter state.
        for module_name in tuple(sys.modules):
            if module_name == "spas_sage_attn" or module_name.startswith("spas_sage_attn."):
                del sys.modules[module_name]
        from sparse_vggt.models.vggt import sparse_aggregator_from_vggt

        model.aggregator, _ = sparse_aggregator_from_vggt(
            model.aggregator, sparse_ratio=args.sparse_vggt_sparse_ratio,
            cdf_threshold=args.sparse_vggt_cdf_threshold, pool_mode="avg", verbose=True,
        )
    return model.eval()


def load_litevggt(checkpoint: Path, image_size: int, device: torch.device):
    if not LITE_ROOT.is_dir():
        raise FileNotFoundError(f"LiteVGGT repository is missing: {LITE_ROOT}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"LiteVGGT checkpoint is missing: {checkpoint}")
    sys.path.insert(0, str(LITE_ROOT))
    from vggt.models.vggt import VGGT

    # Keep this identical to LiteVGGT's official demo loader.  Its released
    # TE-remapped checkpoint intentionally has a few framework-specific
    # ``_extra_state`` entries absent, so ``strict=False`` is required.
    model = VGGT().to(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            "LiteVGGT checkpoint loaded with strict=False "
            f"(missing={len(incompatible.missing_keys)}, "
            f"unexpected={len(incompatible.unexpected_keys)})."
        )
    model.update_patch_dimensions(image_size // 14, image_size // 14)
    return model.to(dtype=torch.bfloat16).eval()


def load_come(
    checkpoint: Path, confidence_checkpoint: Path, device: torch.device,
):
    if not COME_ROOT.is_dir():
        raise FileNotFoundError(f"CoMe repository is missing: {COME_ROOT}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"VGGT checkpoint is missing: {checkpoint}")
    if not (confidence_checkpoint / "checkpoint.pth").is_file():
        raise FileNotFoundError(f"CoMe confidence checkpoint is missing: {confidence_checkpoint}")
    sys.path.insert(0, str(COME_ROOT))
    from src.accelerate.token_merger.co_me_2d import CoMe_2D_TokenMerger
    from src.accelerate.vggt.fused import fused_accelerate
    from src.thirdparty.vggt.models.vggt import VGGT

    # CoMe replaces attention blocks during fused_accelerate().  Those blocks
    # assert eval mode at construction time, so set it before patching rather
    # than only on the returned model.
    model = VGGT().eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=False)
    unexpected = [key for key in incompatible.unexpected_keys if not key.startswith("track_head.")]
    if incompatible.missing_keys or unexpected:
        raise RuntimeError(
            f"CoMe VGGT checkpoint mismatch: missing={incompatible.missing_keys}, unexpected={unexpected}"
        )
    merger = CoMe_2D_TokenMerger(
        start_idx=model.patch_start_index,
        group_h=3,
        group_w=3,
        threshold=0.5,
        ckpt_path=confidence_checkpoint,
        device=device,
    )
    # The released CoMe fused/Jagged path is float32-only.  In particular,
    # forcing its patched blocks to bf16 causes a Float/BFloat16 mismatch in
    # the custom attention path.
    return fused_accelerate(model, merger, encoder_layer=14).to(device=device).eval()


def run_model(method: str, model: Any, images: torch.Tensor, da_chunk_size: int) -> Any:
    if method == "come":
        from src.interface.geometric_model import MultiViewInput
        # CoMe's public interface is jaxtyping-checked and requires a
        # float32 input tensor.  The surrounding autocast context still runs
        # the model kernels in bf16 where appropriate.
        return model(MultiViewInput(images=images.float(), intrinsics=None))
    if method == "da-vggt":
        return model.forward_da_vggt(images, chunk_size=da_chunk_size)
    return model(images)


def release_warmup_cuda_state(model: Any) -> None:
    """Release method-owned transient tensors before the measured forward.

    The 1000-frame run is close to device capacity.  Deleting the returned
    dictionary alone is insufficient because U-M and a few patched backends
    keep plans/statistics on modules, and PyTorch's caching allocator keeps
    the freed blocks reserved.  None of these objects are model parameters or
    persistent inference state, so clearing them between independent forwards
    is safe and makes warmup memory-neutral.
    """
    for module in model.modules():
        if hasattr(module, "_um_plan"):
            module._um_plan = None
        for attribute in ("last_token_merging_stats", "last_frame_merge_stats"):
            value = getattr(module, attribute, None)
            if isinstance(value, list):
                value.clear()
    gc.collect()
    torch.cuda.empty_cache()


def output_summary(method: str, output: Any) -> dict[str, Any]:
    if method == "come":
        return {
            "depth_shape": list(output.depths.shape) if output.depths is not None else None,
            "pose_shape": list(output.poses.shape) if output.poses is not None else None,
        }
    return {
        "depth_shape": list(output["depth"].shape) if "depth" in output else None,
        "pose_shape": list(output["pose_enc"].shape) if "pose_enc" in output else (
            list(output["da_w2c"].shape) if "da_w2c" in output else None
        ),
    }


def _resize_cover_crop(array: np.ndarray, image_size: int) -> np.ndarray:
    """Apply the RGB cover-resize/centre-crop transform to one depth map."""
    height, width = array.shape
    scale = max(image_size / width, image_size / height)
    resized_w, resized_h = round(width * scale), round(height * scale)
    resized = np.asarray(Image.fromarray(array).resize((resized_w, resized_h), Image.Resampling.NEAREST))
    left, top = (resized_w - image_size) // 2, (resized_h - image_size) // 2
    return resized[top:top + image_size, left:left + image_size]


def _resize_nearest(array: np.ndarray, width: int, height: int) -> np.ndarray:
    if array.shape == (height, width):
        return array
    return np.asarray(Image.fromarray(array).resize((width, height), Image.Resampling.NEAREST))


def _7scenes_ground_truth(paths: list[Path], image_size: int, output_hw: tuple[int, int]) -> np.ndarray:
    """Read metre-depth GT and transform it exactly as the corresponding RGB input."""
    height, width = output_hw
    maps: list[np.ndarray] = []
    for image_path in paths:
        depth_path = image_path.with_name(image_path.name.replace(".color.png", ".depth.proj.png"))
        if not depth_path.is_file():
            depth_path = image_path.with_name(image_path.name.replace(".color.png", ".depth.png"))
        if not depth_path.is_file():
            raise FileNotFoundError(f"7Scenes depth annotation is missing: {depth_path}")
        raw = np.asarray(Image.open(depth_path))
        depth = raw.astype(np.float32) / 1000.0
        depth[(raw == 0) | (raw == 65535)] = -1.0
        maps.append(_resize_nearest(_resize_cover_crop(depth, image_size), width, height))
    return np.stack(maps)


def _scannet_ground_truth(paths: list[Path], image_size: int, output_hw: tuple[int, int]) -> np.ndarray:
    """Read ScanNet's millimetre depth with the exact RGB cover/crop transform."""
    height, width = output_hw
    maps: list[np.ndarray] = []
    for image_path in paths:
        depth_path = image_path.parent.parent / "depth" / f"{image_path.stem}.png"
        if not depth_path.is_file():
            raise FileNotFoundError(f"ScanNet depth annotation is missing: {depth_path}")
        raw = np.asarray(Image.open(depth_path), dtype=np.uint16)
        depth = raw.astype(np.float32) / 1000.0
        depth[raw == 0] = -1.0
        maps.append(_resize_nearest(_resize_cover_crop(depth, image_size), width, height))
    return np.stack(maps)


def _nrgbd_ground_truth(paths: list[Path], image_size: int, output_hw: tuple[int, int]) -> np.ndarray:
    height, width = output_hw
    maps: list[np.ndarray] = []
    for image_path in paths:
        index = image_path.stem.removeprefix("img")
        depth_path = image_path.parent.parent / "depth" / f"depth{index}.png"
        raw = np.asarray(Image.open(depth_path), dtype=np.uint16)
        depth = raw.astype(np.float32) / 1000.0
        depth[(raw == 0) | (raw == 65535)] = -1.0
        maps.append(_resize_nearest(_resize_cover_crop(depth, image_size), width, height))
    return np.stack(maps)


def _depth_metrics_per_frame_irls(predictions: np.ndarray, ground_truth: np.ndarray) -> dict[str, float]:
    """Independent robust scale/shift depth alignment, then pixel-weighted scoring."""
    if predictions.shape != ground_truth.shape:
        raise ValueError(f"Depth shape mismatch: {predictions.shape} vs {ground_truth.shape}")
    sums = {name: 0.0 for name in ("abs_rel", "sq_rel", "squared", "log_squared", "mae", "d1", "d2", "d3")}
    valid_total = 0
    for prediction, gt in zip(predictions, ground_truth, strict=True):
        valid = np.isfinite(prediction) & np.isfinite(gt) & (gt > 0) & (gt < 10.0)
        if not valid.any():
            raise ValueError("No valid 7Scenes depth pixels in a selected frame")
        x = prediction[valid].astype(np.float64); y = gt[valid].astype(np.float64)
        scale = float(np.median(y) / max(np.median(x), 1e-8)); shift = 0.0
        for _ in range(20):
            residual = scale * x + shift - y
            weights = 1.0 / np.maximum(np.abs(residual), 1e-4)
            mean_x = float((weights * x).sum() / weights.sum())
            mean_y = float((weights * y).sum() / weights.sum())
            centered_x = x - mean_x
            variance = float((weights * centered_x * centered_x).sum())
            if variance > np.finfo(np.float64).eps:
                scale = float((weights * centered_x * (y - mean_y)).sum() / variance)
            shift = mean_y - scale * mean_x
        aligned = np.clip(scale * x + shift, 1e-5, 10.0)
        diff = aligned - y; ratio = np.maximum(aligned / y, y / aligned); count = len(y)
        sums["abs_rel"] += float((np.abs(diff) / y).sum()); sums["sq_rel"] += float((diff * diff / y).sum())
        sums["squared"] += float((diff * diff).sum()); sums["log_squared"] += float((np.log(aligned / y) ** 2).sum())
        sums["mae"] += float(np.abs(diff).sum()); sums["d1"] += float((ratio < 1.25).sum())
        sums["d2"] += float((ratio < 1.25**2).sum()); sums["d3"] += float((ratio < 1.25**3).sum()); valid_total += count
    return {
        "abs_rel": sums["abs_rel"] / valid_total, "sq_rel": sums["sq_rel"] / valid_total,
        "rmse_m": float(np.sqrt(sums["squared"] / valid_total)), "rmse_log": float(np.sqrt(sums["log_squared"] / valid_total)),
        "mae_m": sums["mae"] / valid_total, "delta_1_25_percent": 100 * sums["d1"] / valid_total,
        "delta_1_25_sq_percent": 100 * sums["d2"] / valid_total,
        "delta_1_25_cu_percent": 100 * sums["d3"] / valid_total, "valid_depth_pixels": float(valid_total),
    }


def _quaternion_xyzw_to_matrix(quaternions: np.ndarray) -> np.ndarray:
    """Convert CoMe camera-to-world xyzw quaternions to rotation matrices."""
    q = np.asarray(quaternions, dtype=np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    x, y, z, w = (q[..., index] for index in range(4))
    result = np.empty((*q.shape[:-1], 3, 3), dtype=np.float64)
    result[..., 0, 0] = 1 - 2 * (y * y + z * z); result[..., 0, 1] = 2 * (x * y - z * w); result[..., 0, 2] = 2 * (x * z + y * w)
    result[..., 1, 0] = 2 * (x * y + z * w); result[..., 1, 1] = 1 - 2 * (x * x + z * z); result[..., 1, 2] = 2 * (y * z - x * w)
    result[..., 2, 0] = 2 * (x * z - y * w); result[..., 2, 1] = 2 * (y * z + x * w); result[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return result


def output_geometry(method: str, output: Any, images: torch.Tensor) -> tuple[np.ndarray, np.ndarray | None]:
    """Normalise backend output to [F,H,W] depth and [F,4,4] c2w poses."""
    if method == "come":
        if output.depths is None or output.poses is None:
            raise ValueError("CoMe did not return both depth and camera poses")
        depths = output.depths[0, :, 0].float().cpu().numpy(); packed = output.poses[0].float().cpu().numpy()
        poses = np.repeat(np.eye(4, dtype=np.float64)[None], len(packed), axis=0)
        poses[:, :3, :3] = _quaternion_xyzw_to_matrix(packed[:, 3:]); poses[:, :3, 3] = packed[:, :3]
        return depths, poses
    depths = output["depth"][0, ..., 0].float().cpu().numpy()
    if method == "da-vggt" and "da_w2c" in output:
        w2c = output["da_w2c"][0].float().cpu().numpy()
        bottom = np.broadcast_to(np.array([0.0, 0.0, 0.0, 1.0]), (len(w2c), 1, 4))
        return depths, np.linalg.inv(np.concatenate((w2c, bottom), axis=1))
    if "pose_enc" not in output:
        return depths, None
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    w2c, _ = pose_encoding_to_extri_intri(output["pose_enc"].float(), images.shape[-2:])
    bottom = torch.zeros((*w2c.shape[:-2], 1, 4), device=w2c.device, dtype=w2c.dtype); bottom[..., 0, 3] = 1.0
    return depths, torch.linalg.inv(torch.cat((w2c, bottom), dim=-2)[0]).float().cpu().numpy()


def _crop_intrinsics(image_path: Path, image_size: int, output_hw: tuple[int, int], frames: int) -> np.ndarray:
    """Transform the canonical 7Scenes K through the cover/crop preprocessing."""
    source_w, source_h = Image.open(image_path).size; scale = max(image_size / source_w, image_size / source_h)
    left, top = (round(source_w * scale) - image_size) // 2, (round(source_h * scale) - image_size) // 2
    k = np.array([[525.0 * scale, 0.0, 320.0 * scale - left], [0.0, 525.0 * scale, 240.0 * scale - top], [0.0, 0.0, 1.0]])
    height, width = output_hw; k[0, :] *= width / image_size; k[1, :] *= height / image_size; k[2, 2] = 1.0
    return np.repeat(k[None], frames, axis=0)


def _7scenes_poses(paths: list[Path]) -> np.ndarray:
    """Read c2w ground truth without importing the native VGGT evaluator.

    LiteVGGT intentionally provides its own top-level ``vggt`` package, which
    does not include ``vggt.evaluation``.  Directly reading the public 7Scenes
    pose sidecars avoids importing inference.infer_vggt and keeps all isolated
    method environments evaluable.
    """
    pose_paths = [path.with_name(path.name.replace(".color.png", ".pose.txt")) for path in paths]
    missing = [str(path) for path in pose_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"7Scenes pose annotations are missing: {missing[:3]}")
    return np.stack([np.loadtxt(path, dtype=np.float64) for path in pose_paths])


def _scannet_poses(paths: list[Path]) -> np.ndarray:
    pose_paths = [path.parent.parent / "pose" / f"{path.stem}.txt" for path in paths]
    missing = [str(path) for path in pose_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"ScanNet pose annotations are missing: {missing[:3]}")
    poses = np.stack([np.loadtxt(path, dtype=np.float64) for path in pose_paths])
    if poses.shape[1:] != (4, 4) or not np.isfinite(poses).all():
        raise ValueError("ScanNet poses must be finite 4x4 camera-to-world matrices")
    return poses


def _nrgbd_poses(paths: list[Path]) -> np.ndarray:
    values = np.loadtxt(paths[0].parent.parent / "poses.txt", dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4 or values.shape[0] % 4:
        raise ValueError("Expected 4N x 4 NRGBD poses")
    poses = values.reshape(-1, 4, 4)
    poses[:, :3, 1:3] *= -1.0
    indices = np.asarray([int(path.stem.removeprefix("img")) for path in paths], dtype=np.int64)
    return poses[indices]


def _scannet_intrinsics(image_path: Path, image_size: int, output_hw: tuple[int, int], frames: int) -> np.ndarray:
    sequence_root = image_path.parent.parent
    intrinsic_path = sequence_root / "intrinsic" / "intrinsic_color.txt"
    if not intrinsic_path.is_file():
        raise FileNotFoundError(f"ScanNet intrinsics are missing: {intrinsic_path}")
    source_w, source_h = Image.open(image_path).size
    scale = max(image_size / source_w, image_size / source_h)
    resized_w, resized_h = round(source_w * scale), round(source_h * scale)
    left, top = (resized_w - image_size) // 2, (resized_h - image_size) // 2
    k = np.loadtxt(intrinsic_path, dtype=np.float64)[:3, :3].copy()
    k[0, :] *= scale; k[1, :] *= scale
    k[0, 2] -= left; k[1, 2] -= top
    height, width = output_hw
    k[0, :] *= width / image_size; k[1, :] *= height / image_size; k[2, 2] = 1.0
    return np.repeat(k[None], frames, axis=0)


def evaluate_dataset(method: str, output: Any, images: torch.Tensor, paths: list[Path], args: argparse.Namespace) -> dict[str, Any]:
    """Metrics are intentionally outside the CUDA timing window."""
    from geometry_eval import depth_to_world_points, evaluate_pi3_geometry, trajectory_pose_metrics
    from inference.pose_auc import evaluate_pose_auc

    predicted_depth, predicted_c2w = output_geometry(method, output, images)
    if args.dataset == "scannet":
        gt_depth = _scannet_ground_truth(paths, args.image_size, predicted_depth.shape[-2:])
        gt_c2w = _scannet_poses(paths)
        intrinsics = _scannet_intrinsics(paths[0], args.image_size, predicted_depth.shape[-2:], len(predicted_depth))
    elif args.dataset == "nrgbd":
        gt_depth = _nrgbd_ground_truth(paths, args.image_size, predicted_depth.shape[-2:])
        gt_c2w = _nrgbd_poses(paths)
        intrinsics = _crop_intrinsics(paths[0], args.image_size, predicted_depth.shape[-2:], len(predicted_depth))
    else:
        gt_depth = _7scenes_ground_truth(paths, args.image_size, predicted_depth.shape[-2:])
        gt_c2w = _7scenes_poses(paths)
        intrinsics = _crop_intrinsics(paths[0], args.image_size, predicted_depth.shape[-2:], len(predicted_depth))
    metrics: dict[str, Any] = _depth_metrics_per_frame_irls(predicted_depth, gt_depth)
    if predicted_c2w is None:
        raise ValueError("Model did not provide camera poses")
    pose_auc = evaluate_pose_auc(predicted_c2w, gt_c2w)
    metrics.update({"auc_3_percent": pose_auc["AUC@3"], "auc_5_percent": pose_auc["AUC@5"], "auc_10_percent": pose_auc["AUC@10"], "auc_15_percent": pose_auc["AUC@15"], "auc_30_percent": pose_auc["AUC@30"], **trajectory_pose_metrics(predicted_c2w, gt_c2w)})
    valid = np.isfinite(predicted_depth) & (predicted_depth > 0) & (gt_depth > 0) & (gt_depth < 10.0)
    metrics.update(evaluate_pi3_geometry(depth_to_world_points(predicted_depth, predicted_c2w, intrinsics), depth_to_world_points(gt_depth, gt_c2w, intrinsics), valid))
    return metrics


def main() -> int:
    args = parse_args()
    if args.image_size % 42:
        raise ValueError("--image-size must be divisible by 42 (14 px patches and Co-Me 3x3 groups)")
    if args.timing_repeats < 1:
        raise ValueError("--timing-repeats must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    paths = selected_images(args)
    actual_frames = len(paths)
    images = load_square(paths, args.image_size, device)

    result: dict[str, Any] = {
        "model": "VGGT",
        "dataset": args.dataset,
        "method": args.method,
        "sequence": args.sequence,
        "frames": actual_frames,
        "sampling_stride": args.sampling_stride,
        "sampling_mode": args.sampling_mode,
        "image_preprocess": f"aspect_preserving_center_crop_{args.image_size}x{args.image_size}",
        "timing": "one_untimed_warmup_then_cuda_event_median",
        "timing_repeats": args.timing_repeats,
        "litevggt_checkpoint": str(args.litevggt_checkpoint) if args.method == "litevggt" else None,
        "come_confidence_checkpoint": str(args.come_confidence_checkpoint) if args.method == "come" else None,
        "avggt_subsample_factor": args.avggt_subsample_factor if args.method == "avggt" else None,
        "merge_ratio": args.merge_ratio if args.method == "fastvggt" else None,
        "um_lambda": args.um_lambda if args.method == "u-m" else None,
        "sparse_vggt_sparse_ratio": args.sparse_vggt_sparse_ratio if args.method == "sparse-vggt" else None,
        "sparse_vggt_cdf_threshold": args.sparse_vggt_cdf_threshold if args.method == "sparse-vggt" else None,
        "da_chunk_size": args.da_chunk_size if args.method == "da-vggt" else None,
    }
    phase = "model_load"
    try:
        if args.method in ("baseline", "fastvggt", "da-vggt", "sparse-vggt", "u-m", "avggt"):
            model = load_native(args.method, args.checkpoint, device, args)
        elif args.method == "litevggt":
            model = load_litevggt(args.litevggt_checkpoint, args.image_size, device)
        else:
            model = load_come(args.checkpoint, args.come_confidence_checkpoint, device)
        # This process is normally fresh, but explicit cleanup also covers
        # temporary CPU->CUDA checkpoint-loading allocations.
        gc.collect()
        torch.cuda.empty_cache()
        phase = "warmup_forward"
        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=args.method != "come"
        ):
            warmup_output = run_model(args.method, model, images, args.da_chunk_size)
        torch.cuda.synchronize(device)
        del warmup_output
        phase = "warmup_cleanup"
        release_warmup_cuda_state(model)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        durations: list[float] = []
        output: Any = None
        for _ in range(args.timing_repeats):
            if output is not None:
                del output
                output = None
                release_warmup_cuda_state(model)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            phase = "measured_forward"
            start.record()
            with torch.inference_mode(), torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=args.method != "come"
            ):
                output = run_model(args.method, model, images, args.da_chunk_size)
            end.record()
            end.synchronize()
            durations.append(float(start.elapsed_time(end)))
        # Snapshot VRAM before evaluation.  Open3D/NumPy work below is not a
        # part of model inference and must never contaminate the VRAM timing.
        latency_ms = float(np.median(durations))
        runtime = {
            "model_latency_ms_mean": latency_ms,
            "fps": actual_frames / (latency_ms / 1000.0),
            "peak_allocated_gib_max": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_gib_max": torch.cuda.max_memory_reserved(device) / 2**30,
        }
        phase = "metric_evaluation"
        accuracy = evaluate_dataset(args.method, output, images, paths, args)
        overall = {**accuracy, **runtime}
        result.update(
            success=True,
            # Preserve the original top-level fields for the smoke-test
            # consumers, while the full, table-ready record lives in overall.
            model_latency_ms=runtime["model_latency_ms_mean"],
            fps=runtime["fps"],
            peak_allocated_gib=runtime["peak_allocated_gib_max"],
            peak_reserved_gib=runtime["peak_reserved_gib_max"],
            **output_summary(args.method, output),
            evaluation_protocol={
                "dataset": args.dataset, "depth_alignment": "per_frame_irls_scale_shift_20_iterations",
                "depth_gt_preprocess": "same_cover_resize_center_crop_as_rgb",
                "pose": "official_relative_pose_auc_plus_sim3_trajectory",
                "geometry": "pi3_sim3_icp_bidirectional_point_cloud",
                "timing_excludes_metric_computation": True,
            },
            overall=overall,
            per_sequence=[{"sequence": args.sequence, **overall}],
        )
    except torch.cuda.OutOfMemoryError as exc:
        result.update(
            success=False,
            error="CUDA out of memory",
            detail=str(exc),
            oom_phase=phase,
            peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30,
        )
    except Exception as exc:
        # Persist the complete stack for per-method smoke-test diagnosis.
        # The one-line exception message alone is ambiguous when several
        # optional acceleration backends are imported in distinct processes.
        result.update(
            success=False,
            error=type(exc).__name__,
            detail=str(exc),
            traceback=traceback.format_exc(),
        )
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result.get("success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
