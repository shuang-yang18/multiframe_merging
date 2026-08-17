from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from pi3.models.pi3 import Pi3
from pi3.utils.geometry import se3_inverse


DEFAULT_ROOTS = {
    "tum_dynamic": "/data/mmc_syang/dataset/TUM-Dynamics",
    "7scenes": "/data/mmc_syang/dataset/7scenes/test",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["tum_dynamic", "7scenes"], required=True)
    parser.add_argument("--dataset-root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pretrained", default="yyfz233/Pi3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-frames-per-seq", type=int, default=300)
    parser.add_argument("--load-img-size", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--token-merging-method", choices=["none", "fastvggt", "frame_persistent_spatial"], default="none")
    parser.add_argument("--token-merging-ratio", type=float, default=0.9)
    parser.add_argument("--token-merging-frame-alpha", type=float, default=0.1)
    parser.add_argument("--token-merging-frame-segment-threshold", type=float, default=0.9)
    parser.add_argument("--token-merging-frame-merge-threshold", type=float, default=0.1)
    parser.add_argument("--token-merging-frame-max-window", type=int, default=20)
    parser.add_argument("--token-merging-frame-pool-stride", type=int, default=2)
    parser.add_argument("--token-merging-frame-multi-max-group-size", type=int, default=4)
    parser.add_argument("--token-merging-frame-multi-pair-threshold", type=float, default=0.95)
    parser.add_argument("--token-merging-frame-multi-span-threshold", type=float, default=0.93)
    return parser.parse_args()


def list_sequences(dataset: str, root: Path) -> list[str]:
    if dataset == "tum_dynamic":
        return sorted(path.name for path in root.iterdir() if (path / "rgb").is_dir())
    return sorted(
        f"{scene.name}/{seq.name}"
        for scene in root.iterdir()
        if scene.is_dir()
        for seq in scene.iterdir()
        if seq.is_dir() and any(seq.glob("*.color.png"))
    )


def sequence_images(dataset: str, root: Path, seq: str) -> list[Path]:
    if dataset == "tum_dynamic":
        return sorted((root / seq / "rgb").glob("*.png"))
    return sorted((root / seq).glob("*.color.png"))


def sequence_depths(dataset: str, root: Path, seq: str) -> list[Path]:
    if dataset == "tum_dynamic":
        return sorted((root / seq / "depth").glob("*.png"))
    return sorted((root / seq).glob("*.depth.proj.png"))


def sequence_poses(dataset: str, root: Path, seq: str, image_paths: list[Path]) -> np.ndarray | None:
    if dataset == "7scenes":
        poses = [np.loadtxt(str(path).replace(".color.png", ".pose.txt")) for path in image_paths]
        return np.stack(poses).astype(np.float64)
    gt_file = root / seq / "groundtruth.txt"
    if not gt_file.is_file():
        gt_file = root / seq / "groundtruth_90.txt"
    if not gt_file.is_file():
        return None
    rows = []
    with gt_file.open() as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append([float(x) for x in line.split()])
    if not rows:
        return None
    stamps = np.array([r[0] for r in rows])
    mats = np.stack([_tum_row_to_c2w(r) for r in rows])
    image_stamps = np.array([float(path.stem) for path in image_paths])
    indices = np.abs(stamps[:, None] - image_stamps[None, :]).argmin(axis=0)
    return mats[indices].astype(np.float64)


def _tum_row_to_c2w(row):
    _, tx, ty, tz, qx, qy, qz, qw = row
    x, y, z, w = qx, qy, qz, qw
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = rot
    mat[:3, 3] = [tx, ty, tz]
    return mat


def read_depth(dataset: str, path: Path) -> np.ndarray:
    depth_png = np.asarray(Image.open(path))
    if dataset == "tum_dynamic":
        depth = depth_png.astype(np.float32) / 5000.0
    else:
        depth = depth_png.astype(np.float32) / 1000.0
        depth[depth_png == 65535] = 0.0
    depth[depth <= 1e-6] = np.nan
    return depth


def load_images(paths: list[Path], new_width: int, device: torch.device):
    sources = [Image.open(path).convert("RGB") for path in paths]
    width, height = sources[0].size
    target_width = new_width
    target_height = round(height * (new_width / width) / 14) * 14
    arrays = []
    for image in sources:
        image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        arr = np.asarray(image).astype(np.float32) / 255.0
        arrays.append(torch.from_numpy(arr).permute(2, 0, 1))
    return torch.stack(arrays, dim=0).to(device).unsqueeze(0)


def depth_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    mask = np.isfinite(gt) & np.isfinite(pred) & (gt > 1e-6) & (pred > 1e-6)
    if mask.sum() == 0:
        return {"valid_pixels": 0}
    gt_v = gt[mask].astype(np.float64)
    pred_v = pred[mask].astype(np.float64)
    pred_v = _scale_shift_align(pred_v, gt_v)
    pred_v = np.maximum(pred_v, 1e-6)
    ratio = np.maximum(pred_v / gt_v, gt_v / pred_v)
    return {
        "Abs Rel": float(np.mean(np.abs(pred_v - gt_v) / gt_v)),
        "Sq Rel": float(np.mean((pred_v - gt_v) ** 2 / gt_v)),
        "RMSE": float(np.sqrt(np.mean((pred_v - gt_v) ** 2))),
        "Log RMSE": float(np.sqrt(np.mean((np.log(pred_v) - np.log(gt_v)) ** 2))),
        "delta < 1.25": float(np.mean(ratio < 1.25)),
        "delta < 1.25^2": float(np.mean(ratio < 1.25**2)),
        "delta < 1.25^3": float(np.mean(ratio < 1.25**3)),
        "valid_pixels": int(mask.sum()),
    }


def _scale_shift_align(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    A = np.stack([pred, np.ones_like(pred)], axis=1)
    scale, shift = np.linalg.lstsq(A, gt, rcond=None)[0]
    return pred * scale + shift


def evaluate_pose_auc(pred: np.ndarray, gt: np.ndarray) -> dict:
    count = min(len(pred), len(gt))
    pred = np.asarray(pred[:count], dtype=np.float64)
    gt = np.asarray(gt[:count], dtype=np.float64)
    scale, rotation, translation = _umeyama_alignment(pred[:, :3, 3], gt[:, :3, 3])
    aligned = pred.copy()
    aligned[:, :3, :3] = rotation @ pred[:, :3, :3]
    aligned[:, :3, 3] = scale * (rotation @ pred[:, :3, 3].T).T + translation

    pose_errors = []
    rotation_errors = []
    translation_errors = []
    for i in range(count):
        for j in range(i + 1, count):
            pred_rel_rotation = aligned[j, :3, :3].T @ aligned[i, :3, :3]
            gt_rel_rotation = gt[j, :3, :3].T @ gt[i, :3, :3]
            rot_error = _rotation_angle(pred_rel_rotation @ gt_rel_rotation.T)
            trans_error = _translation_angle(
                aligned[j, :3, 3] - aligned[i, :3, 3],
                gt[j, :3, 3] - gt[i, :3, 3],
            )
            rotation_errors.append(rot_error)
            if trans_error is None:
                pose_errors.append(rot_error)
            else:
                translation_errors.append(trans_error)
                pose_errors.append(max(rot_error, trans_error))
    return {
        "AUC@3": _auc(pose_errors, 3.0),
        "AUC@30": _auc(pose_errors, 30.0),
        "pairs": int(len(pose_errors)),
        "mean_pose_error_deg": float(np.mean(pose_errors)) if pose_errors else 0.0,
        "median_pose_error_deg": float(np.median(pose_errors)) if pose_errors else 0.0,
        "rotation_errors_deg": [float(v) for v in rotation_errors],
        "translation_errors_deg": [float(v) for v in translation_errors],
        "pose_errors_deg": [float(v) for v in pose_errors],
    }


def _rotation_angle(rot: np.ndarray) -> float:
    cos = np.clip((np.trace(rot) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def _translation_angle(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return None
    cos = np.clip(float(np.dot(a, b) / (na * nb)), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def _auc(errors, threshold):
    if not errors:
        return 0.0
    values = np.asarray(errors, dtype=np.float64)
    return float(np.mean(np.maximum(1.0 - values / threshold, 0.0)) * 100.0)


def _umeyama_alignment(src: np.ndarray, dst: np.ndarray):
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_zero = src - src_mean
    dst_zero = dst - dst_mean
    cov = dst_zero.T @ src_zero / max(len(src), 1)
    u, singular_values, vt = np.linalg.svd(cov)
    sign = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[-1] = -1.0
    rotation = u @ np.diag(sign) @ vt
    variance = np.sum(src_zero * src_zero) / max(len(src), 1)
    scale = np.sum(singular_values * sign) / max(variance, 1e-12)
    translation = dst_mean - scale * (rotation @ src_mean)
    return float(scale), rotation, translation


def summarize_pose(rows: list[dict]) -> dict:
    pose_errors = [v for row in rows for v in row.get("pose_errors_deg", [])]
    rotation_errors = [v for row in rows for v in row.get("rotation_errors_deg", [])]
    translation_errors = [v for row in rows for v in row.get("translation_errors_deg", [])]
    return {
        "AUC@3": _auc(pose_errors, 3.0),
        "AUC@30": _auc(pose_errors, 30.0),
        "pairs": int(len(pose_errors)),
        "mean_pose_error_deg": float(np.mean(pose_errors)) if pose_errors else 0.0,
        "median_pose_error_deg": float(np.median(pose_errors)) if pose_errors else 0.0,
        "mean_rotation_error_deg": float(np.mean(rotation_errors)) if rotation_errors else 0.0,
        "mean_translation_error_deg": float(np.mean(translation_errors)) if translation_errors else 0.0,
    }


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    root = Path(args.dataset_root or DEFAULT_ROOTS[args.dataset])
    output_root = Path(args.output_dir) / args.dataset
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    enable = args.token_merging_method != "none"
    model = Pi3.from_pretrained(
        args.pretrained,
        enable_token_merging=enable,
        token_merging_method=args.token_merging_method,
        token_merging_ratio=args.token_merging_ratio,
        token_merging_frame_alpha=args.token_merging_frame_alpha,
        token_merging_frame_segment_threshold=args.token_merging_frame_segment_threshold,
        token_merging_frame_merge_threshold=args.token_merging_frame_merge_threshold,
        token_merging_frame_max_window=args.token_merging_frame_max_window,
        token_merging_frame_pool_stride=args.token_merging_frame_pool_stride,
        token_merging_frame_multi_max_group_size=args.token_merging_frame_multi_max_group_size,
        token_merging_frame_multi_pair_threshold=args.token_merging_frame_multi_pair_threshold,
        token_merging_frame_multi_span_threshold=args.token_merging_frame_multi_span_threshold,
    ).to(device).eval()

    sequences = list_sequences(args.dataset, root)
    depth_rows = []
    pose_rows = []
    total_time = 0.0
    total_frames = 0
    frame_merge_stats = []
    token_merging_stats = []
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16

    for seq in tqdm(sequences, desc=f"Pi3 {args.dataset}"):
        seq_out = output_root / seq
        complete_file = seq_out / "_metrics.json"
        if complete_file.is_file() and not args.overwrite:
            with complete_file.open() as handle:
                cached = json.load(handle)
            depth_rows.append(cached["depth"])
            if cached.get("pose"):
                pose_rows.append(cached["pose"])
            total_time += cached["speed"]["time"]
            total_frames += cached["speed"]["frames"]
            continue

        image_paths = sequence_images(args.dataset, root, seq)[: args.max_frames_per_seq]
        depth_paths = sequence_depths(args.dataset, root, seq)[: len(image_paths)]
        gt_poses = sequence_poses(args.dataset, root, seq, image_paths)
        images = load_images(image_paths, args.load_img_size, device)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        with torch.inference_mode(), torch.amp.autocast(device.type, dtype=dtype):
            pred = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

        pred_depth = pred["local_points"][0, ..., -1].detach().float().cpu().numpy()
        pred_poses = pred["camera_poses"][0].detach().float().cpu()
        pred_poses = pred_poses.numpy()
        gt_depths = np.stack([read_depth(args.dataset, path) for path in depth_paths])
        pred_depth = np.stack(
            [cv2.resize(pred_depth[i], (gt_depths.shape[2], gt_depths.shape[1]), interpolation=cv2.INTER_CUBIC)
             for i in range(len(pred_depth))]
        )

        seq_metrics = depth_metrics(pred_depth, gt_depths)
        seq_metrics.update({"dataset": args.dataset, "seq": seq, "frames": len(image_paths), "fps": len(image_paths) / elapsed})
        pose_metrics = evaluate_pose_auc(pred_poses, gt_poses) if gt_poses is not None else None
        if pose_metrics:
            pose_rows.append(pose_metrics)
        depth_rows.append(seq_metrics)
        total_time += elapsed
        total_frames += len(image_paths)
        frame_merge_stats.extend(getattr(model, "last_frame_merge_stats", []))
        token_merging_stats.extend(getattr(model, "last_token_merging_stats", []))

        seq_out.mkdir(parents=True, exist_ok=True)
        with (seq_out / "_metrics.json").open("w") as handle:
            json.dump({"depth": seq_metrics, "pose": pose_metrics, "speed": {"time": elapsed, "frames": len(image_paths)}}, handle, indent=2)

    valid_weights = np.asarray([row["valid_pixels"] for row in depth_rows], dtype=np.float64)
    depth_summary = {
        key: float(np.average([row[key] for row in depth_rows], weights=valid_weights))
        for key in ["Abs Rel", "Sq Rel", "RMSE", "Log RMSE", "delta < 1.25", "delta < 1.25^2", "delta < 1.25^3"]
    }
    depth_summary.update(
        {
            "dataset": args.dataset,
            "sequences": len(depth_rows),
            "frames": int(total_frames),
            "time": float(total_time),
            "fps": float(total_frames / total_time if total_time > 0 else 0.0),
            "token_merging_method": args.token_merging_method,
            "token_merging_ratio": args.token_merging_ratio,
            "valid_pixels": int(valid_weights.sum()),
        }
    )
    if frame_merge_stats:
        depth_summary["frame_merge_active_frames_mean"] = float(np.mean([s["active_frames_mean"] for s in frame_merge_stats]))
        depth_summary["frame_merge_merge_ratio_mean"] = float(np.mean([s["merge_ratio_mean"] for s in frame_merge_stats]))
        depth_summary["frame_merge_stats"] = frame_merge_stats
    if token_merging_stats:
        depth_summary["token_merging_full_attention_token_ratio_mean"] = float(
            np.mean([s["full_attention_token_ratio"] for s in token_merging_stats])
        )
        depth_summary["token_merging_stats"] = token_merging_stats

    pose_summary = summarize_pose(pose_rows) if pose_rows else None
    complete = {"video_depth": depth_summary, "pose_auc": pose_summary, "speed": {"frames": total_frames, "time": total_time, "fps": depth_summary["fps"]}}
    with (output_root / "_summary_scale_shift.json").open("w") as handle:
        json.dump(depth_summary, handle, indent=2)
    if pose_summary:
        with (output_root / "_summary_pose_auc.json").open("w") as handle:
            json.dump(pose_summary, handle, indent=2)
    with (output_root / "_summary_complete_scale_shift.json").open("w") as handle:
        json.dump(complete, handle, indent=2)
    write_csv(output_root / "_sequence_metrics_scale_shift.csv", depth_rows)
    print(json.dumps(complete, indent=2)[:4000])


if __name__ == "__main__":
    main()
