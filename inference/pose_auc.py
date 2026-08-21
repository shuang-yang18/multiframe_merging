"""Camera-pose AUC helpers for VGGT-Omega video-depth evaluation."""

from __future__ import annotations

import json
from pathlib import Path
import zlib

import numpy as np


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    value = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(value)))


def _translation_angle_deg(
    pred: np.ndarray,
    gt: np.ndarray,
    eps: float = 1e-15,
    ambiguity: bool = True,
    default_err: float = 1e6,
) -> float:
    pred_norm = np.linalg.norm(pred)
    gt_norm = np.linalg.norm(gt)
    pred = pred / (pred_norm + eps)
    gt = gt / (gt_norm + eps)
    loss = max(1.0 - float(np.dot(pred, gt)) ** 2, eps)
    angle = float(np.degrees(np.arccos(np.sqrt(1.0 - loss))))
    if not np.isfinite(angle):
        return float(default_err)
    return min(angle, abs(180.0 - angle)) if ambiguity else angle


def _auc(errors: list[float], threshold: float) -> float:
    if not errors:
        return 0.0
    values = np.asarray(errors, dtype=np.float64)
    bins = np.arange(int(threshold) + 1)
    hist, _ = np.histogram(values, bins=bins)
    return float(np.mean(np.cumsum(hist.astype(np.float64) / float(len(values)))) * 100.0)


def _sample_indices(count: int, num_frames: int = 10, seed: int = 0, sequence: str = "") -> np.ndarray:
    if num_frames <= 0 or count <= num_frames:
        return np.arange(count, dtype=np.int64)
    seq_seed = int(seed) ^ zlib.crc32(sequence.encode("utf-8"))
    rng = np.random.default_rng(seq_seed)
    return np.sort(rng.choice(count, size=num_frames, replace=False))


def _to_homogeneous(poses: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float64)
    if poses.shape[-2:] == (4, 4):
        return poses.copy()
    if poses.shape[-2:] == (3, 4):
        bottom = np.zeros((*poses.shape[:-2], 1, 4), dtype=np.float64)
        bottom[..., 0, 3] = 1.0
        return np.concatenate([poses, bottom], axis=-2)
    raise ValueError(f"Expected poses with shape Nx4x4 or Nx3x4, got {poses.shape}.")


def _relative_pose_errors(pred_c2w: np.ndarray, gt_c2w: np.ndarray) -> tuple[list[float], list[float], list[float]]:
    pred_w2c = np.linalg.inv(_to_homogeneous(pred_c2w))
    gt_w2c = np.linalg.inv(_to_homogeneous(gt_c2w))

    pose_errors = []
    rotation_errors = []
    translation_errors = []
    count = len(pred_w2c)
    for i in range(count):
        for j in range(i + 1, count):
            pred_rel = pred_w2c[i] @ np.linalg.inv(pred_w2c[j])
            gt_rel = gt_w2c[i] @ np.linalg.inv(gt_w2c[j])
            rot_error = _rotation_angle_deg(gt_rel[:3, :3] @ pred_rel[:3, :3].T)
            trans_error = _translation_angle_deg(pred_rel[:3, 3], gt_rel[:3, 3])
            rotation_errors.append(rot_error)
            translation_errors.append(trans_error)
            pose_errors.append(max(rot_error, trans_error))
    return pose_errors, rotation_errors, translation_errors


def evaluate_pose_auc(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    num_frames: int = 0,
    seed: int = 0,
    sequence: str = "",
) -> dict:
    count = min(len(pred), len(gt))
    if count < 2:
        raise ValueError("Pose AUC needs at least two poses.")
    sampled_indices = _sample_indices(count, num_frames=num_frames, seed=seed, sequence=sequence)
    pred = np.asarray(pred[:count], dtype=np.float64)[sampled_indices]
    gt = np.asarray(gt[:count], dtype=np.float64)[sampled_indices]
    count = len(sampled_indices)
    pose_errors, rotation_errors, translation_errors = _relative_pose_errors(pred, gt)

    return {
        "AUC@3": _auc(pose_errors, 3.0),
        "AUC@5": _auc(pose_errors, 5.0),
        "AUC@10": _auc(pose_errors, 10.0),
        "AUC@15": _auc(pose_errors, 15.0),
        "AUC@30": _auc(pose_errors, 30.0),
        "frames": int(count),
        "pairs": int(len(pose_errors)),
        "sampled_frame_indices": [int(value) for value in sampled_indices.tolist()],
        "pose_eval_frames": int(num_frames),
        "pose_eval_seed": int(seed),
        "pose_auc_protocol": "vggt_official_relative_pose_auc",
        "pose_convention": "input_c2w_converted_to_w2c",
        "relative_pose_pairs": "all_i_less_than_j",
        "auc_integration": "histogram_cumsum_1deg_bins",
        "sim3_alignment": False,
        "translation_angle_ambiguity": True,
        "translation_nan_inf_error_deg": 1e6,
        "mean_pose_error_deg": float(np.mean(pose_errors)) if pose_errors else 0.0,
        "median_pose_error_deg": float(np.median(pose_errors)) if pose_errors else 0.0,
        "mean_rotation_error_deg": float(np.mean(rotation_errors)) if rotation_errors else 0.0,
        "mean_translation_error_deg": float(np.mean(translation_errors)) if translation_errors else 0.0,
        "pose_errors_deg": [float(value) for value in pose_errors],
        "rotation_errors_deg": [float(value) for value in rotation_errors],
        "translation_errors_deg": [float(value) for value in translation_errors],
    }


def summarize_pose_auc(rows: list[dict]) -> dict:
    pose_errors = []
    rotation_errors = []
    translation_errors = []
    for row in rows:
        pose_errors.extend(row.get("pose_errors_deg", []))
        rotation_errors.extend(row.get("rotation_errors_deg", []))
        translation_errors.extend(row.get("translation_errors_deg", []))
    summary = {
        "AUC@3": _auc(pose_errors, 3.0),
        "AUC@5": _auc(pose_errors, 5.0),
        "AUC@10": _auc(pose_errors, 10.0),
        "AUC@15": _auc(pose_errors, 15.0),
        "AUC@30": _auc(pose_errors, 30.0),
        "sequences": len(rows),
        "pairs": len(pose_errors),
        "mean_pose_error_deg": float(np.mean(pose_errors)) if pose_errors else 0.0,
        "median_pose_error_deg": float(np.median(pose_errors)) if pose_errors else 0.0,
        "mean_rotation_error_deg": float(np.mean(rotation_errors)) if rotation_errors else 0.0,
        "mean_translation_error_deg": float(np.mean(translation_errors)) if translation_errors else 0.0,
    }
    if rows:
        summary["pose_eval_frames"] = rows[0].get("pose_eval_frames")
        summary["pose_eval_seed"] = rows[0].get("pose_eval_seed")
        summary["pose_auc_protocol"] = rows[0].get("pose_auc_protocol")
        summary["pose_convention"] = rows[0].get("pose_convention")
        summary["relative_pose_pairs"] = rows[0].get("relative_pose_pairs")
        summary["auc_integration"] = rows[0].get("auc_integration")
        summary["sim3_alignment"] = rows[0].get("sim3_alignment")
        summary["translation_angle_ambiguity"] = rows[0].get("translation_angle_ambiguity")
        summary["translation_nan_inf_error_deg"] = rows[0].get("translation_nan_inf_error_deg")
    return summary


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)
