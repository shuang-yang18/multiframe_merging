"""Camera-pose AUC helpers for VGGT-Omega video-depth evaluation."""

from __future__ import annotations

import json
from pathlib import Path
import zlib

import numpy as np


def _umeyama_alignment(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
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


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    value = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(value)))


def _translation_angle_deg(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8, ambiguity: bool = True) -> float | None:
    pred_norm = np.linalg.norm(pred)
    gt_norm = np.linalg.norm(gt)
    if pred_norm < eps or gt_norm < eps:
        return None
    value = np.clip(np.dot(pred, gt) / (pred_norm * gt_norm), -1.0, 1.0)
    angle = float(np.degrees(np.arccos(value)))
    if ambiguity:
        angle = min(angle, abs(180.0 - angle))
    return angle


def _auc(errors: list[float], threshold: float) -> float:
    if not errors:
        return 0.0
    values = np.asarray(errors, dtype=np.float64)
    return float(np.mean(np.maximum(1.0 - values / threshold, 0.0)) * 100.0)


def _sample_indices(count: int, num_frames: int = 10, seed: int = 0, sequence: str = "") -> np.ndarray:
    if num_frames <= 0 or count <= num_frames:
        return np.arange(count, dtype=np.int64)
    seq_seed = int(seed) ^ zlib.crc32(sequence.encode("utf-8"))
    rng = np.random.default_rng(seq_seed)
    return np.sort(rng.choice(count, size=num_frames, replace=False))


def evaluate_pose_auc(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    num_frames: int = 10,
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
            rot_error = _rotation_angle_deg(pred_rel_rotation @ gt_rel_rotation.T)
            trans_error = _translation_angle_deg(
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
        "frames": int(count),
        "pairs": int(len(pose_errors)),
        "sampled_frame_indices": [int(value) for value in sampled_indices.tolist()],
        "pose_eval_frames": int(num_frames),
        "pose_eval_seed": int(seed),
        "translation_angle_ambiguity": True,
        "mean_pose_error_deg": float(np.mean(pose_errors)) if pose_errors else 0.0,
        "median_pose_error_deg": float(np.median(pose_errors)) if pose_errors else 0.0,
        "mean_rotation_error_deg": float(np.mean(rotation_errors)) if rotation_errors else 0.0,
        "mean_translation_error_deg": float(np.mean(translation_errors)) if translation_errors else 0.0,
        "alignment_scale": scale,
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
        summary["translation_angle_ambiguity"] = rows[0].get("translation_angle_ambiguity")
    return summary


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)
