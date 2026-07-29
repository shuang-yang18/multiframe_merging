"""Evaluate saved c2w predictions with VGGT's official relative-pose AUC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--pred-dir", required=True, help="Inference output directory containing the bonn subdirectory.")
    parser.add_argument("--dataset", default="bonn")
    return parser.parse_args()


def read_tum_poses(path: Path, indices: list[int]) -> np.ndarray:
    values = np.atleast_2d(np.loadtxt(path, comments="#"))[indices]
    poses = np.tile(np.eye(4), (len(values), 1, 1))
    poses[:, :3, :3] = Rotation.from_quat(values[:, 4:8]).as_matrix()
    poses[:, :3, 3] = values[:, 1:4]
    return poses


def rotation_error_deg(reference: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    difference = reference @ np.transpose(prediction, (0, 2, 1))
    cosine = np.clip((np.trace(difference, axis1=1, axis2=2) - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def translation_error_deg(reference: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    reference = reference / np.maximum(np.linalg.norm(reference, axis=1, keepdims=True), 1e-15)
    prediction = prediction / np.maximum(np.linalg.norm(prediction, axis=1, keepdims=True), 1e-15)
    # VGGT's compare_translation_by_angle squares the dot product, making
    # translation direction sign-ambiguous.
    cosine = np.clip(np.abs(np.sum(reference * prediction, axis=1)), 0.0, 1.0)
    return np.degrees(np.arccos(cosine))


def calculate_auc(errors: np.ndarray, threshold: int) -> float:
    # Matches vggt/evaluation_branch/evaluation/test_co3d.py::calculate_auc_np.
    histogram, _ = np.histogram(errors, bins=np.arange(threshold + 1))
    return float(np.mean(np.cumsum(histogram.astype(np.float64) / len(errors))) * 100.0)


def evaluate_sequence(pred_c2w: np.ndarray, gt_c2w: np.ndarray) -> dict:
    pred_w2c = np.linalg.inv(np.asarray(pred_c2w, dtype=np.float64))
    gt_w2c = np.linalg.inv(np.asarray(gt_c2w, dtype=np.float64))
    first, second = np.triu_indices(len(pred_w2c), k=1)
    pred_relative = pred_w2c[first] @ np.linalg.inv(pred_w2c[second])
    gt_relative = gt_w2c[first] @ np.linalg.inv(gt_w2c[second])
    rotation_errors = rotation_error_deg(gt_relative[:, :3, :3], pred_relative[:, :3, :3])
    translation_errors = translation_error_deg(gt_relative[:, :3, 3], pred_relative[:, :3, 3])
    pose_errors = np.maximum(rotation_errors, translation_errors)
    return {
        "AUC@3": calculate_auc(pose_errors, 3),
        "AUC@30": calculate_auc(pose_errors, 30),
        "frames": int(len(pred_w2c)),
        "pairs": int(len(pose_errors)),
        "mean_pose_error_deg": float(np.mean(pose_errors)),
        "mean_rotation_error_deg": float(np.mean(rotation_errors)),
        "mean_translation_error_deg": float(np.mean(translation_errors)),
        "pose_errors_deg": pose_errors.tolist(),
        "rotation_errors_deg": rotation_errors.tolist(),
        "translation_errors_deg": translation_errors.tolist(),
        "pose_auc_protocol": "vggt_official_relative_pose_auc",
        "pose_convention": "input_c2w_converted_to_w2c",
        "relative_pose_pairs": "all_i_less_than_j",
        "translation_angle_ambiguity": True,
        "auc_integration": "vggt_histogram_cumulative_mean",
    }


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_root = Path(args.pred_dir) / args.dataset
    rows = []
    for sequence_dir in sorted(path for path in output_root.iterdir() if path.is_dir()):
        manifest_path = sequence_dir / "_bonn_associations.json"
        prediction_path = sequence_dir / "pred_poses.npy"
        if not manifest_path.is_file() or not prediction_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text())
        gt_poses = read_tum_poses(
            dataset_root / sequence_dir.name / "groundtruth.txt",
            [item["pose_index"] for item in manifest["associations"]],
        )
        row = evaluate_sequence(np.load(prediction_path), gt_poses)
        row["sequence"] = sequence_dir.name
        (sequence_dir / "_pose_auc_vggt_official.json").write_text(json.dumps(row, indent=2))
        rows.append(row)

    if not rows:
        raise FileNotFoundError(f"No Bonn prediction/association pairs found below {output_root}")
    all_errors = np.asarray([error for row in rows for error in row["pose_errors_deg"]], dtype=np.float64)
    all_rotation = np.asarray([error for row in rows for error in row["rotation_errors_deg"]], dtype=np.float64)
    all_translation = np.asarray([error for row in rows for error in row["translation_errors_deg"]], dtype=np.float64)
    summary = {
        "AUC@3": calculate_auc(all_errors, 3),
        "AUC@30": calculate_auc(all_errors, 30),
        "sequences": len(rows),
        "frames": int(sum(row["frames"] for row in rows)),
        "pairs": int(len(all_errors)),
        "mean_pose_error_deg": float(np.mean(all_errors)),
        "mean_rotation_error_deg": float(np.mean(all_rotation)),
        "mean_translation_error_deg": float(np.mean(all_translation)),
        "pose_auc_protocol": "vggt_official_relative_pose_auc",
        "translation_angle_ambiguity": True,
        "auc_integration": "vggt_histogram_cumulative_mean",
    }
    (output_root / "bonn-pose-auc-vggt-official.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
