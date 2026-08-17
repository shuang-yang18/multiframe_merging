"""Pi3 camera-pose evaluation on timestamp-aligned Bonn test sequences."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from pi3.models.pi3 import Pi3
from relpose.evo_utils import eval_metrics, get_tum_poses, save_tum_poses
from utils.interfaces import infer_cameras_c2w


BONN_TEST_SEQUENCES = (
    "rgbd_bonn_balloon2",
    "rgbd_bonn_crowd2",
    "rgbd_bonn_crowd3",
    "rgbd_bonn_person_tracking2",
    "rgbd_bonn_synchronous",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="/data/mmc_syang/vggt-omega/datasets/Bonn/rgbd_bonn_dataset")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pretrained", default="checkpoints/Pi3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-frames-per-seq", type=int, default=300)
    parser.add_argument("--load-img-size", type=int, default=512)
    parser.add_argument("--association-max-diff", type=float, default=0.02)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def timestamp(path: str | Path) -> float:
    return float(Path(path).stem)


def one_to_one_matches(reference: np.ndarray, target: np.ndarray, max_difference: float) -> dict[int, tuple[int, float]]:
    candidates: list[tuple[float, int, int]] = []
    target_start = 0
    for ref_index, ref_time in enumerate(reference):
        while target_start < len(target) and target[target_start] < ref_time - max_difference:
            target_start += 1
        target_index = target_start
        while target_index < len(target) and target[target_index] <= ref_time + max_difference:
            candidates.append((abs(float(target[target_index] - ref_time)), ref_index, target_index))
            target_index += 1
    matches: dict[int, tuple[int, float]] = {}
    used_targets: set[int] = set()
    for difference, ref_index, target_index in sorted(candidates):
        if ref_index not in matches and target_index not in used_targets:
            matches[ref_index] = (target_index, difference)
            used_targets.add(target_index)
    return matches


def read_tum_poses(path: Path) -> tuple[np.ndarray, np.ndarray]:
    values = np.atleast_2d(np.loadtxt(path, comments="#"))
    poses = np.tile(np.eye(4), (len(values), 1, 1))
    poses[:, :3, :3] = Rotation.from_quat(values[:, 4:8]).as_matrix()
    poses[:, :3, 3] = values[:, 1:4]
    return values[:, 0], poses


def build_associations(root: Path, sequence: str, max_difference: float) -> list[dict]:
    sequence_root = root / sequence
    rgb_paths = sorted((sequence_root / "rgb").glob("*.png"))
    depth_paths = sorted((sequence_root / "depth").glob("*.png"))
    pose_times, _ = read_tum_poses(sequence_root / "groundtruth.txt")
    rgb_times = np.asarray([timestamp(path) for path in rgb_paths])
    depth_times = np.asarray([timestamp(path) for path in depth_paths])
    depth_matches = one_to_one_matches(rgb_times, depth_times, max_difference)
    pose_matches = one_to_one_matches(rgb_times, pose_times, max_difference)
    rows = []
    for rgb_index in sorted(set(depth_matches) & set(pose_matches)):
        depth_index, depth_delta = depth_matches[rgb_index]
        pose_index, pose_delta = pose_matches[rgb_index]
        rows.append(
            {
                "rgb_index": rgb_index,
                "rgb": rgb_paths[rgb_index].name,
                "depth_index": depth_index,
                "depth": depth_paths[depth_index].name,
                "pose_index": pose_index,
                "depth_delta_ms": depth_delta * 1000.0,
                "pose_delta_ms": pose_delta * 1000.0,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = Path(args.dataset_root)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = Pi3.from_pretrained(args.pretrained).to(device).eval()
    cfg = argparse.Namespace(device=args.device, load_img_size=args.load_img_size, verbose=False)

    rows = []
    all_times = []
    all_frames = []
    for sequence in tqdm(BONN_TEST_SEQUENCES, desc="Pi3 Bonn pose"):
        sequence_output = output_root / sequence
        metric_path = sequence_output / "_pose_metrics.json"
        if metric_path.is_file() and not args.overwrite:
            cached = json.loads(metric_path.read_text())
            rows.append(cached)
            all_times.append(cached["time"])
            all_frames.append(cached["frames"])
            continue

        associations = build_associations(root, sequence, args.association_max_diff)[: args.max_frames_per_seq]
        if len(associations) != args.max_frames_per_seq:
            raise ValueError(f"{sequence}: expected {args.max_frames_per_seq} timestamp triples, got {len(associations)}")
        _, gt_all = read_tum_poses(root / sequence / "groundtruth.txt")
        gt_poses = gt_all[[item["pose_index"] for item in associations]]
        image_paths = [str(root / sequence / "rgb" / item["rgb"]) for item in associations]

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        pred_poses, _ = infer_cameras_c2w(image_paths, model, cfg)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        pred_poses = pred_poses.detach().cpu().numpy()

        sequence_output.mkdir(parents=True, exist_ok=True)
        np.save(sequence_output / "pred_poses.npy", pred_poses)
        np.save(sequence_output / "gt_poses.npy", gt_poses)
        (sequence_output / "_bonn_associations.json").write_text(
            json.dumps(
                {
                    "protocol": "one_to_one_rgb_depth_pose_timestamp_association",
                    "max_difference_ms": args.association_max_diff * 1000.0,
                    "associations": associations,
                },
                indent=2,
            )
        )
        pred_trajectory = get_tum_poses(pred_poses)
        gt_trajectory = get_tum_poses(gt_poses)
        save_tum_poses(pred_trajectory, sequence_output / "pred_traj.txt")
        save_tum_poses(gt_trajectory, sequence_output / "gt_traj.txt")
        ate, rpe_trans, rpe_rot = eval_metrics(
            pred_trajectory,
            gt_trajectory,
            seq=sequence,
            filename=sequence_output / "eval_metric.txt",
            verbose=False,
        )
        row = {
            "sequence": sequence,
            "ATE": float(ate),
            "RPE trans": float(rpe_trans),
            "RPE rot": float(rpe_rot),
            "frames": len(image_paths),
            "time": float(elapsed),
            "fps": float(len(image_paths) / elapsed),
            "max_depth_delta_ms": float(max(item["depth_delta_ms"] for item in associations)),
            "max_pose_delta_ms": float(max(item["pose_delta_ms"] for item in associations)),
        }
        metric_path.write_text(json.dumps(row, indent=2))
        rows.append(row)
        all_times.append(elapsed)
        all_frames.append(len(image_paths))

    summary = {
        "dataset": "bonn_test",
        "sequences": len(rows),
        "frames": int(sum(all_frames)),
        "time": float(sum(all_times)),
        "fps": float(sum(all_frames) / sum(all_times)),
        "ATE": float(np.mean([row["ATE"] for row in rows])),
        "RPE trans": float(np.mean([row["RPE trans"] for row in rows])),
        "RPE rot": float(np.mean([row["RPE rot"] for row in rows])),
        "association_protocol": "one_to_one_rgb_depth_pose_timestamp_association",
        "association_max_difference_ms": args.association_max_diff * 1000.0,
    }
    write_csv(output_root / "sequence_metrics.csv", rows)
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
