"""Run VGGT-Omega video-depth inference on supported video benchmarks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from vggt_omega.evaluation import (
    INTER_FRAME_ATTENTION_MODES,
    infer_sequence,
    load_model,
    save_depth_preview,
    sequence_images as sintel_sequence_images,
    sequence_names as sintel_sequence_names,
)

BONN_SEQUENCES = [
    "rgbd_bonn_balloon2",
    "rgbd_bonn_crowd2",
    "rgbd_bonn_crowd3",
    "rgbd_bonn_person_tracking2",
    "rgbd_bonn_synchronous",
]
DEFAULT_DATASET_ROOTS = {
    "sintel": "datasets/Sintel/training",
    "bonn": "datasets/Bonn/rgbd_bonn_dataset",
    "7scenes": "datasets/7scenes",
    "tum_dynamic": "datasets/TUM-Dynamics",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["sintel", "bonn", "7scenes", "tum_dynamic", "all"], default="all")
    parser.add_argument("--dataset-root")
    parser.add_argument("--bonn-rgb-dir", default="rgb_110", help="Bonn RGB subdirectory; use 'rgb' for full-length sequences.")
    parser.add_argument("--bonn-depth-dir", default="depth_110", help="Bonn depth subdirectory; use 'depth' for full-length sequences.")
    parser.add_argument("--checkpoint", default="checkpoints/vggt_omega_1b_512.pt")
    parser.add_argument("--output-dir", default="outputs/video_depth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--window-size", type=int, default=4, help="Use 0 to infer a whole sequence jointly.")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--input-mode", choices=["balanced", "max_size"], default="balanced")
    parser.add_argument(
        "--inter-frame-attention",
        choices=INTER_FRAME_ATTENTION_MODES,
        default="partial",
        help="Use all global inter-frame attention, the released partial-register schedule, or all register attention.",
    )
    parser.add_argument(
        "--register-patch-sample-tokens",
        type=int,
        default=0,
        help="Sample this many patch tokens per frame into register inter-frame attention blocks.",
    )
    parser.add_argument(
        "--register-patch-sample-ratio",
        type=float,
        default=0.0,
        help="Sample this fraction of patch tokens per frame when --register-patch-sample-tokens is 0.",
    )
    parser.add_argument(
        "--register-patch-sample-mode",
        choices=[
            "uniform",
            "norm",
            "register_cosine",
            "qkv_register_low_conf",
            "rgb_gradient",
            "attention_proxy",
            "temporal_change",
            "depth_uncertainty",
        ],
        default="uniform",
        help="Patch-token scoring mode for register inter-frame attention blocks.",
    )
    parser.add_argument(
        "--register-patch-merge-sources",
        action="store_true",
        default=False,
        help="Merge non-sampled patch tokens into sampled patch destination tokens in register attention blocks.",
    )
    parser.add_argument(
        "--register-patch-merge-protect-first-frame",
        action="store_true",
        default=False,
        help="Keep all first-frame patch tokens as destinations when register patch source merging is enabled.",
    )
    parser.add_argument("--enable-token-merging", action="store_true", default=False)
    parser.add_argument("--token-merging-start", type=int, default=0)
    parser.add_argument(
        "--token-merging-ratio",
        type=float,
        default=0.9,
        help="Fraction of patch tokens to merge away inside global inter-frame attention; 0.9 keeps about 10%%.",
    )
    parser.add_argument(
        "--token-merging-layer-ratios",
        default="",
        help="Optional 1-based layer schedule, e.g. '1-10:0.9,11-18:0.3,19-24:0.9'.",
    )
    parser.add_argument(
        "--token-merging-method",
        choices=[
            "spatial",
            "protected_spatial",
            "tstm",
            "flashvid_encoder",
            "frame_temporary",
            "frame_persistent",
            "frame_persistent_spatial",
            "frame_persistent_decoupled",
            "frame_persistent_decoupled_window",
        ],
        default="spatial",
        help="Token merging strategy used inside global inter-frame attention blocks.",
    )
    parser.add_argument(
        "--token-merging-tstm-threshold",
        type=float,
        default=0.8,
        help="Cosine threshold for local inter-frame merging when --token-merging-method=tstm.",
    )
    parser.add_argument(
        "--token-merging-tstm-neighbor-size",
        type=int,
        default=3,
        help="Odd local window size used by TSTM to find previous-frame candidates; 0 searches the full previous frame.",
    )
    parser.add_argument("--token-merging-flashvid-alpha", type=float, default=0.7)
    parser.add_argument("--token-merging-flashvid-expansion", type=float, default=1.25)
    parser.add_argument("--token-merging-flashvid-pool-stride", type=int, default=2)
    parser.add_argument("--token-merging-frame-restore-layer", type=int, default=16)
    parser.add_argument("--token-merging-frame-alpha", type=float, default=0.9)
    parser.add_argument("--token-merging-frame-segment-threshold", type=float, default=0.8)
    parser.add_argument("--token-merging-frame-merge-threshold", type=float, default=0.8)
    parser.add_argument("--token-merging-frame-max-window", type=int, default=6)
    parser.add_argument("--token-merging-frame-pool-stride", type=int, default=2)
    parser.add_argument("--token-merging-frame-multi-max-group-size", type=int, default=2)
    parser.add_argument("--token-merging-frame-multi-pair-threshold", type=float, default=0.95)
    parser.add_argument("--token-merging-frame-multi-span-threshold", type=float, default=0.93)
    parser.add_argument("--token-merging-frame-protect-period", type=int, default=0)
    parser.add_argument("--token-merging-frame-protect-prefix", type=int, default=0)
    parser.add_argument(
        "--token-merging-frame-group-strategy",
        choices=["local", "segment_middle", "global_cluster", "global_top_pairs"],
        default="local",
        help="Frame grouping strategy inside each segment.",
    )
    parser.add_argument(
        "--omega-accelerator",
        choices=["none", "da_vggt", "sparse_vggt"],
        default="none",
        help="Explicit external acceleration adapter; separate from the existing token-merging methods.",
    )
    parser.add_argument("--sparse-vggt-sparse-ratio", type=float, default=0.5)
    parser.add_argument("--sparse-vggt-cdf-threshold", type=float)
    parser.add_argument("--sparse-vggt-pool-mode", choices=["avg", "max"], default="avg")
    parser.add_argument("--da-vggt-max-frames", type=int, default=0)
    parser.add_argument("--da-vggt-sampling-method", choices=["fl_maxmin", "step"], default="fl_maxmin")
    parser.add_argument("--da-vggt-n-anchors", type=int, default=1)
    parser.add_argument("--da-vggt-dino-batch-size", type=int, default=256)
    parser.add_argument("--da-vggt-lambda-div", type=float, default=0.0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--eval", action="store_true", help="Evaluate this prediction run immediately after inference.")
    parser.add_argument("--eval-align", choices=["metric", "scale", "scale_shift"], default="scale_shift")
    parser.add_argument(
        "--pose-eval-frames",
        type=int,
        default=0,
        help=(
            "Number of frames to sample only for pose AUC. Default 0 evaluates all inferred frames; "
            "for paper-style 10-view VGGT evaluation, run inference with 10 input frames instead of "
            "subsampling a longer prediction afterward."
        ),
    )
    parser.add_argument("--pose-eval-seed", type=int, default=0, help="Seed for deterministic pose-AUC frame sampling.")
    parser.add_argument("--max-depth", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--all-scenes", action="store_true", default=True, help="Run every scene available in the dataset root.")
    parser.add_argument("--sequences", nargs="*")
    parser.add_argument(
        "--seven-scenes-split",
        choices=["test", "train", "all"],
        default="test",
        help="7Scenes split to use when --dataset 7scenes. Defaults to the official test split.",
    )
    parser.add_argument("--max-frames-per-seq", type=int, default=0, help="Limit each sequence to this many frames after stride.")
    parser.add_argument(
        "--frame-sample-mode",
        choices=["first", "random"],
        default="first",
        help="How to choose frames when --max-frames-per-seq is set.",
    )
    parser.add_argument("--frame-sample-seed", type=int, default=0, help="Seed for random frame input sampling.")
    return parser.parse_args()


def _parse_7scenes_split_line(line: str) -> str | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("sequence"):
        number = line.removeprefix("sequence")
        if number.isdigit():
            return f"seq-{int(number):02d}"
    if line.startswith("seq-"):
        return line
    raise ValueError(f"Unsupported 7Scenes split entry: {line!r}")


def _read_7scenes_split(scene_dir: Path, split: str) -> list[str]:
    if split == "all" or not (scene_dir / "TestSplit.txt").exists():
        return sorted(
            seq.name
            for seq in scene_dir.iterdir()
            if seq.is_dir() and seq.name.startswith("seq-") and any(seq.glob("*.color.png"))
        )

    split_file = scene_dir / ("TestSplit.txt" if split == "test" else "TrainSplit.txt")
    if not split_file.is_file():
        raise FileNotFoundError(f"Missing 7Scenes {split} split file: {split_file}")

    sequences = []
    with split_file.open() as handle:
        for line in handle:
            seq_name = _parse_7scenes_split_line(line)
            if seq_name is not None:
                sequences.append(seq_name)
    return sequences


def seven_scenes_sequence_names(dataset_root: str | Path, split: str = "test") -> list[str]:
    root = Path(dataset_root)
    split_container_names = {"train", "test"}
    sequences = []
    for scene in sorted(path for path in root.iterdir() if path.is_dir()):
        if scene.name.lower() in split_container_names:
            continue
        for seq_name in _read_7scenes_split(scene, split):
            seq_dir = scene / seq_name
            if seq_dir.is_dir() and any(seq_dir.glob("*.color.png")):
                sequences.append(f"{scene.name}/{seq_name}")
            else:
                raise FileNotFoundError(f"7Scenes split references missing sequence: {seq_dir}")
    return sequences


def sequence_names(
    dataset: str,
    dataset_root: str | Path,
    requested: list[str] | None,
    all_scenes: bool = False,
    seven_scenes_split: str = "test",
    bonn_rgb_dir: str = "rgb_110",
) -> list[str]:
    if dataset == "sintel":
        if all_scenes and not requested:
            return sorted(path.name for path in (Path(dataset_root) / "final").iterdir() if path.is_dir())
        return sintel_sequence_names(dataset_root, requested)
    if dataset == "7scenes":
        if all_scenes and not requested:
            return seven_scenes_sequence_names(dataset_root, seven_scenes_split)
        requested = requested or []
        missing = [seq for seq in requested if not (Path(dataset_root) / seq).is_dir()]
        if missing:
            raise FileNotFoundError(f"Missing 7Scenes sequences below {dataset_root}: {missing}")
        if seven_scenes_split != "all":
            allowed = set(seven_scenes_sequence_names(dataset_root, seven_scenes_split))
            outside_split = [seq for seq in requested if seq not in allowed]
            if outside_split:
                raise ValueError(
                    f"Requested 7Scenes sequences are outside the {seven_scenes_split} split: {outside_split}. "
                    "Use --seven-scenes-split all to run an explicit mixed split."
                )
        return requested
    if dataset == "tum_dynamic":
        if all_scenes and not requested:
            return sorted(path.name for path in Path(dataset_root).iterdir() if (path / "rgb").is_dir())
        requested = requested or sorted(path.name for path in Path(dataset_root).iterdir() if (path / "rgb").is_dir())
        missing = [seq for seq in requested if not (Path(dataset_root) / seq / "rgb").is_dir()]
        if missing:
            raise FileNotFoundError(f"Missing TUM-Dynamics rgb directories below {dataset_root}: {missing}")
        return requested
    if all_scenes and not requested:
        requested = sorted(path.name for path in Path(dataset_root).iterdir() if (path / bonn_rgb_dir).is_dir())
    else:
        requested = requested or BONN_SEQUENCES
    missing = [seq for seq in requested if not (Path(dataset_root) / seq / bonn_rgb_dir).is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing Bonn {bonn_rgb_dir} directories below {dataset_root}: {missing}")
    return requested


def sequence_images(dataset: str, dataset_root: str | Path, seq: str, bonn_rgb_dir: str = "rgb_110") -> list[str]:
    if dataset == "sintel":
        return sintel_sequence_images(dataset_root, seq)
    if dataset == "7scenes":
        paths = sorted((Path(dataset_root) / seq).glob("*.color.png"))
        if not paths:
            raise FileNotFoundError(f"No 7Scenes input images found for sequence {seq}")
        return [str(path) for path in paths]
    if dataset == "tum_dynamic":
        paths = sorted((Path(dataset_root) / seq / "rgb").glob("*.png"))
        if not paths:
            raise FileNotFoundError(f"No TUM-Dynamics rgb images found for sequence {seq}")
        return [str(path) for path in paths]
    paths = sorted((Path(dataset_root) / seq / bonn_rgb_dir).glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No Bonn input images found for sequence {seq} in {bonn_rgb_dir}")
    return [str(path) for path in paths]


def _sequence_seed(seed: int, sequence: str | None) -> int:
    if not sequence:
        return seed
    digest = hashlib.sha1(sequence.encode("utf-8")).digest()
    return (seed + int.from_bytes(digest[:4], "little")) % (2**32)


def limit_frames(
    paths: list[str],
    max_frames: int,
    sample_mode: str = "first",
    seed: int = 0,
    sequence: str | None = None,
) -> list[str]:
    if not max_frames or max_frames <= 0 or len(paths) <= max_frames:
        return paths
    if sample_mode == "random":
        rng = np.random.default_rng(_sequence_seed(seed, sequence))
        indices = sorted(rng.choice(len(paths), size=max_frames, replace=False).tolist())
        return [paths[index] for index in indices]
    return paths[:max_frames]


def _read_tum_poses(path: Path) -> np.ndarray:
    values = np.loadtxt(path, comments="#")
    values = np.atleast_2d(values)
    poses = np.tile(np.eye(4), (values.shape[0], 1, 1))
    poses[:, :3, :3] = Rotation.from_quat(values[:, 4:8]).as_matrix()
    poses[:, :3, 3] = values[:, 1:4]
    return poses


def _read_tum_pose_entries(path: Path) -> tuple[np.ndarray, np.ndarray]:
    values = np.loadtxt(path, comments="#")
    values = np.atleast_2d(values)
    return values[:, 0], _read_tum_poses(path)


def _timestamp(path: str | Path) -> float:
    return float(Path(path).stem)


def sequence_poses(
    dataset: str,
    dataset_root: str | Path,
    seq: str,
    frame_count: int,
    image_paths: list[str] | None = None,
) -> np.ndarray | None:
    root = Path(dataset_root)
    if dataset == "7scenes":
        if image_paths is not None:
            paths = [Path(path).with_name(Path(path).name.replace(".color.png", ".pose.txt")) for path in image_paths]
        else:
            paths = sorted((root / seq).glob("*.pose.txt"))[:frame_count]
        if not paths:
            return None
        return np.stack([np.loadtxt(path) for path in paths])
    if dataset == "tum_dynamic":
        path = root / seq / "groundtruth.txt"
        if not path.is_file():
            return None
        if image_paths is None:
            return _read_tum_poses(path)[:frame_count]
        pose_times, poses = _read_tum_pose_entries(path)
        matched = []
        for image_path in image_paths[:frame_count]:
            idx = int(np.argmin(np.abs(pose_times - _timestamp(image_path))))
            matched.append(poses[idx])
        return np.stack(matched)
    return None


def save_tum_trajectory(path: Path, poses: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for index, pose in enumerate(poses):
            qx, qy, qz, qw = Rotation.from_matrix(pose[:3, :3]).as_quat()
            x, y, z = pose[:3, 3]
            handle.write(f"{index} {x} {y} {z} {qx} {qy} {qz} {qw}\n")


def save_frame_similarity_matrices(output_dir: Path, speed_metrics: dict) -> list[str]:
    saved_paths: list[str] = []
    frame_merge_stats = speed_metrics.get("frame_merge_stats")
    if not isinstance(frame_merge_stats, list):
        return saved_paths
    for event_idx, stat in enumerate(frame_merge_stats):
        if not isinstance(stat, dict):
            continue
        matrices = stat.pop("similarity_matrices", None)
        if not matrices:
            continue
        event_paths = []
        for batch_idx, matrix in enumerate(matrices):
            suffix = f"event{event_idx:03d}"
            if len(matrices) > 1:
                suffix = f"{suffix}_batch{batch_idx:03d}"
            matrix_array = np.asarray(matrix, dtype=np.float32)
            npy_path = output_dir / f"similarity_matrix_{suffix}.npy"
            csv_path = output_dir / f"similarity_matrix_{suffix}.csv"
            np.save(npy_path, matrix_array)
            np.savetxt(csv_path, matrix_array, delimiter=",", fmt="%.6f")
            event_paths.extend([npy_path.name, csv_path.name])
            saved_paths.extend([npy_path.name, csv_path.name])
        stat["similarity_matrix_files"] = event_paths
    return saved_paths


def write_rows(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def maybe_evaluate_dataset(args: argparse.Namespace, dataset: str, dataset_root: str | Path, output_root: Path) -> None:
    if not args.eval:
        return
    if __package__:
        from .eval import depth_metrics, preprocess_depth, resize_prediction_to_gt, resolve_max_depth, sequence_depths
    else:
        from eval import depth_metrics, preprocess_depth, resize_prediction_to_gt, resolve_max_depth, sequence_depths

    sequence_rows = []
    sequences = sequence_names(
        dataset,
        dataset_root,
        args.sequences,
        args.all_scenes,
        args.seven_scenes_split,
        args.bonn_rgb_dir,
    )
    for seq in tqdm(sequences, desc=f"VGGT-Omega {dataset} video depth eval"):
        images = limit_frames(
            sequence_images(dataset, dataset_root, seq, args.bonn_rgb_dir),
            args.max_frames_per_seq,
            args.frame_sample_mode,
            args.frame_sample_seed,
            seq,
        )
        gt_paths = sequence_depths(dataset, dataset_root, seq, args.bonn_depth_dir, args.bonn_rgb_dir)
        seq_dir = output_root / seq
        pred_paths = sorted(seq_dir.glob("frame_[0-9][0-9][0-9][0-9].npy"))
        gt_paths = limit_frames(gt_paths, args.max_frames_per_seq, args.frame_sample_mode, args.frame_sample_seed, seq)
        if not (len(pred_paths) == len(gt_paths) == len(images)):
            raise ValueError(
                f"{dataset}/{seq}: found {len(pred_paths)} predictions, "
                f"{len(gt_paths)} ground truth depths, and {len(images)} images"
            )
        with (seq_dir / "_time.json").open() as handle:
            timing = json.load(handle)
        width, height = timing["resolution"]
        gt_frames = [preprocess_depth(dataset, image, depth, (width, height)) for image, depth in zip(images, gt_paths)]
        pred = np.stack([resize_prediction_to_gt(np.load(path), gt) for path, gt in zip(pred_paths, gt_frames)])
        gt = np.stack(gt_frames)
        row = {"sequence": seq, **depth_metrics(pred, gt, args.eval_align, resolve_max_depth(dataset, args.max_depth))}
        row["frames"] = timing["frames"]
        row["time"] = timing["time"]
        row["fps"] = timing.get("fps", timing["frames"] / timing["time"])
        row["seconds_per_frame"] = timing.get("seconds_per_frame", timing["time"] / timing["frames"])
        row["peak_memory_allocated_gb"] = timing.get("peak_memory_allocated_gb")
        row["peak_memory_reserved_gb"] = timing.get("peak_memory_reserved_gb")
        row["inter_frame_attention"] = timing.get("inter_frame_attention")
        row["register_patch_sample_tokens"] = timing.get("register_patch_sample_tokens")
        row["register_patch_sample_ratio"] = timing.get("register_patch_sample_ratio")
        row["register_patch_sample_mode"] = timing.get("register_patch_sample_mode")
        row["register_patch_merge_sources"] = timing.get("register_patch_merge_sources")
        row["register_patch_merge_protect_first_frame"] = timing.get("register_patch_merge_protect_first_frame")
        row["enable_token_merging"] = timing.get("enable_token_merging")
        row["token_merging_method"] = timing.get("token_merging_method")
        row["token_merging_start"] = timing.get("token_merging_start")
        row["token_merging_ratio"] = timing.get("token_merging_ratio")
        row["token_merging_layer_ratios"] = timing.get("token_merging_layer_ratios")
        row["token_merging_tstm_threshold"] = timing.get("token_merging_tstm_threshold")
        row["token_merging_tstm_neighbor_size"] = timing.get("token_merging_tstm_neighbor_size")
        row["token_merging_flashvid_alpha"] = timing.get("token_merging_flashvid_alpha")
        row["token_merging_flashvid_expansion"] = timing.get("token_merging_flashvid_expansion")
        row["token_merging_flashvid_pool_stride"] = timing.get("token_merging_flashvid_pool_stride")
        row["token_merging_frame_restore_layer"] = timing.get("token_merging_frame_restore_layer")
        row["token_merging_frame_alpha"] = timing.get("token_merging_frame_alpha")
        row["token_merging_frame_segment_threshold"] = timing.get("token_merging_frame_segment_threshold")
        row["token_merging_frame_merge_threshold"] = timing.get("token_merging_frame_merge_threshold")
        row["token_merging_frame_max_window"] = timing.get("token_merging_frame_max_window")
        row["token_merging_frame_pool_stride"] = timing.get("token_merging_frame_pool_stride")
        row["token_merging_frame_multi_max_group_size"] = timing.get(
            "token_merging_frame_multi_max_group_size"
        )
        row["token_merging_frame_multi_pair_threshold"] = timing.get(
            "token_merging_frame_multi_pair_threshold"
        )
        row["token_merging_frame_multi_span_threshold"] = timing.get(
            "token_merging_frame_multi_span_threshold"
        )
        row["token_merging_frame_group_strategy"] = timing.get("token_merging_frame_group_strategy")
        row["token_merging_frame_protect_period"] = timing.get("token_merging_frame_protect_period")
        row["token_merging_frame_protect_prefix"] = timing.get("token_merging_frame_protect_prefix")
        row["omega_accelerator"] = timing.get("omega_accelerator")
        row["sparse_vggt_sparsity_mean"] = timing.get("sparse_vggt_sparsity_mean")
        row["sparse_vggt_sparse_ratio"] = timing.get("sparse_vggt_sparse_ratio")
        row["sparse_vggt_cdf_threshold"] = timing.get("sparse_vggt_cdf_threshold")
        row["sparse_vggt_pool_mode"] = timing.get("sparse_vggt_pool_mode")
        row["da_vggt_num_chunks_mean"] = timing.get("da_vggt_num_chunks_mean")
        row["da_vggt_chunk_size"] = timing.get("da_vggt_chunk_size")
        row["da_vggt_sampling_method"] = timing.get("da_vggt_sampling_method")
        row["frame_merge_active_frames_mean"] = timing.get("frame_merge_active_frames_mean")
        row["frame_merge_retention_ratio_mean"] = timing.get("frame_merge_retention_ratio_mean")
        row["frame_merge_merge_ratio_mean"] = timing.get("frame_merge_merge_ratio_mean")
        row["token_merging_active_over_frame_merged_token_ratio_mean"] = timing.get(
            "token_merging_active_over_frame_merged_token_ratio_mean"
        )
        row["token_merging_active_over_frame_original_token_ratio_mean"] = timing.get(
            "token_merging_active_over_frame_original_token_ratio_mean"
        )
        row["token_merging_full_attention_token_ratio_mean"] = timing.get(
            "token_merging_full_attention_token_ratio_mean"
        )
        sequence_rows.append(row)

    weights = np.asarray([row["valid_pixels"] for row in sequence_rows], dtype=np.float64)
    summary = {
        key: float(np.average([row[key] for row in sequence_rows], weights=weights))
        for key in sequence_rows[0]
        if key
        not in {
            "sequence",
            "frames",
            "valid_pixels",
            "time",
            "fps",
            "seconds_per_frame",
            "peak_memory_allocated_gb",
            "peak_memory_reserved_gb",
            "inter_frame_attention",
            "register_patch_sample_tokens",
            "register_patch_sample_ratio",
            "register_patch_sample_mode",
            "register_patch_merge_sources",
            "register_patch_merge_protect_first_frame",
            "enable_token_merging",
            "token_merging_method",
            "token_merging_start",
            "token_merging_ratio",
            "token_merging_layer_ratios",
            "token_merging_tstm_threshold",
            "token_merging_tstm_neighbor_size",
            "token_merging_flashvid_alpha",
            "token_merging_flashvid_expansion",
            "token_merging_flashvid_pool_stride",
            "token_merging_frame_restore_layer",
            "token_merging_frame_alpha",
            "token_merging_frame_segment_threshold",
            "token_merging_frame_merge_threshold",
            "token_merging_frame_max_window",
            "token_merging_frame_pool_stride",
            "token_merging_frame_multi_max_group_size",
            "token_merging_frame_multi_pair_threshold",
            "token_merging_frame_multi_span_threshold",
            "token_merging_frame_group_strategy",
            "token_merging_frame_protect_period",
            "token_merging_frame_protect_prefix",
            "omega_accelerator",
            "sparse_vggt_pool_mode",
            "da_vggt_sampling_method",
            "depth_eval_protocol",
        }
        and all(isinstance(row.get(key), (int, float)) and row.get(key) is not None for row in sequence_rows)
    }
    total_frames = int(sum(row["frames"] for row in sequence_rows))
    total_time = float(sum(row["time"] for row in sequence_rows))
    summary.update(
        {
            "dataset": dataset,
            "sequences": len(sequence_rows),
            "frames": total_frames,
            "time": total_time,
            "fps": float(np.average([row["fps"] for row in sequence_rows])),
            "seconds_per_frame": total_time / total_frames if total_frames > 0 else None,
            "peak_memory_allocated_gb": max(
                row["peak_memory_allocated_gb"] for row in sequence_rows if row["peak_memory_allocated_gb"] is not None
            ),
            "peak_memory_reserved_gb": max(
                row["peak_memory_reserved_gb"] for row in sequence_rows if row["peak_memory_reserved_gb"] is not None
            ),
            "inter_frame_attention": args.inter_frame_attention,
            "register_patch_sample_tokens": args.register_patch_sample_tokens,
            "register_patch_sample_ratio": args.register_patch_sample_ratio,
            "register_patch_sample_mode": args.register_patch_sample_mode,
            "register_patch_merge_sources": args.register_patch_merge_sources,
            "register_patch_merge_protect_first_frame": args.register_patch_merge_protect_first_frame,
            "enable_token_merging": args.enable_token_merging,
            "token_merging_method": args.token_merging_method,
            "token_merging_start": args.token_merging_start,
            "token_merging_ratio": args.token_merging_ratio,
            "token_merging_layer_ratios": args.token_merging_layer_ratios,
            "token_merging_tstm_threshold": args.token_merging_tstm_threshold,
            "token_merging_tstm_neighbor_size": args.token_merging_tstm_neighbor_size,
            "token_merging_flashvid_alpha": args.token_merging_flashvid_alpha,
            "token_merging_flashvid_expansion": args.token_merging_flashvid_expansion,
            "token_merging_flashvid_pool_stride": args.token_merging_flashvid_pool_stride,
            "token_merging_frame_restore_layer": args.token_merging_frame_restore_layer,
            "token_merging_frame_alpha": args.token_merging_frame_alpha,
            "token_merging_frame_segment_threshold": args.token_merging_frame_segment_threshold,
            "token_merging_frame_merge_threshold": args.token_merging_frame_merge_threshold,
            "token_merging_frame_max_window": args.token_merging_frame_max_window,
            "token_merging_frame_pool_stride": args.token_merging_frame_pool_stride,
            "token_merging_frame_multi_max_group_size": args.token_merging_frame_multi_max_group_size,
            "token_merging_frame_multi_pair_threshold": args.token_merging_frame_multi_pair_threshold,
            "token_merging_frame_multi_span_threshold": args.token_merging_frame_multi_span_threshold,
            "token_merging_frame_group_strategy": args.token_merging_frame_group_strategy,
            "token_merging_frame_protect_period": args.token_merging_frame_protect_period,
            "token_merging_frame_protect_prefix": args.token_merging_frame_protect_prefix,
            "omega_accelerator": args.omega_accelerator,
            "sparse_vggt_sparse_ratio": args.sparse_vggt_sparse_ratio,
            "sparse_vggt_cdf_threshold": args.sparse_vggt_cdf_threshold,
            "sparse_vggt_pool_mode": args.sparse_vggt_pool_mode,
            "da_vggt_max_frames": args.da_vggt_max_frames,
            "da_vggt_sampling_method": args.da_vggt_sampling_method,
            "da_vggt_n_anchors": args.da_vggt_n_anchors,
            "da_vggt_dino_batch_size": args.da_vggt_dino_batch_size,
            "da_vggt_lambda_div": args.da_vggt_lambda_div,
            "eval_align": args.eval_align,
            "valid_pixels": int(weights.sum()),
        }
    )
    write_rows(output_root / f"_sequence_metrics_{args.eval_align}.csv", sequence_rows)
    write_rows(output_root / f"_summary_{args.eval_align}.csv", [summary])
    with (output_root / f"_summary_{args.eval_align}.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    if __package__:
        from .pose_auc import evaluate_pose_auc, summarize_pose_auc, write_json
    else:
        from pose_auc import evaluate_pose_auc, summarize_pose_auc, write_json

    pose_rows = []
    for seq in sequences:
        pose_path = output_root / seq / "_pose_auc.json"
        pred_pose_path = output_root / seq / "pred_poses.npy"
        images = limit_frames(
            sequence_images(dataset, dataset_root, seq, args.bonn_rgb_dir),
            args.max_frames_per_seq,
            args.frame_sample_mode,
            args.frame_sample_seed,
            seq,
        )
        gt_poses = sequence_poses(dataset, dataset_root, seq, len(images), images)
        if pred_pose_path.is_file() and gt_poses is not None:
            pred_poses = np.load(pred_pose_path)
            pose_row = evaluate_pose_auc(
                pred_poses,
                gt_poses,
                num_frames=args.pose_eval_frames,
                seed=args.pose_eval_seed,
                sequence=seq,
            )
            write_json(pose_path, pose_row)
            pose_row["sequence"] = seq
            pose_rows.append(pose_row)
        elif pose_path.is_file():
            with pose_path.open() as handle:
                pose_row = json.load(handle)
            pose_row["sequence"] = seq
            pose_rows.append(pose_row)
    pose_summary = summarize_pose_auc(pose_rows) if pose_rows else None
    if pose_summary is not None:
        write_json(output_root / "_summary_pose_auc.json", pose_summary)
    write_json(
        output_root / f"_summary_complete_{args.eval_align}.json",
        {
            "video_depth": summary,
            "pose_auc": pose_summary,
            "speed": {
                "frames": summary.get("frames"),
                "time": summary.get("time"),
                "fps": summary.get("fps"),
                "seconds_per_frame": summary.get("seconds_per_frame"),
                "peak_memory_allocated_gb": summary.get("peak_memory_allocated_gb"),
                "peak_memory_reserved_gb": summary.get("peak_memory_reserved_gb"),
                "frame_merge_active_frames_mean": summary.get("frame_merge_active_frames_mean"),
                "frame_merge_retention_ratio_mean": summary.get("frame_merge_retention_ratio_mean"),
                "frame_merge_merge_ratio_mean": summary.get("frame_merge_merge_ratio_mean"),
                "token_merging_active_over_frame_merged_token_ratio_mean": summary.get(
                    "token_merging_active_over_frame_merged_token_ratio_mean"
                ),
                "token_merging_active_over_frame_original_token_ratio_mean": summary.get(
                    "token_merging_active_over_frame_original_token_ratio_mean"
                ),
                "token_merging_full_attention_token_ratio_mean": summary.get(
                    "token_merging_full_attention_token_ratio_mean"
                ),
            },
        },
    )
    print(f"{dataset} {args.eval_align}: {summary}")


def main() -> None:
    args = parse_args()
    if args.dataset == "all" and args.dataset_root:
        raise ValueError("--dataset-root cannot be used with --dataset all; use the default dataset roots.")
    datasets = ["sintel", "bonn", "7scenes", "tum_dynamic"] if args.dataset == "all" else [args.dataset]
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = load_model(
        args.checkpoint,
        device,
        enable_camera=True,
        inter_frame_attention=args.inter_frame_attention,
        register_patch_sample_tokens=args.register_patch_sample_tokens,
        register_patch_sample_ratio=args.register_patch_sample_ratio,
        register_patch_sample_mode=args.register_patch_sample_mode,
        register_patch_merge_sources=args.register_patch_merge_sources,
        register_patch_merge_protect_first_frame=args.register_patch_merge_protect_first_frame,
        enable_token_merging=args.enable_token_merging,
        token_merging_start=args.token_merging_start,
        token_merging_ratio=args.token_merging_ratio,
        token_merging_layer_ratios=args.token_merging_layer_ratios,
        token_merging_method=args.token_merging_method,
        token_merging_tstm_threshold=args.token_merging_tstm_threshold,
        token_merging_tstm_neighbor_size=args.token_merging_tstm_neighbor_size,
        token_merging_flashvid_alpha=args.token_merging_flashvid_alpha,
        token_merging_flashvid_expansion=args.token_merging_flashvid_expansion,
        token_merging_flashvid_pool_stride=args.token_merging_flashvid_pool_stride,
        token_merging_frame_restore_layer=args.token_merging_frame_restore_layer,
        token_merging_frame_alpha=args.token_merging_frame_alpha,
        token_merging_frame_segment_threshold=args.token_merging_frame_segment_threshold,
        token_merging_frame_merge_threshold=args.token_merging_frame_merge_threshold,
        token_merging_frame_max_window=args.token_merging_frame_max_window,
        token_merging_frame_pool_stride=args.token_merging_frame_pool_stride,
        token_merging_frame_multi_max_group_size=args.token_merging_frame_multi_max_group_size,
        token_merging_frame_multi_pair_threshold=args.token_merging_frame_multi_pair_threshold,
        token_merging_frame_multi_span_threshold=args.token_merging_frame_multi_span_threshold,
        token_merging_frame_group_strategy=args.token_merging_frame_group_strategy,
        token_merging_frame_protect_period=args.token_merging_frame_protect_period,
        token_merging_frame_protect_prefix=args.token_merging_frame_protect_prefix,
        omega_accelerator=args.omega_accelerator,
        sparse_vggt_sparse_ratio=args.sparse_vggt_sparse_ratio,
        sparse_vggt_cdf_threshold=args.sparse_vggt_cdf_threshold,
        sparse_vggt_pool_mode=args.sparse_vggt_pool_mode,
        da_vggt_max_frames=args.da_vggt_max_frames,
        da_vggt_sampling_method=args.da_vggt_sampling_method,
        da_vggt_n_anchors=args.da_vggt_n_anchors,
        da_vggt_dino_batch_size=args.da_vggt_dino_batch_size,
        da_vggt_lambda_div=args.da_vggt_lambda_div,
    )
    for dataset in datasets:
        dataset_root = args.dataset_root or DEFAULT_DATASET_ROOTS[dataset]
        sequences = sequence_names(
            dataset,
            dataset_root,
            args.sequences,
            args.all_scenes,
            args.seven_scenes_split,
            args.bonn_rgb_dir,
        )
        output_root = Path(args.output_dir) / dataset

        for seq in tqdm(sequences, desc=f"VGGT-Omega {dataset} video depth"):
            images = limit_frames(
                sequence_images(dataset, dataset_root, seq, args.bonn_rgb_dir),
                args.max_frames_per_seq,
                args.frame_sample_mode,
                args.frame_sample_seed,
                seq,
            )
            output_dir = output_root / seq
            time_path = output_dir / "_time.json"
            if (
                not args.overwrite
                and time_path.is_file()
                and len(list(output_dir.glob("frame_[0-9][0-9][0-9][0-9].npy"))) == len(images)
            ):
                continue
            if device.type == "cuda":
                torch.cuda.empty_cache()
            output_dir.mkdir(parents=True, exist_ok=True)
            elapsed, depths, poses, _, resolution, speed_metrics = infer_sequence(
                images,
                model,
                device,
                window_size=args.window_size,
                image_resolution=args.image_resolution,
                input_mode=args.input_mode,
                use_amp=not args.no_amp,
            )
            similarity_matrix_files = save_frame_similarity_matrices(output_dir, speed_metrics)
            assert depths is not None
            for frame_idx, depth in enumerate(depths.numpy()):
                np.save(output_dir / f"frame_{frame_idx:04d}.npy", depth)
                save_depth_preview(depth, output_dir / f"frame_{frame_idx:04d}.png")
            if poses is not None:
                pred_poses = poses.numpy()
                np.save(output_dir / "pred_poses.npy", pred_poses)
                save_tum_trajectory(output_dir / "pred_traj.txt", pred_poses)
                gt_poses = sequence_poses(dataset, dataset_root, seq, len(images), images)
                if gt_poses is not None and len(gt_poses) == len(pred_poses):
                    if __package__:
                        from .pose_auc import evaluate_pose_auc, write_json
                    else:
                        from pose_auc import evaluate_pose_auc, write_json

                    pose_metrics = evaluate_pose_auc(
                        pred_poses,
                        gt_poses,
                        num_frames=args.pose_eval_frames,
                        seed=args.pose_eval_seed,
                        sequence=seq,
                    )
                    write_json(output_dir / "_pose_auc.json", pose_metrics)
            with time_path.open("w") as handle:
                json.dump(
                    {
                        **speed_metrics,
                        "resolution": list(resolution),
                        "window_size": args.window_size,
                        "input_mode": args.input_mode,
                        "image_resolution": args.image_resolution,
                        "bonn_rgb_dir": args.bonn_rgb_dir,
                        "bonn_depth_dir": args.bonn_depth_dir,
                        "pose_eval_frames": args.pose_eval_frames,
                        "pose_eval_seed": args.pose_eval_seed,
                        "frame_sample_mode": args.frame_sample_mode,
                        "frame_sample_seed": args.frame_sample_seed,
                        "sampled_frames": [Path(image).name for image in images],
                        "inter_frame_attention": args.inter_frame_attention,
                        "register_patch_sample_tokens": args.register_patch_sample_tokens,
                        "register_patch_sample_ratio": args.register_patch_sample_ratio,
                        "register_patch_sample_mode": args.register_patch_sample_mode,
                        "register_patch_merge_sources": args.register_patch_merge_sources,
                        "register_patch_merge_protect_first_frame": args.register_patch_merge_protect_first_frame,
                        "enable_token_merging": args.enable_token_merging,
                        "token_merging_method": args.token_merging_method,
                        "token_merging_start": args.token_merging_start,
                        "token_merging_ratio": args.token_merging_ratio,
                        "token_merging_layer_ratios": args.token_merging_layer_ratios,
                        "token_merging_tstm_threshold": args.token_merging_tstm_threshold,
                        "token_merging_tstm_neighbor_size": args.token_merging_tstm_neighbor_size,
                        "token_merging_flashvid_alpha": args.token_merging_flashvid_alpha,
                        "token_merging_flashvid_expansion": args.token_merging_flashvid_expansion,
                        "token_merging_flashvid_pool_stride": args.token_merging_flashvid_pool_stride,
                        "token_merging_frame_restore_layer": args.token_merging_frame_restore_layer,
                        "token_merging_frame_alpha": args.token_merging_frame_alpha,
                        "token_merging_frame_segment_threshold": args.token_merging_frame_segment_threshold,
                        "token_merging_frame_merge_threshold": args.token_merging_frame_merge_threshold,
                        "token_merging_frame_max_window": args.token_merging_frame_max_window,
                        "token_merging_frame_pool_stride": args.token_merging_frame_pool_stride,
                        "token_merging_frame_multi_max_group_size": args.token_merging_frame_multi_max_group_size,
                        "token_merging_frame_multi_pair_threshold": args.token_merging_frame_multi_pair_threshold,
                        "token_merging_frame_multi_span_threshold": args.token_merging_frame_multi_span_threshold,
                        "token_merging_frame_group_strategy": args.token_merging_frame_group_strategy,
                        "token_merging_frame_protect_period": args.token_merging_frame_protect_period,
                        "token_merging_frame_protect_prefix": args.token_merging_frame_protect_prefix,
                        "omega_accelerator": args.omega_accelerator,
                        "sparse_vggt_sparse_ratio": args.sparse_vggt_sparse_ratio,
                        "sparse_vggt_cdf_threshold": args.sparse_vggt_cdf_threshold,
                        "sparse_vggt_pool_mode": args.sparse_vggt_pool_mode,
                        "da_vggt_max_frames": args.da_vggt_max_frames,
                        "da_vggt_sampling_method": args.da_vggt_sampling_method,
                        "da_vggt_n_anchors": args.da_vggt_n_anchors,
                        "da_vggt_dino_batch_size": args.da_vggt_dino_batch_size,
                        "da_vggt_lambda_div": args.da_vggt_lambda_div,
                        "frame_similarity_matrix_files": similarity_matrix_files,
                        "elapsed": elapsed,
                    },
                    handle,
                        indent=2,
                    )
            if device.type == "cuda":
                torch.cuda.empty_cache()
        maybe_evaluate_dataset(args, dataset, dataset_root, output_root)


if __name__ == "__main__":
    main()
