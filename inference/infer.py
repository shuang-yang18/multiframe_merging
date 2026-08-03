"""Run VGGT-Omega video-depth inference on supported video benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm

if __package__:
    from .bonn_association import association_paths, build_bonn_associations
else:
    from bonn_association import association_paths, build_bonn_associations

from vggt_omega.evaluation import (
    INTER_FRAME_ATTENTION_MODES,
    infer_sequence,
    load_model,
    read_sintel_camera,
    save_depth_preview,
    sequence_images as sintel_sequence_images,
    sequence_names as sintel_sequence_names,
)

BONN_TEST_SEQUENCES = [
    "rgbd_bonn_balloon2",
    "rgbd_bonn_crowd2",
    "rgbd_bonn_crowd3",
    "rgbd_bonn_person_tracking2",
    "rgbd_bonn_synchronous",
]
BONN_SEQUENCES = BONN_TEST_SEQUENCES
DEFAULT_DATASET_ROOTS = {
    "sintel": "datasets/Sintel/training",
    "bonn": "datasets/Bonn/rgbd_bonn_dataset",
    "7scenes": "datasets/7scenes",
    "tum_dynamic": "datasets/TUM-Dynamics",
    "nrgbd": "datasets/NRGBD",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["sintel", "bonn", "7scenes", "tum_dynamic", "nrgbd", "all"], default="all")
    parser.add_argument("--dataset-root")
    parser.add_argument("--bonn-rgb-dir", default="rgb_110", help="Bonn RGB subdirectory; use 'rgb' for full-length sequences.")
    parser.add_argument("--bonn-depth-dir", default="depth_110", help="Bonn depth subdirectory; use 'depth' for full-length sequences.")
    parser.add_argument(
        "--bonn-split",
        choices=["test", "all"],
        default="test",
        help="Bonn split to use. Defaults to the five-sequence benchmark test split.",
    )
    parser.add_argument(
        "--bonn-association-max-diff",
        type=float,
        default=0.02,
        help="Maximum RGB/depth or RGB/pose timestamp difference in seconds for Bonn full streams.",
    )
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
            "flashvid_encoder",
            "frame_temporary",
            "frame_persistent",
            "frame_persistent_spatial",
            "frame_persistent_adaptive",
            "frame_persistent_adaptive_spatial",
            "frame_temporary_adaptive_spatial",
            "frame_staged_adaptive_spatial",
            "token_only_adaptive_spatial",
            "frame_persistent_decoupled",
            "frame_persistent_decoupled_window",
            "frame_anchor_hybrid",
            "frame_anchor_adaptive",
            "frame_anchor_adaptive_spatial",
            "dynamic_spatial",
            "dynamic_spatial_hybrid",
            "segment_patch_bank",
        ],
        default="spatial",
        help="Token merging strategy used inside global inter-frame attention blocks.",
    )
    parser.add_argument(
        "--dynamic-fastvggt-schedule",
        choices=["all", "middle", "late", "middle_late"],
        default="all",
        help="Global layers that run dynamic-aware merging after layer-4 segmentation.",
    )
    parser.add_argument(
        "--skip-global-attention-blocks",
        default="",
        help="Comma-separated 0-based global block indices/ranges to skip, e.g. '0-8'.",
    )
    parser.add_argument(
        "--skip-inter-frame-attention-blocks",
        default="",
        help="Comma-separated 0-based inter-frame block indices/ranges to skip, regardless of attention type.",
    )
    parser.add_argument(
        "--frame-only-inter-frame-blocks",
        default="",
        help="Run listed 0-based inter-frame blocks independently within each frame instead of across frames.",
    )
    parser.add_argument(
        "--register-only-blocks",
        default="",
        help="Override register-only attention blocks with 0-based indices/ranges, e.g. '1-7,9,14,20'.",
    )
    parser.add_argument(
        "--enable-adaptive-frame-token-fusion",
        action="store_true",
        help="Enable the independent adaptive frame fusion plus token-merging path in global attention blocks.",
    )
    parser.add_argument(
        "--adaptive-frame-representation",
        choices=["global_pool", "cluster_center", "spatial_grid"],
        default="global_pool",
    )
    parser.add_argument("--adaptive-representation-pca-dim", type=int, default=512)
    parser.add_argument("--adaptive-representation-clusters", type=int, choices=[2, 3, 4], default=3)
    parser.add_argument("--adaptive-spatial-grid", type=int, default=4)
    parser.add_argument("--adaptive-grouping", choices=["serial", "parallel"], default="serial")
    parser.add_argument("--adaptive-reference-selection", choices=["first", "medoid", "diverse"], default="first")
    parser.add_argument(
        "--adaptive-reference-excluded",
        action="store_true",
        help="Protect reference frames from fusion; other group members still form a separate fused active frame.",
    )
    parser.add_argument("--adaptive-group-similarity-threshold", type=float, default=0.98)
    parser.add_argument("--adaptive-group-max-size", type=int, default=4)
    parser.add_argument("--adaptive-parallel-window", type=int, default=10)
    parser.add_argument(
        "--adaptive-update-policy",
        choices=["initial_only", "stage_update", "per_layer_update"],
        default="initial_only",
    )
    parser.add_argument("--adaptive-update-after-blocks", default="9,17")
    parser.add_argument("--adaptive-frame-fusion", choices=["direct", "token_wise"], default="direct")
    parser.add_argument("--adaptive-frame-fusion-weighting", choices=["uniform", "similarity"], default="similarity")
    parser.add_argument("--adaptive-frame-token-similarity-threshold", type=float, default=0.95)
    parser.add_argument(
        "--adaptive-token-merging",
        choices=["fast_bipartite", "category_topk_norm"],
        default="fast_bipartite",
    )
    parser.add_argument("--adaptive-token-keep-ratio", type=float, default=0.1)
    parser.add_argument("--adaptive-token-clusters", type=int, choices=[3, 4, 5], default=4)
    parser.add_argument("--adaptive-token-cluster-budget", choices=["proportional", "dispersion"], default="proportional")
    parser.add_argument("--adaptive-token-kmeans-iterations", type=int, default=12)
    parser.add_argument("--token-merging-flashvid-alpha", type=float, default=0.7)
    parser.add_argument("--token-merging-flashvid-expansion", type=float, default=1.25)
    parser.add_argument("--token-merging-flashvid-pool-stride", type=int, default=2)
    parser.add_argument("--token-merging-flashvid-tstm-threshold", type=float, default=0.8)
    parser.add_argument(
        "--token-merging-fastvggt-destination-selector",
        choices=["random", "local_medoid"],
        default="random",
        help=(
            "Choose 2x2 FastVGGT destination patches by fixed random sampling or "
            "local Q-feature medoids with value-norm tie breaking."
        ),
    )
    parser.add_argument(
        "--token-merging-fastvggt-destination-policy",
        choices=["grid_2x2", "global_25pct"],
        default="grid_2x2",
        help="Use one destination per 2x2 patch block or an equal-size global patch destination set.",
    )
    parser.add_argument(
        "--token-merging-fastvggt-uniform-protect-ratio",
        type=float,
        default=0.0,
        help="Additional uniformly protected token fraction used by spatial FastVGGT.",
    )
    parser.add_argument(
        "--token-merging-fastvggt-exclusive-protection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Protect only source candidates so protected tokens never duplicate destination tokens.",
    )
    parser.add_argument(
        "--token-merging-fastvggt-protect-anchor-frames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep all patch tokens from shared-anchor chunk frames out of FastVGGT source merging.",
    )
    parser.add_argument("--token-merging-frame-restore-layer", type=int, default=16)
    parser.add_argument("--token-merging-frame-alpha", type=float, default=0.9)
    parser.add_argument("--token-merging-frame-segment-threshold", type=float, default=0.8)
    parser.add_argument("--token-merging-frame-merge-threshold", type=float, default=0.8)
    parser.add_argument("--token-merging-frame-max-window", type=int, default=6)
    parser.add_argument("--token-merging-frame-pool-stride", type=int, default=2)
    parser.add_argument("--token-merging-frame-multi-max-group-size", type=int, default=2)
    parser.add_argument("--token-merging-frame-multi-pair-threshold", type=float, default=0.95)
    parser.add_argument("--token-merging-frame-multi-span-threshold", type=float, default=0.93)
    parser.add_argument(
        "--token-merging-frame-upper-adaptive",
        action="store_true",
        default=False,
        help="For frame fusion only, raise pair/span thresholds together to target a 45%% merge rate above 50%% raw fusion.",
    )
    parser.add_argument(
        "--token-merging-frame-staged-ranges",
        default="0-9,10-17,18-23",
        help="Comma-separated 0-based closed block ranges for frame_staged_adaptive_spatial.",
    )
    parser.add_argument("--token-merging-frame-staged-late-segment-threshold", type=float)
    parser.add_argument("--token-merging-frame-staged-late-pair-threshold", type=float)
    parser.add_argument("--token-merging-frame-staged-late-span-threshold", type=float)
    parser.add_argument("--token-merging-frame-protect-period", type=int, default=0)
    parser.add_argument("--token-merging-frame-protect-prefix", type=int, default=0)
    parser.add_argument(
        "--token-merging-frame-anchor-count",
        type=int,
        default=4,
        help="Number of full global anchor frames retained by frame_anchor_hybrid.",
    )
    parser.add_argument(
        "--token-merging-frame-anchor-selection",
        choices=["uniform", "farthest"],
        default="uniform",
        help="Global anchor selection for frame_anchor_hybrid.",
    )
    parser.add_argument(
        "--token-merging-frame-adaptive-boundary-z",
        type=float,
        default=2.5,
        help="Robust-MAD z multiplier for adaptive segment change-point detection.",
    )
    parser.add_argument(
        "--token-merging-frame-adaptive-medoid-z",
        type=float,
        default=1.5,
        help="Robust-MAD z multiplier for the within-segment medoid residual budget.",
    )
    parser.add_argument(
        "--token-merging-frame-patch-fusion-quantile",
        type=float,
        default=0.75,
        help="Keep only this upper quantile of mutual patch-correspondence confidences when fusing a medoid group.",
    )
    parser.add_argument(
        "--token-merging-frame-special-cross-attention",
        action="store_true",
        help="Let active camera/register tokens cross-attend to pre-merge full-frame special-token memory.",
    )
    parser.add_argument("--token-merging-frame-special-cross-attention-alpha", type=float, default=0.1)
    parser.add_argument("--token-merging-segment-bank-pair-threshold", type=float, default=0.986)
    parser.add_argument("--token-merging-segment-bank-span-threshold", type=float, default=0.948)
    parser.add_argument("--token-merging-segment-bank-max-group-size", type=int, choices=[3, 4], default=4)
    parser.add_argument(
        "--token-merging-frame-group-strategy",
        choices=["local", "segment_middle", "global_cluster", "global_top_pairs"],
        default="local",
        help="Frame grouping strategy inside each segment.",
    )
    parser.add_argument(
        "--omega-accelerator",
        choices=[
            "none",
            "da_vggt",
            "sparse_vggt",
            "shared_anchor_chunks",
            "da_chunk_strided_shared_anchor",
        ],
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
    parser.add_argument("--da-chunk-strided-groups", type=int, default=5)
    parser.add_argument("--da-chunk-strided-anchor-count", type=int, default=5)
    parser.add_argument("--shared-anchor-num-chunks", type=int, default=10)
    parser.add_argument("--shared-anchor-count", type=int, default=10)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--eval",
        dest="eval",
        action="store_true",
        default=True,
        help="Run video-depth and pose-AUC evaluation after inference (the default).",
    )
    parser.add_argument(
        "--no-eval",
        dest="eval",
        action="store_false",
        help="Skip both video-depth and pose-AUC evaluation.",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="With --no-eval, write only input/timing metadata and skip prediction, trajectory, and pose files.",
    )
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
    parser.add_argument(
        "--max-frames-per-seq",
        type=int,
        default=0,
        help="Uniformly sample this many temporally ordered frames over each full sequence; 0 keeps all frames.",
    )
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
    bonn_split: str = "test",
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
    if dataset == "nrgbd":
        root = Path(dataset_root)
        if all_scenes and not requested:
            requested = sorted(path.name for path in root.iterdir() if path.is_dir())
        requested = requested or sorted(path.name for path in root.iterdir() if path.is_dir())
        missing = [
            seq
            for seq in requested
            if not ((root / seq / "images").is_dir() and (root / seq / "depth").is_dir() and (root / seq / "poses.txt").is_file())
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing NRGBD images/depth/poses.txt below {dataset_root}: {missing}"
            )
        return requested
    if requested:
        requested = requested
    elif bonn_split == "all":
        requested = sorted(path.name for path in Path(dataset_root).iterdir() if (path / bonn_rgb_dir).is_dir())
    else:
        requested = BONN_TEST_SEQUENCES
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
    if dataset == "nrgbd":
        paths = sorted(
            (Path(dataset_root) / seq / "images").glob("img*.png"),
            key=lambda path: int(path.stem.removeprefix("img")),
        )
        if not paths:
            raise FileNotFoundError(f"No NRGBD input images found for sequence {seq}")
        return [str(path) for path in paths]
    paths = sorted((Path(dataset_root) / seq / bonn_rgb_dir).glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No Bonn input images found for sequence {seq} in {bonn_rgb_dir}")
    return [str(path) for path in paths]


def uniform_frame_indices(num_frames: int, max_frames: int) -> list[int]:
    """Select temporally ordered views spanning the complete input sequence."""
    if num_frames < 0:
        raise ValueError("num_frames must be non-negative")
    if not max_frames or max_frames <= 0 or num_frames <= max_frames:
        return list(range(num_frames))
    indices = np.linspace(0, num_frames - 1, num=max_frames, dtype=np.int64).tolist()
    if len(indices) != max_frames or len(set(indices)) != max_frames:
        raise RuntimeError(f"Uniform selection failed for {num_frames=} and {max_frames=}")
    return indices


def select_uniform_frames(paths: list[str], max_frames: int) -> tuple[list[str], list[int]]:
    indices = uniform_frame_indices(len(paths), max_frames)
    return [paths[index] for index in indices], indices


def input_frames_manifest(source_count: int, selected_indices: list[int], images: list[str]) -> dict:
    strides = np.diff(np.asarray(selected_indices, dtype=np.float64))
    return {
        "protocol": "uniform_full_sequence_v1",
        "source_frame_count": source_count,
        "selected_frame_count": len(images),
        "selected_source_indices": selected_indices,
        "mean_source_stride": float(strides.mean()) if len(strides) else 0.0,
        "sampled_frames": [Path(image).name for image in images],
    }


def load_manifest_selected_images(output_dir: str | Path, source_images: list[str]) -> tuple[list[str], list[int]]:
    """Recover exactly the RGB frames used by inference from its persisted manifest."""
    manifest_path = Path(output_dir) / "_input_frames.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing input-frame manifest {manifest_path}; rerun inference with the uniform_full_sequence_v1 protocol."
        )
    with manifest_path.open() as handle:
        manifest = json.load(handle)
    if manifest.get("protocol") != "uniform_full_sequence_v1":
        raise ValueError(f"Unsupported frame-selection protocol in {manifest_path}: {manifest.get('protocol')!r}")
    indices = manifest.get("selected_source_indices")
    if not isinstance(indices, list) or any(not isinstance(index, int) for index in indices):
        raise ValueError(f"Invalid selected_source_indices in {manifest_path}")
    if manifest.get("source_frame_count") != len(source_images):
        raise ValueError(
            f"{manifest_path}: source frame count {manifest.get('source_frame_count')} does not match "
            f"the current dataset ({len(source_images)})."
        )
    if any(index < 0 or index >= len(source_images) for index in indices):
        raise ValueError(f"{manifest_path}: selected source index is out of bounds")
    images = [source_images[index] for index in indices]
    if manifest.get("sampled_frames") != [Path(image).name for image in images]:
        raise ValueError(f"{manifest_path}: sampled frame names no longer match the dataset")
    return images, indices


def bonn_associated_frames(
    dataset_root: str | Path,
    seq: str,
    *,
    rgb_dir: str,
    depth_dir: str,
    max_difference: float,
    max_frames: int,
) -> tuple[list[dict], list[str], list[str], list[int], int]:
    associations = build_bonn_associations(
        dataset_root,
        seq,
        rgb_dir=rgb_dir,
        depth_dir=depth_dir,
        max_difference=max_difference,
    )
    source_count = len(associations)
    indices = uniform_frame_indices(source_count, max_frames)
    associations = [associations[index] for index in indices]
    images, depths = association_paths(dataset_root, seq, associations, rgb_dir=rgb_dir, depth_dir=depth_dir)
    return associations, images, depths, indices, source_count


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
    pose_indices: list[int] | None = None,
) -> np.ndarray | None:
    root = Path(dataset_root)
    if dataset == "sintel":
        if image_paths is None:
            camera_paths = sorted((root / "camdata_left" / seq).glob("*.cam"))[:frame_count]
        else:
            camera_dir = root / "camdata_left" / seq
            camera_paths = [camera_dir / f"{Path(image_path).stem}.cam" for image_path in image_paths[:frame_count]]
        if len(camera_paths) != frame_count or any(not path.is_file() for path in camera_paths):
            raise FileNotFoundError(f"Missing Sintel camera annotations for {seq}")
        return np.stack([read_sintel_camera(path) for path in camera_paths])
    if dataset == "7scenes":
        if image_paths is not None:
            paths = [Path(path).with_name(Path(path).name.replace(".color.png", ".pose.txt")) for path in image_paths]
        else:
            paths = sorted((root / seq).glob("*.pose.txt"))[:frame_count]
        if not paths:
            return None
        return np.stack([np.loadtxt(path) for path in paths])
    if dataset in {"tum_dynamic", "bonn"}:
        path = root / seq / "groundtruth.txt"
        if not path.is_file():
            return None
        if pose_indices is not None:
            _, poses = _read_tum_pose_entries(path)
            return poses[np.asarray(pose_indices, dtype=np.int64)]
        if image_paths is None:
            return _read_tum_poses(path)[:frame_count]
        pose_times, poses = _read_tum_pose_entries(path)
        matched = []
        for image_path in image_paths[:frame_count]:
            idx = int(np.argmin(np.abs(pose_times - _timestamp(image_path))))
            matched.append(poses[idx])
        return np.stack(matched)
    if dataset == "nrgbd":
        pose_path = root / seq / "poses.txt"
        if not pose_path.is_file():
            return None
        values = np.loadtxt(pose_path, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 4 or values.shape[0] % 4:
            raise ValueError(f"Expected 4N x 4 NRGBD poses in {pose_path}, got {values.shape}")
        poses = values.reshape(-1, 4, 4)
        # NRGBD poses use OpenGL camera axes; VGGT predictions are OpenCV c2w.
        poses[:, :3, 1:3] *= -1.0
        if image_paths is None:
            return poses[:frame_count]
        indices = [int(Path(path).stem.removeprefix("img")) for path in image_paths[:frame_count]]
        return poses[np.asarray(indices, dtype=np.int64)]
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
    # This ratio is already reduced within each sequence by the inference path.
    # Keep its dataset aggregation sequence-uniform, rather than pixel-weighted.
    final_token_over_initial_token_ratios = []
    frame_token_ratios = []
    sequences = sequence_names(
        dataset,
        dataset_root,
        args.sequences,
        args.all_scenes,
        args.seven_scenes_split,
        args.bonn_rgb_dir,
        args.bonn_split,
    )
    for seq in tqdm(sequences, desc=f"VGGT-Omega {dataset} video depth eval"):
        seq_dir = output_root / seq
        if dataset == "bonn":
            association_path = seq_dir / "_bonn_associations.json"
            if not association_path.is_file():
                raise ValueError(
                    f"{dataset}/{seq}: missing timestamp association manifest {association_path}; "
                    "rerun Bonn inference so prediction frame indices match RGB/depth/pose triples."
                )
            with association_path.open() as handle:
                manifest = json.load(handle)
            associations = manifest["associations"]
            images, gt_paths = association_paths(
                dataset_root,
                seq,
                associations,
                rgb_dir=manifest["rgb_dir"],
                depth_dir=manifest["depth_dir"],
            )
        else:
            source_images = sequence_images(dataset, dataset_root, seq, args.bonn_rgb_dir)
            images, selected_indices = load_manifest_selected_images(seq_dir, source_images)
            all_gt_paths = sequence_depths(
                dataset,
                dataset_root,
                seq,
                args.bonn_depth_dir,
                args.bonn_rgb_dir,
                image_paths=images if dataset == "tum_dynamic" else None,
            )
            if dataset == "tum_dynamic":
                gt_paths = all_gt_paths
            else:
                if len(all_gt_paths) != len(source_images):
                    raise ValueError(
                        f"{dataset}/{seq}: {len(all_gt_paths)} depth frames do not match "
                        f"{len(source_images)} RGB frames."
                    )
                gt_paths = [all_gt_paths[index] for index in selected_indices]
        pred_paths = sorted(seq_dir.glob("frame_[0-9][0-9][0-9][0-9].npy"))
        if not (len(pred_paths) == len(gt_paths) == len(images)):
            raise ValueError(
                f"{dataset}/{seq}: found {len(pred_paths)} predictions, "
                f"{len(gt_paths)} ground truth depths, and {len(images)} images"
            )
        with (seq_dir / "_time.json").open() as handle:
            timing = json.load(handle)
        final_token_ratio = timing.get("adaptive_fusion_token_over_pre_frame_token_ratio_mean")
        if isinstance(final_token_ratio, (int, float)) and np.isfinite(final_token_ratio):
            final_token_over_initial_token_ratios.append(float(final_token_ratio))
        frame_token_ratio = timing.get("adaptive_fusion_frame_token_ratio_mean")
        if isinstance(frame_token_ratio, (int, float)) and np.isfinite(frame_token_ratio):
            frame_token_ratios.append(float(frame_token_ratio))
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
        row["token_merging_frame_special_cross_attention"] = timing.get(
            "token_merging_frame_special_cross_attention"
        )
        row["token_merging_frame_special_cross_attention_alpha"] = timing.get(
            "token_merging_frame_special_cross_attention_alpha"
        )
        row["omega_accelerator"] = timing.get("omega_accelerator")
        row["sparse_vggt_sparsity_mean"] = timing.get("sparse_vggt_sparsity_mean")
        row["sparse_vggt_sparse_ratio"] = timing.get("sparse_vggt_sparse_ratio")
        row["sparse_vggt_cdf_threshold"] = timing.get("sparse_vggt_cdf_threshold")
        row["sparse_vggt_pool_mode"] = timing.get("sparse_vggt_pool_mode")
        row["da_vggt_num_chunks_mean"] = timing.get("da_vggt_num_chunks_mean")
        row["da_vggt_chunk_size"] = timing.get("da_vggt_chunk_size")
        row["da_vggt_sampling_method"] = timing.get("da_vggt_sampling_method")
        row["shared_anchor_num_chunks_mean"] = timing.get("shared_anchor_num_chunks_mean")
        row["shared_anchor_chunk_size"] = timing.get("shared_anchor_chunk_size")
        row["shared_anchor_count"] = timing.get("shared_anchor_count")
        row["shared_anchor_selection"] = timing.get("shared_anchor_selection")
        row["frame_merge_active_frames_mean"] = timing.get("frame_merge_active_frames_mean")
        row["frame_merge_events"] = timing.get("frame_merge_events")
        row["frame_merge_retention_ratio_mean"] = timing.get("frame_merge_retention_ratio_mean")
        row["frame_merge_merge_ratio_mean"] = timing.get("frame_merge_merge_ratio_mean")
        row["frame_merge_raw_merge_ratio_mean"] = timing.get("frame_merge_raw_merge_ratio_mean")
        row["frame_merge_adaptive_policy"] = timing.get("frame_merge_adaptive_policy")
        row["frame_merge_selected_pair_threshold_mean"] = timing.get("frame_merge_selected_pair_threshold_mean")
        row["frame_merge_selected_span_threshold_mean"] = timing.get("frame_merge_selected_span_threshold_mean")
        row["frame_merge_anchor_count"] = timing.get("frame_merge_anchor_count")
        row["frame_merge_anchor_selection"] = timing.get("frame_merge_anchor_selection")
        row["frame_fusion_cuda_ms_mean"] = timing.get("frame_fusion_cuda_ms_mean")
        row["frame_fusion_cuda_ms_total"] = timing.get("frame_fusion_cuda_ms_total")
        row["frame_fusion_host_wall_ms_mean"] = timing.get("frame_fusion_host_wall_ms_mean")
        row["frame_fusion_host_wall_ms_total"] = timing.get("frame_fusion_host_wall_ms_total")
        row["frame_special_cross_attention_events"] = timing.get("frame_special_cross_attention_events")
        row["frame_special_cross_attention_alpha"] = timing.get("frame_special_cross_attention_alpha")
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
            "dynamic_fastvggt_schedule",
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
            "token_merging_frame_special_cross_attention",
            "token_merging_frame_special_cross_attention_alpha",
            "frame_special_cross_attention_alpha",
            "frame_merge_anchor_selection",
            "frame_merge_events",
            "frame_fusion_cuda_ms_mean",
            "frame_fusion_cuda_ms_total",
            "frame_fusion_host_wall_ms_mean",
            "frame_fusion_host_wall_ms_total",
            "omega_accelerator",
            "sparse_vggt_pool_mode",
            "da_vggt_sampling_method",
            "depth_eval_protocol",
        }
        and all(isinstance(row.get(key), (int, float)) and row.get(key) is not None for row in sequence_rows)
    }
    total_frames = int(sum(row["frames"] for row in sequence_rows))
    total_time = float(sum(row["time"] for row in sequence_rows))
    frame_fusion_events = sum(int(row.get("frame_merge_events") or 0) for row in sequence_rows)
    for name in ("frame_fusion_cuda_ms", "frame_fusion_host_wall_ms"):
        totals = [row.get(f"{name}_total") for row in sequence_rows]
        if frame_fusion_events and all(isinstance(value, (int, float)) for value in totals):
            total = float(sum(totals))
            summary[f"{name}_total"] = total
            summary[f"{name}_mean"] = total / frame_fusion_events
    summary["adaptive_fusion_final_token_over_initial_token_ratio_sequence_mean"] = (
        float(np.mean(final_token_over_initial_token_ratios))
        if final_token_over_initial_token_ratios
        else None
    )
    summary["adaptive_fusion_final_token_over_initial_token_ratio_sequence_count"] = len(
        final_token_over_initial_token_ratios
    )
    summary["adaptive_fusion_frame_token_ratio_sequence_mean"] = (
        float(np.mean(frame_token_ratios)) if frame_token_ratios else None
    )
    summary["adaptive_fusion_frame_token_ratio_sequence_count"] = len(frame_token_ratios)
    summary.update(
        {
            "dataset": dataset,
            "sequences": len(sequence_rows),
            "frames": total_frames,
            "time": total_time,
            # Dataset throughput is total inferred frames divided by total model time.
            "fps": float(total_frames / total_time) if total_time > 0 else 0.0,
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
            "dynamic_fastvggt_schedule": args.dynamic_fastvggt_schedule,
            "skip_global_attention_blocks": args.skip_global_attention_blocks,
            "skip_inter_frame_attention_blocks": args.skip_inter_frame_attention_blocks,
            "frame_only_inter_frame_blocks": args.frame_only_inter_frame_blocks,
            "register_only_blocks": args.register_only_blocks,
            "adaptive_frame_token_fusion": {
                "enabled": args.enable_adaptive_frame_token_fusion,
                "frame_representation": args.adaptive_frame_representation,
                "representation_pca_dim": args.adaptive_representation_pca_dim,
                "representation_clusters": args.adaptive_representation_clusters,
                "spatial_grid": args.adaptive_spatial_grid,
                "grouping": args.adaptive_grouping,
                "reference_selection": args.adaptive_reference_selection,
                "reference_participates": not args.adaptive_reference_excluded,
                "group_similarity_threshold": args.adaptive_group_similarity_threshold,
                "group_max_size": args.adaptive_group_max_size,
                "parallel_window": args.adaptive_parallel_window,
                "update_policy": args.adaptive_update_policy,
                "update_after_blocks": args.adaptive_update_after_blocks,
                "frame_fusion": args.adaptive_frame_fusion,
                "frame_fusion_weighting": args.adaptive_frame_fusion_weighting,
                "frame_special_tokens": "protected",
                "frame_token_similarity_threshold": args.adaptive_frame_token_similarity_threshold,
                "token_merging": args.adaptive_token_merging,
                "token_keep_ratio": args.adaptive_token_keep_ratio,
                "token_clusters": args.adaptive_token_clusters,
                "token_cluster_budget": args.adaptive_token_cluster_budget,
            },
            "token_merging_flashvid_alpha": args.token_merging_flashvid_alpha,
            "token_merging_flashvid_expansion": args.token_merging_flashvid_expansion,
            "token_merging_flashvid_pool_stride": args.token_merging_flashvid_pool_stride,
            "token_merging_flashvid_tstm_threshold": args.token_merging_flashvid_tstm_threshold,
            "token_merging_fastvggt_destination_selector": args.token_merging_fastvggt_destination_selector,
            "token_merging_fastvggt_destination_policy": args.token_merging_fastvggt_destination_policy,
            "token_merging_fastvggt_uniform_protect_ratio": args.token_merging_fastvggt_uniform_protect_ratio,
            "token_merging_fastvggt_exclusive_protection": args.token_merging_fastvggt_exclusive_protection,
            "token_merging_fastvggt_protect_anchor_frames": args.token_merging_fastvggt_protect_anchor_frames,
            "token_merging_frame_restore_layer": args.token_merging_frame_restore_layer,
            "token_merging_frame_alpha": args.token_merging_frame_alpha,
            "token_merging_frame_segment_threshold": args.token_merging_frame_segment_threshold,
            "token_merging_frame_merge_threshold": args.token_merging_frame_merge_threshold,
            "token_merging_frame_max_window": args.token_merging_frame_max_window,
            "token_merging_frame_pool_stride": args.token_merging_frame_pool_stride,
            "token_merging_frame_multi_max_group_size": args.token_merging_frame_multi_max_group_size,
            "token_merging_frame_multi_pair_threshold": args.token_merging_frame_multi_pair_threshold,
            "token_merging_frame_multi_span_threshold": args.token_merging_frame_multi_span_threshold,
            "token_merging_frame_upper_adaptive": args.token_merging_frame_upper_adaptive,
            "token_merging_frame_staged_ranges": args.token_merging_frame_staged_ranges,
            "token_merging_frame_staged_late_segment_threshold": args.token_merging_frame_staged_late_segment_threshold,
            "token_merging_frame_staged_late_pair_threshold": args.token_merging_frame_staged_late_pair_threshold,
            "token_merging_frame_staged_late_span_threshold": args.token_merging_frame_staged_late_span_threshold,
            "token_merging_frame_group_strategy": args.token_merging_frame_group_strategy,
            "token_merging_frame_protect_period": args.token_merging_frame_protect_period,
            "token_merging_frame_protect_prefix": args.token_merging_frame_protect_prefix,
            "token_merging_frame_special_cross_attention": args.token_merging_frame_special_cross_attention,
            "token_merging_frame_special_cross_attention_alpha": args.token_merging_frame_special_cross_attention_alpha,
            "omega_accelerator": args.omega_accelerator,
            "sparse_vggt_sparse_ratio": args.sparse_vggt_sparse_ratio
            if args.omega_accelerator == "sparse_vggt"
            else None,
            "sparse_vggt_cdf_threshold": args.sparse_vggt_cdf_threshold
            if args.omega_accelerator == "sparse_vggt"
            else None,
            "sparse_vggt_pool_mode": args.sparse_vggt_pool_mode if args.omega_accelerator == "sparse_vggt" else None,
            "da_vggt_max_frames": (
                args.da_vggt_max_frames
                if args.omega_accelerator in {"da_vggt", "da_chunk_strided_shared_anchor"}
                else None
            ),
            "da_vggt_sampling_method": (
                args.da_vggt_sampling_method
                if args.omega_accelerator in {"da_vggt", "da_chunk_strided_shared_anchor"}
                else None
            ),
            "da_vggt_n_anchors": args.da_vggt_n_anchors if args.omega_accelerator == "da_vggt" else None,
            "da_chunk_strided_groups": (
                args.da_chunk_strided_groups
                if args.omega_accelerator == "da_chunk_strided_shared_anchor"
                else None
            ),
            "da_chunk_strided_anchor_count": (
                args.da_chunk_strided_anchor_count
                if args.omega_accelerator == "da_chunk_strided_shared_anchor"
                else None
            ),
            "da_vggt_dino_batch_size": (
                args.da_vggt_dino_batch_size
                if args.omega_accelerator in {"da_vggt", "da_chunk_strided_shared_anchor"}
                else None
            ),
            "da_vggt_lambda_div": (
                args.da_vggt_lambda_div
                if args.omega_accelerator in {"da_vggt", "da_chunk_strided_shared_anchor"}
                else None
            ),
            "shared_anchor_num_chunks": (
                args.shared_anchor_num_chunks if args.omega_accelerator == "shared_anchor_chunks" else None
            ),
            "shared_anchor_count": args.shared_anchor_count if args.omega_accelerator == "shared_anchor_chunks" else None,
            "eval_align": args.eval_align,
            "valid_pixels": int(weights.sum()),
        }
    )
    summary["depth_eval_protocol"] = sequence_rows[0]["depth_eval_protocol"]
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
        if dataset == "bonn":
            association_path = output_root / seq / "_bonn_associations.json"
            if not association_path.is_file():
                raise ValueError(f"{dataset}/{seq}: missing timestamp association manifest {association_path}")
            with association_path.open() as handle:
                manifest = json.load(handle)
            images, _ = association_paths(
                dataset_root,
                seq,
                manifest["associations"],
                rgb_dir=manifest["rgb_dir"],
                depth_dir=manifest["depth_dir"],
            )
            pose_indices = [item["pose_index"] for item in manifest["associations"]]
        else:
            source_images = sequence_images(dataset, dataset_root, seq, args.bonn_rgb_dir)
            images, _ = load_manifest_selected_images(output_root / seq, source_images)
            pose_indices = None
        gt_poses = sequence_poses(dataset, dataset_root, seq, len(images), images, pose_indices=pose_indices)
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
                "frame_merge_raw_merge_ratio_mean": summary.get("frame_merge_raw_merge_ratio_mean"),
                "frame_merge_adaptive_policy": summary.get("frame_merge_adaptive_policy"),
                "frame_merge_selected_pair_threshold_mean": summary.get(
                    "frame_merge_selected_pair_threshold_mean"
                ),
                "frame_merge_selected_span_threshold_mean": summary.get(
                    "frame_merge_selected_span_threshold_mean"
                ),
                "frame_merge_anchor_count": summary.get("frame_merge_anchor_count"),
                "frame_merge_anchor_selection": summary.get("frame_merge_anchor_selection"),
                "frame_fusion_cuda_ms_mean": summary.get("frame_fusion_cuda_ms_mean"),
                "frame_fusion_cuda_ms_total": summary.get("frame_fusion_cuda_ms_total"),
                "frame_fusion_host_wall_ms_mean": summary.get("frame_fusion_host_wall_ms_mean"),
                "frame_fusion_host_wall_ms_total": summary.get("frame_fusion_host_wall_ms_total"),
                "frame_special_cross_attention_events": summary.get("frame_special_cross_attention_events"),
                "frame_special_cross_attention_alpha": summary.get("frame_special_cross_attention_alpha"),
                "token_merging_active_over_frame_merged_token_ratio_mean": summary.get(
                    "token_merging_active_over_frame_merged_token_ratio_mean"
                ),
                "token_merging_active_over_frame_original_token_ratio_mean": summary.get(
                    "token_merging_active_over_frame_original_token_ratio_mean"
                ),
                "token_merging_full_attention_token_ratio_mean": summary.get(
                    "token_merging_full_attention_token_ratio_mean"
                ),
                "adaptive_fusion_final_token_over_initial_token_ratio_sequence_mean": summary.get(
                    "adaptive_fusion_final_token_over_initial_token_ratio_sequence_mean"
                ),
                "adaptive_fusion_final_token_over_initial_token_ratio_sequence_count": summary.get(
                    "adaptive_fusion_final_token_over_initial_token_ratio_sequence_count"
                ),
                "adaptive_fusion_frame_token_ratio_sequence_mean": summary.get(
                    "adaptive_fusion_frame_token_ratio_sequence_mean"
                ),
                "adaptive_fusion_frame_token_ratio_sequence_count": summary.get(
                    "adaptive_fusion_frame_token_ratio_sequence_count"
                ),
                "dynamic_fastvggt_patch_merge_ratio": summary.get("dynamic_fastvggt_patch_merge_ratio"),
                "dynamic_fastvggt_dynamic_merge_ratio": summary.get("dynamic_fastvggt_dynamic_merge_ratio"),
                "dynamic_fastvggt_static_merge_ratio": summary.get("dynamic_fastvggt_static_merge_ratio"),
                "dynamic_fastvggt_cross_type_merged_tokens": summary.get(
                    "dynamic_fastvggt_cross_type_merged_tokens"
                ),
            },
        },
    )
    print(f"{dataset} {args.eval_align}: {summary}")


def main() -> None:
    args = parse_args()
    if args.stats_only and args.eval:
        raise ValueError("--stats-only requires --no-eval")
    if args.dataset == "all" and args.dataset_root:
        raise ValueError("--dataset-root cannot be used with --dataset all; use the default dataset roots.")
    datasets = ["sintel", "bonn", "7scenes", "tum_dynamic", "nrgbd"] if args.dataset == "all" else [args.dataset]
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
        token_merging_flashvid_alpha=args.token_merging_flashvid_alpha,
        token_merging_flashvid_expansion=args.token_merging_flashvid_expansion,
        token_merging_flashvid_pool_stride=args.token_merging_flashvid_pool_stride,
        token_merging_flashvid_tstm_threshold=args.token_merging_flashvid_tstm_threshold,
        token_merging_fastvggt_destination_selector=args.token_merging_fastvggt_destination_selector,
        token_merging_fastvggt_destination_policy=args.token_merging_fastvggt_destination_policy,
        token_merging_fastvggt_uniform_protect_ratio=args.token_merging_fastvggt_uniform_protect_ratio,
        token_merging_fastvggt_exclusive_protection=args.token_merging_fastvggt_exclusive_protection,
        token_merging_fastvggt_protect_anchor_frames=args.token_merging_fastvggt_protect_anchor_frames,
        token_merging_frame_restore_layer=args.token_merging_frame_restore_layer,
        token_merging_frame_alpha=args.token_merging_frame_alpha,
        token_merging_frame_segment_threshold=args.token_merging_frame_segment_threshold,
        token_merging_frame_merge_threshold=args.token_merging_frame_merge_threshold,
        token_merging_frame_max_window=args.token_merging_frame_max_window,
        token_merging_frame_pool_stride=args.token_merging_frame_pool_stride,
        token_merging_frame_multi_max_group_size=args.token_merging_frame_multi_max_group_size,
        token_merging_frame_multi_pair_threshold=args.token_merging_frame_multi_pair_threshold,
        token_merging_frame_multi_span_threshold=args.token_merging_frame_multi_span_threshold,
        token_merging_frame_upper_adaptive=args.token_merging_frame_upper_adaptive,
        token_merging_frame_staged_ranges=args.token_merging_frame_staged_ranges,
        token_merging_frame_staged_late_segment_threshold=args.token_merging_frame_staged_late_segment_threshold,
        token_merging_frame_staged_late_pair_threshold=args.token_merging_frame_staged_late_pair_threshold,
        token_merging_frame_staged_late_span_threshold=args.token_merging_frame_staged_late_span_threshold,
        token_merging_frame_group_strategy=args.token_merging_frame_group_strategy,
        token_merging_frame_protect_period=args.token_merging_frame_protect_period,
        token_merging_frame_protect_prefix=args.token_merging_frame_protect_prefix,
        token_merging_frame_anchor_count=args.token_merging_frame_anchor_count,
        token_merging_frame_anchor_selection=args.token_merging_frame_anchor_selection,
        token_merging_frame_adaptive_boundary_z=args.token_merging_frame_adaptive_boundary_z,
        token_merging_frame_adaptive_medoid_z=args.token_merging_frame_adaptive_medoid_z,
        token_merging_frame_patch_fusion_quantile=args.token_merging_frame_patch_fusion_quantile,
        token_merging_frame_special_cross_attention=args.token_merging_frame_special_cross_attention,
        token_merging_frame_special_cross_attention_alpha=args.token_merging_frame_special_cross_attention_alpha,
        token_merging_segment_bank_pair_threshold=args.token_merging_segment_bank_pair_threshold,
        token_merging_segment_bank_span_threshold=args.token_merging_segment_bank_span_threshold,
        token_merging_segment_bank_max_group_size=args.token_merging_segment_bank_max_group_size,
        omega_accelerator=args.omega_accelerator,
        sparse_vggt_sparse_ratio=args.sparse_vggt_sparse_ratio,
        sparse_vggt_cdf_threshold=args.sparse_vggt_cdf_threshold,
        sparse_vggt_pool_mode=args.sparse_vggt_pool_mode,
        da_vggt_max_frames=args.da_vggt_max_frames,
        da_vggt_sampling_method=args.da_vggt_sampling_method,
        da_vggt_n_anchors=args.da_vggt_n_anchors,
        da_vggt_dino_batch_size=args.da_vggt_dino_batch_size,
        da_vggt_lambda_div=args.da_vggt_lambda_div,
        da_chunk_strided_groups=args.da_chunk_strided_groups,
        da_chunk_strided_anchor_count=args.da_chunk_strided_anchor_count,
        shared_anchor_num_chunks=args.shared_anchor_num_chunks,
        shared_anchor_count=args.shared_anchor_count,
        dynamic_fastvggt_schedule=args.dynamic_fastvggt_schedule,
        skip_global_attention_blocks=args.skip_global_attention_blocks,
        skip_inter_frame_attention_blocks=args.skip_inter_frame_attention_blocks,
        frame_only_inter_frame_blocks=args.frame_only_inter_frame_blocks,
        register_only_blocks=args.register_only_blocks,
        enable_adaptive_frame_token_fusion=args.enable_adaptive_frame_token_fusion,
        adaptive_frame_representation=args.adaptive_frame_representation,
        adaptive_representation_pca_dim=args.adaptive_representation_pca_dim,
        adaptive_representation_clusters=args.adaptive_representation_clusters,
        adaptive_spatial_grid=args.adaptive_spatial_grid,
        adaptive_grouping=args.adaptive_grouping,
        adaptive_reference_selection=args.adaptive_reference_selection,
        adaptive_reference_participates=not args.adaptive_reference_excluded,
        adaptive_group_similarity_threshold=args.adaptive_group_similarity_threshold,
        adaptive_group_max_size=args.adaptive_group_max_size,
        adaptive_parallel_window=args.adaptive_parallel_window,
        adaptive_update_policy=args.adaptive_update_policy,
        adaptive_update_after_blocks=args.adaptive_update_after_blocks,
        adaptive_frame_fusion=args.adaptive_frame_fusion,
        adaptive_frame_fusion_weighting=args.adaptive_frame_fusion_weighting,
        adaptive_frame_token_similarity_threshold=args.adaptive_frame_token_similarity_threshold,
        adaptive_token_merging=args.adaptive_token_merging,
        adaptive_token_keep_ratio=args.adaptive_token_keep_ratio,
        adaptive_token_clusters=args.adaptive_token_clusters,
        adaptive_token_cluster_budget=args.adaptive_token_cluster_budget,
        adaptive_token_kmeans_iterations=args.adaptive_token_kmeans_iterations,
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
            args.bonn_split,
        )
        output_root = Path(args.output_dir) / dataset

        for seq in tqdm(sequences, desc=f"VGGT-Omega {dataset} video depth"):
            output_dir = output_root / seq
            bonn_associations = None
            if dataset == "bonn":
                bonn_associations, images, _, selected_indices, source_frame_count = bonn_associated_frames(
                    dataset_root,
                    seq,
                    rgb_dir=args.bonn_rgb_dir,
                    depth_dir=args.bonn_depth_dir,
                    max_difference=args.bonn_association_max_diff,
                    max_frames=args.max_frames_per_seq,
                )
            else:
                source_images = sequence_images(dataset, dataset_root, seq, args.bonn_rgb_dir)
                images, selected_indices = select_uniform_frames(source_images, args.max_frames_per_seq)
                source_frame_count = len(source_images)
            time_path = output_dir / "_time.json"
            input_manifest_path = output_dir / "_input_frames.json"
            if (
                not args.overwrite
                and time_path.is_file()
                and input_manifest_path.is_file()
                and (dataset != "bonn" or (output_dir / "_bonn_associations.json").is_file())
                and len(list(output_dir.glob("frame_[0-9][0-9][0-9][0-9].npy"))) == len(images)
            ):
                continue
            if device.type == "cuda":
                torch.cuda.empty_cache()
            output_dir.mkdir(parents=True, exist_ok=True)
            with input_manifest_path.open("w") as handle:
                json.dump(input_frames_manifest(source_frame_count, selected_indices, images), handle, indent=2)
            if bonn_associations is not None:
                with (output_dir / "_bonn_associations.json").open("w") as handle:
                    json.dump(
                        {
                            "protocol": "one_to_one_rgb_depth_pose_timestamp_association",
                            "rgb_dir": args.bonn_rgb_dir,
                            "depth_dir": args.bonn_depth_dir,
                            "max_difference_ms": args.bonn_association_max_diff * 1000.0,
                            "associations": bonn_associations,
                        },
                        handle,
                        indent=2,
                    )
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
            if not args.stats_only:
                for frame_idx, depth in enumerate(depths.numpy()):
                    np.save(output_dir / f"frame_{frame_idx:04d}.npy", depth)
                    save_depth_preview(depth, output_dir / f"frame_{frame_idx:04d}.png")
            if poses is not None and not args.stats_only:
                pred_poses = poses.numpy()
                np.save(output_dir / "pred_poses.npy", pred_poses)
                save_tum_trajectory(output_dir / "pred_traj.txt", pred_poses)
                gt_poses = sequence_poses(
                    dataset,
                    dataset_root,
                    seq,
                    len(images),
                    images,
                    pose_indices=[item["pose_index"] for item in bonn_associations]
                    if bonn_associations is not None
                    else None,
                )
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
                        "bonn_split": args.bonn_split,
                        "bonn_association_max_diff_ms": args.bonn_association_max_diff * 1000.0
                        if dataset == "bonn"
                        else None,
                        "seven_scenes_split": args.seven_scenes_split,
                        "pose_eval_frames": args.pose_eval_frames,
                        "pose_eval_seed": args.pose_eval_seed,
                        "frame_selection_protocol": "uniform_full_sequence_v1",
                        "source_frame_count": source_frame_count,
                        "selected_source_indices": selected_indices,
                        "mean_source_stride": input_frames_manifest(
                            source_frame_count, selected_indices, images
                        )["mean_source_stride"],
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
                        "dynamic_fastvggt_schedule": args.dynamic_fastvggt_schedule,
                        "skip_global_attention_blocks": args.skip_global_attention_blocks,
                        "skip_inter_frame_attention_blocks": args.skip_inter_frame_attention_blocks,
                        "frame_only_inter_frame_blocks": args.frame_only_inter_frame_blocks,
                        "register_only_blocks": args.register_only_blocks,
                        "adaptive_frame_token_fusion": {
                            "enabled": args.enable_adaptive_frame_token_fusion,
                            "frame_representation": args.adaptive_frame_representation,
                            "representation_pca_dim": args.adaptive_representation_pca_dim,
                            "representation_clusters": args.adaptive_representation_clusters,
                            "spatial_grid": args.adaptive_spatial_grid,
                            "grouping": args.adaptive_grouping,
                            "reference_selection": args.adaptive_reference_selection,
                            "reference_participates": not args.adaptive_reference_excluded,
                            "group_similarity_threshold": args.adaptive_group_similarity_threshold,
                            "group_max_size": args.adaptive_group_max_size,
                            "parallel_window": args.adaptive_parallel_window,
                            "update_policy": args.adaptive_update_policy,
                            "update_after_blocks": args.adaptive_update_after_blocks,
                            "frame_fusion": args.adaptive_frame_fusion,
                            "frame_fusion_weighting": args.adaptive_frame_fusion_weighting,
                            "frame_special_tokens": "protected",
                            "frame_token_similarity_threshold": args.adaptive_frame_token_similarity_threshold,
                            "token_merging": args.adaptive_token_merging,
                            "token_keep_ratio": args.adaptive_token_keep_ratio,
                            "token_clusters": args.adaptive_token_clusters,
                            "token_cluster_budget": args.adaptive_token_cluster_budget,
                        },
                        "token_merging_flashvid_alpha": args.token_merging_flashvid_alpha,
                        "token_merging_flashvid_expansion": args.token_merging_flashvid_expansion,
                        "token_merging_flashvid_pool_stride": args.token_merging_flashvid_pool_stride,
                        "token_merging_flashvid_tstm_threshold": args.token_merging_flashvid_tstm_threshold,
                        "token_merging_fastvggt_destination_selector": args.token_merging_fastvggt_destination_selector,
                        "token_merging_fastvggt_destination_policy": args.token_merging_fastvggt_destination_policy,
                        "token_merging_fastvggt_uniform_protect_ratio": args.token_merging_fastvggt_uniform_protect_ratio,
                        "token_merging_fastvggt_exclusive_protection": args.token_merging_fastvggt_exclusive_protection,
                        "token_merging_fastvggt_protect_anchor_frames": args.token_merging_fastvggt_protect_anchor_frames,
                        "token_merging_frame_restore_layer": args.token_merging_frame_restore_layer,
                        "token_merging_frame_alpha": args.token_merging_frame_alpha,
                        "token_merging_frame_segment_threshold": args.token_merging_frame_segment_threshold,
                        "token_merging_frame_merge_threshold": args.token_merging_frame_merge_threshold,
                        "token_merging_frame_max_window": args.token_merging_frame_max_window,
                        "token_merging_frame_pool_stride": args.token_merging_frame_pool_stride,
                        "token_merging_frame_multi_max_group_size": args.token_merging_frame_multi_max_group_size,
                        "token_merging_frame_multi_pair_threshold": args.token_merging_frame_multi_pair_threshold,
                        "token_merging_frame_multi_span_threshold": args.token_merging_frame_multi_span_threshold,
                        "token_merging_frame_upper_adaptive": args.token_merging_frame_upper_adaptive,
                        "token_merging_frame_staged_ranges": args.token_merging_frame_staged_ranges,
                        "token_merging_frame_staged_late_segment_threshold": args.token_merging_frame_staged_late_segment_threshold,
                        "token_merging_frame_staged_late_pair_threshold": args.token_merging_frame_staged_late_pair_threshold,
                        "token_merging_frame_staged_late_span_threshold": args.token_merging_frame_staged_late_span_threshold,
                        "token_merging_frame_group_strategy": args.token_merging_frame_group_strategy,
                        "token_merging_frame_protect_period": args.token_merging_frame_protect_period,
                        "token_merging_frame_protect_prefix": args.token_merging_frame_protect_prefix,
                        "token_merging_frame_anchor_count": args.token_merging_frame_anchor_count,
                        "token_merging_frame_anchor_selection": args.token_merging_frame_anchor_selection,
                        "token_merging_frame_adaptive_boundary_z": args.token_merging_frame_adaptive_boundary_z,
                        "token_merging_frame_adaptive_medoid_z": args.token_merging_frame_adaptive_medoid_z,
                        "token_merging_frame_patch_fusion_quantile": args.token_merging_frame_patch_fusion_quantile,
                        "token_merging_frame_special_cross_attention": args.token_merging_frame_special_cross_attention,
                        "token_merging_frame_special_cross_attention_alpha": args.token_merging_frame_special_cross_attention_alpha,
                        "token_merging_segment_bank_pair_threshold": args.token_merging_segment_bank_pair_threshold,
                        "token_merging_segment_bank_span_threshold": args.token_merging_segment_bank_span_threshold,
                        "token_merging_segment_bank_max_group_size": args.token_merging_segment_bank_max_group_size,
                        "omega_accelerator": args.omega_accelerator,
                        "sparse_vggt_sparse_ratio": args.sparse_vggt_sparse_ratio
                        if args.omega_accelerator == "sparse_vggt"
                        else None,
                        "sparse_vggt_cdf_threshold": args.sparse_vggt_cdf_threshold
                        if args.omega_accelerator == "sparse_vggt"
                        else None,
                        "sparse_vggt_pool_mode": args.sparse_vggt_pool_mode
                        if args.omega_accelerator == "sparse_vggt"
                        else None,
                        "da_vggt_max_frames": (
                            args.da_vggt_max_frames
                            if args.omega_accelerator in {"da_vggt", "da_chunk_strided_shared_anchor"}
                            else None
                        ),
                        "da_vggt_sampling_method": (
                            args.da_vggt_sampling_method
                            if args.omega_accelerator in {"da_vggt", "da_chunk_strided_shared_anchor"}
                            else None
                        ),
                        "da_vggt_n_anchors": args.da_vggt_n_anchors if args.omega_accelerator == "da_vggt" else None,
                        "da_chunk_strided_groups": (
                            args.da_chunk_strided_groups
                            if args.omega_accelerator == "da_chunk_strided_shared_anchor"
                            else None
                        ),
                        "da_chunk_strided_anchor_count": (
                            args.da_chunk_strided_anchor_count
                            if args.omega_accelerator == "da_chunk_strided_shared_anchor"
                            else None
                        ),
                        "da_vggt_dino_batch_size": (
                            args.da_vggt_dino_batch_size
                            if args.omega_accelerator in {"da_vggt", "da_chunk_strided_shared_anchor"}
                            else None
                        ),
                        "da_vggt_lambda_div": (
                            args.da_vggt_lambda_div
                            if args.omega_accelerator in {"da_vggt", "da_chunk_strided_shared_anchor"}
                            else None
                        ),
                        "shared_anchor_num_chunks": (
                            args.shared_anchor_num_chunks
                            if args.omega_accelerator == "shared_anchor_chunks"
                            else None
                        ),
                        "shared_anchor_count": (
                            args.shared_anchor_count if args.omega_accelerator == "shared_anchor_chunks" else None
                        ),
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
