"""Timestamp-based RGB, depth, and pose associations for the Bonn RGB-D data."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _timestamp(path: str | Path) -> float:
    return float(Path(path).stem)


def _read_pose_timestamps(path: Path) -> np.ndarray:
    values = np.loadtxt(path, comments="#")
    values = np.atleast_2d(values)
    return values[:, 0].astype(np.float64, copy=False)


def _read_image_stream(sequence_root: Path, directory: str) -> tuple[np.ndarray, list[Path]]:
    """Read Bonn's canonical timestamp manifest when using the full stream."""
    manifest = sequence_root / f"{directory}.txt"
    if manifest.is_file():
        entries = []
        for line in manifest.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                timestamp, relative_path = line.split()[:2]
                entries.append((float(timestamp), sequence_root / relative_path))
        if entries:
            return np.asarray([item[0] for item in entries], dtype=np.float64), [item[1] for item in entries]
    paths = sorted((sequence_root / directory).glob("*.png"))
    return np.asarray([_timestamp(path) for path in paths], dtype=np.float64), paths


def _one_to_one_matches(
    reference_times: np.ndarray,
    target_times: np.ndarray,
    max_difference: float,
) -> dict[int, tuple[int, float]]:
    """Associate monotonic timestamp streams without reusing target samples."""
    candidates: list[tuple[float, int, int]] = []
    left = 0
    for ref_idx, ref_time in enumerate(reference_times):
        while left < len(target_times) and target_times[left] < ref_time - max_difference:
            left += 1
        target_idx = left
        while target_idx < len(target_times) and target_times[target_idx] <= ref_time + max_difference:
            candidates.append((abs(float(target_times[target_idx] - ref_time)), ref_idx, target_idx))
            target_idx += 1

    matches: dict[int, tuple[int, float]] = {}
    used_targets: set[int] = set()
    for difference, ref_idx, target_idx in sorted(candidates):
        if ref_idx in matches or target_idx in used_targets:
            continue
        matches[ref_idx] = (target_idx, difference)
        used_targets.add(target_idx)
    return matches


def build_bonn_associations(
    dataset_root: str | Path,
    sequence: str,
    *,
    rgb_dir: str = "rgb",
    depth_dir: str = "depth",
    max_difference: float = 0.02,
) -> list[dict]:
    """Return timestamp-aligned Bonn RGB/depth/pose triples.

    Bonn stores the three sensors on different clocks. Exact filename equality is
    therefore not available; the returned triples are one-to-one matches within
    ``max_difference`` seconds and are sorted by the RGB timestamp.
    """
    sequence_root = Path(dataset_root) / sequence
    rgb_times, rgb_paths = _read_image_stream(sequence_root, rgb_dir)
    depth_times, depth_paths = _read_image_stream(sequence_root, depth_dir)
    pose_path = sequence_root / "groundtruth.txt"
    if not rgb_paths or not depth_paths or not pose_path.is_file():
        raise FileNotFoundError(f"Bonn sequence is missing rgb, depth, or groundtruth: {sequence_root}")

    pose_times = _read_pose_timestamps(pose_path)
    depth_matches = _one_to_one_matches(rgb_times, depth_times, max_difference)
    pose_matches = _one_to_one_matches(rgb_times, pose_times, max_difference)

    associations = []
    for rgb_idx in sorted(set(depth_matches) & set(pose_matches)):
        depth_idx, depth_delta = depth_matches[rgb_idx]
        pose_idx, pose_delta = pose_matches[rgb_idx]
        associations.append(
            {
                "rgb_index": int(rgb_idx),
                "rgb": rgb_paths[rgb_idx].name,
                "rgb_timestamp": float(rgb_times[rgb_idx]),
                "depth_index": int(depth_idx),
                "depth": depth_paths[depth_idx].name,
                "depth_timestamp": float(depth_times[depth_idx]),
                "depth_delta_ms": float(depth_delta * 1000.0),
                "pose_index": int(pose_idx),
                "pose_timestamp": float(pose_times[pose_idx]),
                "pose_delta_ms": float(pose_delta * 1000.0),
            }
        )
    if not associations:
        raise ValueError(f"No Bonn timestamp triples for {sequence} within {max_difference * 1000:.1f} ms")
    return associations


def association_paths(
    dataset_root: str | Path,
    sequence: str,
    associations: list[dict],
    *,
    rgb_dir: str,
    depth_dir: str,
) -> tuple[list[str], list[str]]:
    sequence_root = Path(dataset_root) / sequence
    rgb_paths = [str(sequence_root / rgb_dir / item["rgb"]) for item in associations]
    depth_paths = [str(sequence_root / depth_dir / item["depth"]) for item in associations]
    return rgb_paths, depth_paths
