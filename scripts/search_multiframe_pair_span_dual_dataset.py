#!/usr/bin/env python3
"""Search pair/span thresholds against raw VGGT-Omega baselines on two datasets."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/data/mmc_syang/miniconda3/envs/fastvggt/bin/python")
DATASETS = ("tum300", "7scenes_test300")


@dataclass(frozen=True)
class Candidate:
    pair: float
    span: float

    @property
    def name(self) -> str:
        return f"p{self.pair:.3f}_s{self.span:.3f}".replace(".", "")


def candidate_grid() -> list[Candidate]:
    # Start at the requested 0.980/0.950, then move outward in a deterministic
    # order so the live ranking becomes useful before the full grid completes.
    pairs = (0.980, 0.982, 0.984, 0.986, 0.988, 0.990, 0.992)
    spans = (0.942, 0.944, 0.946, 0.948, 0.950, 0.952, 0.954)
    start = Candidate(0.980, 0.950)
    candidates = [Candidate(pair, span) for pair in pairs for span in spans]
    return sorted(
        candidates,
        key=lambda cfg: (
            abs(cfg.pair - start.pair) + abs(cfg.span - start.span),
            abs(cfg.pair - start.pair),
            abs(cfg.span - start.span),
        ),
    )


def summary_path(case_root: Path) -> Path:
    return case_root / "summaries" / "_summary_complete_scale_shift.json"


def read_summary(case_root: Path) -> dict:
    with summary_path(case_root).open() as handle:
        return json.load(handle)


def run_case(root: Path, gpu: str, case_root: Path, dataset_key: str, candidate: Candidate | None) -> None:
    if summary_path(case_root).is_file():
        return
    temp_output = case_root / "_temporary"
    summary_dir = case_root / "summaries"
    log_path = case_root / "run.log"
    shutil.rmtree(temp_output, ignore_errors=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    dataset = "tum_dynamic" if dataset_key == "tum300" else "7scenes"
    command = [
        str(PYTHON), str(ROOT / "inference/infer.py"),
        "--dataset", dataset,
        "--output-dir", str(temp_output),
        "--max-frames-per-seq", "300",
        "--window-size", "0",
        "--checkpoint", str(ROOT / "checkpoints/vggt_omega_1b_512.pt"),
        "--overwrite", "--eval",
    ]
    if dataset_key == "7scenes_test300":
        command.extend(["--seven-scenes-split", "test"])
    if candidate is not None:
        command.extend([
            "--skip-inter-frame-attention-blocks", "2",
            "--enable-token-merging",
            "--token-merging-method", "frame_persistent_spatial",
            "--token-merging-ratio", "0.9",
            "--token-merging-start", "0",
            "--token-merging-frame-pool-stride", "2",
            "--token-merging-frame-segment-threshold", "0.9",
            "--token-merging-frame-merge-threshold", "0.1",
            "--token-merging-frame-alpha", "0.1",
            "--token-merging-frame-max-window", "20",
            "--token-merging-frame-restore-layer", "24",
            "--token-merging-frame-multi-max-group-size", "4",
            "--token-merging-frame-multi-pair-threshold", f"{candidate.pair:.3f}",
            "--token-merging-frame-multi-span-threshold", f"{candidate.span:.3f}",
        ])

    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": gpu,
        "PYTHONPATH": str(ROOT),
        "PYTHONNOUSERSITE": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "HF_HOME": str(ROOT / ".cache/huggingface"),
        "TRANSFORMERS_CACHE": str(ROOT / ".cache/huggingface/hub"),
    })
    with log_path.open("w") as handle:
        subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)

    source_root = temp_output / dataset
    for filename in (
        "_summary_complete_scale_shift.json",
        "_summary_scale_shift.json",
        "_sequence_metrics_scale_shift.csv",
        "_summary_pose_auc.json",
    ):
        source = source_root / filename
        if source.is_file():
            shutil.copy2(source, summary_dir / filename)
    shutil.rmtree(temp_output, ignore_errors=True)


def metric_score(baseline: dict, candidate: dict) -> dict[str, float]:
    base_depth, cand_depth = baseline["video_depth"], candidate["video_depth"]
    base_pose, cand_pose = baseline["pose_auc"], candidate["pose_auc"]
    return {
        "Abs Rel score": 100.0 * (base_depth["Abs Rel"] - cand_depth["Abs Rel"]) / base_depth["Abs Rel"],
        "delta < 1.25 score": 100.0 * (cand_depth["delta < 1.25"] - base_depth["delta < 1.25"]) / base_depth["delta < 1.25"],
        "AUC@3 score": 100.0 * (cand_pose["AUC@3"] - base_pose["AUC@3"]) / base_pose["AUC@3"],
        "AUC@30 score": 100.0 * (cand_pose["AUC@30"] - base_pose["AUC@30"]) / base_pose["AUC@30"],
    }


def write_results(root: Path, candidates: list[Candidate]) -> None:
    baseline = {dataset: read_summary(root / "baseline" / dataset) for dataset in DATASETS}
    rows: list[dict[str, object]] = []
    for cfg in candidates:
        summaries = {dataset: root / cfg.name / dataset for dataset in DATASETS}
        if not all(summary_path(path).is_file() for path in summaries.values()):
            continue
        dataset_scores = {}
        dataset_fps = []
        row: dict[str, object] = {"pair_threshold": cfg.pair, "span_threshold": cfg.span}
        for dataset in DATASETS:
            result = read_summary(summaries[dataset])
            scores = metric_score(baseline[dataset], result)
            score = sum(scores.values())
            dataset_scores[dataset] = score
            dataset_fps.append(float(result["speed"]["fps"]))
            prefix = "tum" if dataset == "tum300" else "7scenes"
            row[f"{prefix}_Abs Rel"] = result["video_depth"]["Abs Rel"]
            row[f"{prefix}_delta < 1.25"] = result["video_depth"]["delta < 1.25"]
            row[f"{prefix}_AUC@3"] = result["pose_auc"]["AUC@3"]
            row[f"{prefix}_AUC@30"] = result["pose_auc"]["AUC@30"]
            row.update({f"{prefix}_{key}": value for key, value in scores.items()})
            row[f"{prefix}_score"] = score
            row[f"{prefix}_fps"] = dataset_fps[-1]
        row["total_score"] = sum(dataset_scores.values()) / len(DATASETS)
        row["mean_fps"] = sum(dataset_fps) / len(dataset_fps)
        rows.append(row)

    fields = [
        "pair_threshold", "span_threshold",
        "tum_Abs Rel", "tum_delta < 1.25", "tum_AUC@3", "tum_AUC@30",
        "tum_Abs Rel score", "tum_delta < 1.25 score", "tum_AUC@3 score", "tum_AUC@30 score", "tum_score", "tum_fps",
        "7scenes_Abs Rel", "7scenes_delta < 1.25", "7scenes_AUC@3", "7scenes_AUC@30",
        "7scenes_Abs Rel score", "7scenes_delta < 1.25 score", "7scenes_AUC@3 score", "7scenes_AUC@30 score", "7scenes_score", "7scenes_fps",
        "total_score", "mean_fps",
    ]
    ordered = sorted(rows, key=lambda row: (float(row["total_score"]), float(row["mean_fps"])), reverse=True)
    for filename, content in (("results_all.csv", rows), ("results_ranked.csv", ordered)):
        with (root / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="6")
    parser.add_argument("--output-root", type=Path, default=ROOT / "new_results/2/multiframe_pair_span_dual_search_fastall_r090_skip2")
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    for dataset in DATASETS:
        run_case(root, args.gpu, root / "baseline" / dataset, dataset, None)
    candidates = candidate_grid()
    for cfg in candidates:
        for dataset in DATASETS:
            run_case(root, args.gpu, root / cfg.name / dataset, dataset, cfg)
        write_results(root, candidates)


if __name__ == "__main__":
    main()
