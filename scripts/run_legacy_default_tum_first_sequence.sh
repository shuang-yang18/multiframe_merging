#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/data/mmc_syang/miniconda3/envs/fastvggt/bin/python}"
GPU="${GPU:-1}"
RUN_ROOT="${RUN_ROOT:-$ROOT/new_results/legacy_default_tum_firstseq_revalidate_20260730}"
WORK_DIR="$RUN_ROOT/_work"
SUMMARY_DIR="$RUN_ROOT/summaries"
LOG="$RUN_ROOT/run.log"

mkdir -p "$SUMMARY_DIR"
rm -rf "$WORK_DIR"

CUDA_VISIBLE_DEVICES="$GPU" \
PYTHONPATH="$ROOT" \
PYTHONNOUSERSITE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
HF_HOME="$ROOT/.cache/huggingface" \
TRANSFORMERS_CACHE="$ROOT/.cache/huggingface/hub" \
"$PYTHON_BIN" "$ROOT/inference/infer.py" \
  --dataset tum_dynamic \
  --dataset-root "${TUM_ROOT:-$ROOT/datasets/TUM-Dynamics}" \
  --sequences rgbd_dataset_freiburg3_sitting_halfsphere \
  --max-frames-per-seq 300 \
  --window-size 0 \
  --checkpoint "${CHECKPOINT:-$ROOT/checkpoints/vggt_omega_1b_512.pt}" \
  --output-dir "$WORK_DIR" \
  --overwrite \
  --eval \
  --eval-align scale_shift \
  --enable-token-merging \
  --token-merging-method frame_persistent_spatial \
  --token-merging-ratio 0.9 \
  --token-merging-layer-ratios '1-10:0.9,11-18:0.0,19-24:0.9' \
  --token-merging-start 0 \
  --token-merging-frame-restore-layer 24 \
  --token-merging-frame-group-strategy local \
  --token-merging-frame-pool-stride 2 \
  --token-merging-frame-alpha 0.1 \
  --token-merging-frame-segment-threshold 0.9 \
  --token-merging-frame-merge-threshold 0.1 \
  --token-merging-frame-max-window 20 \
  --token-merging-frame-multi-max-group-size 4 \
  --token-merging-frame-multi-pair-threshold 0.986 \
  --token-merging-frame-multi-span-threshold 0.948 \
  > "$LOG" 2>&1

for artifact in \
  "$WORK_DIR/tum_dynamic/_summary_complete_scale_shift.json" \
  "$WORK_DIR/tum_dynamic/_summary_scale_shift.json" \
  "$WORK_DIR/tum_dynamic/_sequence_metrics_scale_shift.csv" \
  "$WORK_DIR/tum_dynamic/_summary_pose_auc.json"; do
  [[ -f "$artifact" ]] && cp "$artifact" "$SUMMARY_DIR/$(basename "$artifact")"
done

while IFS= read -r -d '' artifact; do
  relative_path="${artifact#"$WORK_DIR/tum_dynamic/"}"
  destination="$SUMMARY_DIR/sequences/$relative_path"
  mkdir -p "$(dirname "$destination")"
  cp "$artifact" "$destination"
done < <(find "$WORK_DIR/tum_dynamic" -type f \( -name '_pose_auc.json' -o -name '_time.json' \) -print0)

rm -rf "$WORK_DIR"
