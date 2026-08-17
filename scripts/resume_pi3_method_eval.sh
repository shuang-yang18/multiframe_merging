#!/usr/bin/env bash
set -uo pipefail

ROOT=/data/mmc_syang/Pi3
PYTHON=/data/mmc_syang/miniconda3/envs/flow3r/bin/python
GPU="${1:?gpu id required}"
METHOD="${2:?method required: none|fastvggt|frame_persistent_spatial}"
NAME="${3:?run name required}"

cd "$ROOT"
export PYTHONPATH="$ROOT"
export HF_HOME=/data/mmc_syang/.cache/huggingface
export CUDA_VISIBLE_DEVICES="$GPU"

OUT="outputs/${NAME}"
COMMON=(
  pi3.pretrained_model_name_or_path=checkpoints/Pi3
  pi3.token_merging_method="$METHOD"
  pi3.token_merging_ratio=0.9
  pi3.token_merging_frame_alpha=0.1
  pi3.token_merging_frame_segment_threshold=0.9
  pi3.token_merging_frame_merge_threshold=0.1
  pi3.token_merging_frame_max_window=20
  pi3.token_merging_frame_pool_stride=2
  pi3.token_merging_frame_multi_max_group_size=4
  pi3.token_merging_frame_multi_pair_threshold=0.95
  pi3.token_merging_frame_multi_span_threshold=0.93
  max_frames_per_seq=300
)

echo "[$(date '+%F %T')] RESUME name=${NAME} method=${METHOD} gpu=${GPU}"

echo "[$(date '+%F %T')] monodepth eval"
"$PYTHON" monodepth/eval.py \
  evaluation=monodepth \
  eval_datasets='[tum_dynamic,7scenes]' \
  output_dir="$OUT/monodepth" \
  "${COMMON[@]}" || exit 1

echo "[$(date '+%F %T')] relpose-distance eval"
"$PYTHON" relpose/eval_dist.py \
  evaluation=relpose-distance \
  eval_datasets='[tum_dynamic,7scenes]' \
  output_dir="$OUT/relpose" \
  "${COMMON[@]}" || exit 1

echo "[$(date '+%F %T')] DONE name=${NAME} method=${METHOD} gpu=${GPU}"
