#!/usr/bin/env bash
set -uo pipefail

ROOT=/data/mmc_syang/Pi3
PYTHON=/data/mmc_syang/miniconda3/envs/flow3r/bin/python
GPU="${1:?gpu id required}"
NAME="${2:-pi3_frame_fastvggt_pair098_span095_300}"

cd "$ROOT"
export PYTHONPATH="$ROOT"
export HF_HOME=/data/mmc_syang/.cache/huggingface
export CUDA_VISIBLE_DEVICES="$GPU"

OUT="outputs/${NAME}"
COMMON=(
  pi3.pretrained_model_name_or_path=checkpoints/Pi3
  pi3.token_merging_method=frame_persistent_spatial
  pi3.token_merging_ratio=0.9
  pi3.token_merging_frame_alpha=0.1
  pi3.token_merging_frame_segment_threshold=0.9
  pi3.token_merging_frame_merge_threshold=0.1
  pi3.token_merging_frame_max_window=20
  pi3.token_merging_frame_pool_stride=2
  pi3.token_merging_frame_multi_max_group_size=4
  pi3.token_merging_frame_multi_pair_threshold=0.98
  pi3.token_merging_frame_multi_span_threshold=0.95
  max_frames_per_seq=300
)

echo "[$(date '+%F %T')] START name=${NAME} method=frame_persistent_spatial r=0.9 multi=4 pair=0.98 span=0.95 gpu=${GPU}"

echo "[$(date '+%F %T')] monodepth infer"
"$PYTHON" monodepth/infer.py \
  evaluation=monodepth \
  eval_datasets='[tum_dynamic,7scenes]' \
  output_dir="$OUT/monodepth" \
  overwrite=true \
  "${COMMON[@]}" || exit 1

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

echo "[$(date '+%F %T')] DONE name=${NAME}"
