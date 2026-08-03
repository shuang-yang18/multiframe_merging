#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GPU="${1:?usage: run_uniform400_seven_methods.sh <gpu: 0-6>}"
RUN_ROOT="${RUN_ROOT:-$ROOT/auc_eval_results/01/400}"
FAST_PYTHON="${FAST_PYTHON:-/data/mmc_syang/miniconda3/envs/fastvggt/bin/python}"
SPARSE_REPO="${SPARSE_REPO:-/data/mmc_syang/sparse-vggt}"
SPARSE_PYTHON="${SPARSE_PYTHON:-$SPARSE_REPO/.venv/bin/python}"
CHECKPOINT="${CHECKPOINT:-$ROOT/checkpoints/vggt_omega_1b_512.pt}"

TUM_ROOT="${TUM_ROOT:-/data/mmc_syang/dataset/TUM-Dynamics}"
SEVEN_SCENES_ROOT="${SEVEN_SCENES_ROOT:-/data/mmc_syang/dataset/7scenes}"
NRGBD_ROOT="${NRGBD_ROOT:-/data/mmc_syang/dataset/NRGBD}"

# Original FastVGGT behavior: 2x2 pseudorandom destinations, native 10% uniform
# protection, and no additional protection for shared-anchor frames.
FASTVGGT_STANDARD_ARGS=(
  --enable-token-merging
  --token-merging-method spatial
  --token-merging-ratio 0.9
  --token-merging-fastvggt-destination-policy grid_2x2
  --token-merging-fastvggt-destination-selector random
  --token-merging-fastvggt-uniform-protect-ratio 0.1
  --no-token-merging-fastvggt-exclusive-protection
  --no-token-merging-fastvggt-protect-anchor-frames
)

case "$GPU" in
  0)
    METHOD="baseline"
    PYTHON_BIN="$FAST_PYTHON"
    METHOD_ARGS=(--omega-accelerator none)
    ;;
  1)
    METHOD="fastvggt_r090_standard"
    PYTHON_BIN="$FAST_PYTHON"
    METHOD_ARGS=(--omega-accelerator none "${FASTVGGT_STANDARD_ARGS[@]}")
    ;;
  2)
    METHOD="sparse_vggt"
    PYTHON_BIN="$SPARSE_PYTHON"
    METHOD_ARGS=(
      --omega-accelerator sparse_vggt
      --sparse-vggt-sparse-ratio 0.5
      --sparse-vggt-pool-mode avg
    )
    ;;
  3)
    METHOD="da_vggt"
    PYTHON_BIN="$FAST_PYTHON"
    METHOD_ARGS=(
      --omega-accelerator da_vggt
      --da-vggt-max-frames 64
      --da-vggt-sampling-method fl_maxmin
      --da-vggt-n-anchors 1
      --da-vggt-lambda-div 0.0
    )
    ;;
  4)
    METHOD="da_vggt_fastvggt_r090_standard"
    PYTHON_BIN="$FAST_PYTHON"
    METHOD_ARGS=(
      --omega-accelerator da_vggt
      --da-vggt-max-frames 64
      --da-vggt-sampling-method fl_maxmin
      --da-vggt-n-anchors 1
      --da-vggt-lambda-div 0.0
      "${FASTVGGT_STANDARD_ARGS[@]}"
    )
    ;;
  5)
    METHOD="ours_k05_a05_fastvggt_r090_blocks00_09_standard"
    PYTHON_BIN="$FAST_PYTHON"
    METHOD_ARGS=(
      --omega-accelerator shared_anchor_chunks
      --shared-anchor-num-chunks 5
      --shared-anchor-count 5
      "${FASTVGGT_STANDARD_ARGS[@]}"
      --token-merging-layer-ratios '1-10:0.9,11-24:0.0'
    )
    ;;
  6)
    METHOD="ours_k05_a05_fastvggt_r090_blocks00_09_grid2x2_uniform00_anchor"
    PYTHON_BIN="$FAST_PYTHON"
    METHOD_ARGS=(
      --omega-accelerator shared_anchor_chunks
      --shared-anchor-num-chunks 5
      --shared-anchor-count 5
      --enable-token-merging
      --token-merging-method spatial
      --token-merging-ratio 0.9
      --token-merging-layer-ratios '1-10:0.9,11-24:0.0'
      --token-merging-fastvggt-destination-policy grid_2x2
      --token-merging-fastvggt-destination-selector random
      --token-merging-fastvggt-uniform-protect-ratio 0.0
      --token-merging-fastvggt-exclusive-protection
      --token-merging-fastvggt-protect-anchor-frames
    )
    ;;
  *)
    echo "Expected GPU in [0, 6], got $GPU" >&2
    exit 2
    ;;
esac

cd "$ROOT"
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/$METHOD"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"

run_case() {
  local label="$1"
  local dataset="$2"
  local dataset_root="$3"
  shift 3
  local log="$RUN_ROOT/logs/${METHOD}_${label}_gpu${GPU}.log"

  echo "[$(date '+%F %T')] start method=${METHOD} dataset=${label} gpu=${GPU}" | tee -a "$log"
  CUDA_VISIBLE_DEVICES="$GPU" \
  PYTHONPATH="$ROOT:$SPARSE_REPO/src:$SPARSE_REPO/external/vggt:$SPARSE_REPO/external/SpargeAttn:$SPARSE_REPO" \
  PYTHONNOUSERSITE=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PYTHON_BIN" "$ROOT/inference/infer.py" \
    --dataset "$dataset" \
    --dataset-root "$dataset_root" \
    --output-dir "$RUN_ROOT/$METHOD" \
    --max-frames-per-seq 400 \
    --window-size 0 \
    --checkpoint "$CHECKPOINT" \
    --overwrite \
    --eval \
    --eval-align scale_shift \
    --pose-eval-frames 0 \
    --pose-eval-seed 0 \
    "${METHOD_ARGS[@]}" \
    "$@" \
    >> "$log" 2>&1
  echo "[$(date '+%F %T')] complete method=${METHOD} dataset=${label} gpu=${GPU}" | tee -a "$log"
}

run_case tum400 tum_dynamic "$TUM_ROOT"
run_case 7scenes_test400 7scenes "$SEVEN_SCENES_ROOT" --seven-scenes-split test
run_case nrgbd400 nrgbd "$NRGBD_ROOT"
