#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GPU="${1:?usage: run_legacy_adaptive_ablation_eval.sh <gpu> <tum|7scenes|nrgbd>}"
DATASET_KEY="${2:?usage: run_legacy_adaptive_ablation_eval.sh <gpu> <tum|7scenes|nrgbd>}"
RUN_ROOT="${RUN_ROOT:-$ROOT/auc_eval_results/legacy_adaptive_ablation_20260731}"
ABLATION_MODE="${ABLATION_MODE:-all}"
PYTHON_BIN="${PYTHON_BIN:-/data/mmc_syang/miniconda3/envs/fastvggt/bin/python}"
CHECKPOINT="${CHECKPOINT:-$ROOT/checkpoints/vggt_omega_1b_512.pt}"

DATASET_ARGS=()
case "$DATASET_KEY" in
  tum)
    DATASET_ARGS=(--dataset tum_dynamic --dataset-root "${TUM_ROOT:-$ROOT/datasets/TUM-Dynamics}")
    PREFIX="tum300"
    ;;
  7scenes)
    DATASET_ARGS=(--dataset 7scenes --dataset-root "${SEVEN_SCENES_ROOT:-$ROOT/datasets/7scenes/test}" --seven-scenes-split test)
    PREFIX="7scenes_test300"
    ;;
  nrgbd)
    DATASET_ARGS=(--dataset nrgbd --dataset-root "${NRGBD_ROOT:-/data/mmc_syang/dataset/NRGBD}")
    PREFIX="nrgbd300"
    ;;
  *)
    echo "Expected dataset tum, 7scenes, or nrgbd; got $DATASET_KEY" >&2
    exit 2
    ;;
esac

cd "$ROOT"
mkdir -p "$RUN_ROOT/logs"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"

run_case() {
  local name="$1"
  shift
  echo "[$(date '+%F %T')] start ${DATASET_KEY} ${name} on GPU ${GPU}"
  CUDA_VISIBLE_DEVICES="$GPU" \
  PYTHONPATH="$ROOT" \
  PYTHONNOUSERSITE=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PYTHON_BIN" inference/infer.py \
    "${DATASET_ARGS[@]}" \
    --max-frames-per-seq 300 \
    --window-size 0 \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$RUN_ROOT/${PREFIX}_${name}" \
    --overwrite \
    --eval \
    --pose-eval-frames 0 \
    --pose-eval-seed 0 \
    --omega-accelerator none \
    "$@"
  echo "[$(date '+%F %T')] complete ${DATASET_KEY} ${name} on GPU ${GPU}"
}

# Frame-only: no FastVGGT call is enabled.  Sequences below 27% raw merge
# remain unmerged and therefore follow the baseline aggregator path.
if [[ "$ABLATION_MODE" == "all" || "$ABLATION_MODE" == "frame_only" ]]; then
run_case frame_only_adaptive \
  --enable-token-merging \
  --token-merging-method frame_persistent_adaptive \
  --token-merging-ratio 0.0 \
  --token-merging-start 0 \
  --token-merging-frame-restore-layer 24 \
  --token-merging-frame-alpha 0.1 \
  --token-merging-frame-segment-threshold 0.9 \
  --token-merging-frame-merge-threshold 0.1 \
  --token-merging-frame-max-window 20 \
  --token-merging-frame-pool-stride 2 \
  --token-merging-frame-multi-max-group-size 4 \
  --token-merging-frame-multi-pair-threshold 0.986 \
  --token-merging-frame-multi-span-threshold 0.948 \
  --token-merging-frame-group-strategy local
fi

# Token-only: raw merge below 27% selects all-layer r=0.9; otherwise retain the
# method's layerwise schedule.  No frame state is ever applied.
# CLI layer ranges are 1-based, corresponding to 0-9 and 18-23 in code.
if [[ "$ABLATION_MODE" == "all" || "$ABLATION_MODE" == "token_only" ]]; then
run_case token_only_layerwise_fastvggt_r090_000_090 \
  --enable-token-merging \
  --token-merging-method token_only_adaptive_spatial \
  --token-merging-ratio 0.9 \
  --token-merging-layer-ratios '1-10:0.9,11-18:0.0,19-24:0.9' \
  --token-merging-start 0 \
  --token-merging-frame-alpha 0.1 \
  --token-merging-frame-segment-threshold 0.9 \
  --token-merging-frame-merge-threshold 0.1 \
  --token-merging-frame-max-window 20 \
  --token-merging-frame-pool-stride 2 \
  --token-merging-frame-multi-max-group-size 4 \
  --token-merging-frame-multi-pair-threshold 0.986 \
  --token-merging-frame-multi-span-threshold 0.948 \
  --token-merging-frame-group-strategy local
fi
