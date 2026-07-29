#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/data/mmc_syang/miniconda3/envs/fastvggt/bin/python}"
TUM_ROOT="${TUM_ROOT:-$ROOT/../dataset/TUM-Dynamics}"
RUN_ROOT="${RUN_ROOT:-$ROOT/new_results/1/tum_firstseq_uniform10_all_global_layers_pca128_k03}"
GPU="${GPU:-7}"

# VGGT-Omega's register-only inter-frame layers are 2, 6, 9, 14 and 20.
GLOBAL_LAYERS=(0 1 3 4 5 7 8 10 11 12 13 15 16 17 18 19 21 22 23)
SEQUENCE="${SEQUENCE:-$(find "$TUM_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'rgbd_dataset_*' -printf '%f\n' | sort | head -n 1)}"

if [[ -z "$SEQUENCE" ]]; then
  echo "No TUM sequence found under $TUM_ROOT" >&2
  exit 1
fi

mkdir -p "$RUN_ROOT/logs"
LOG="$RUN_ROOT/logs/gpu${GPU}.log"
printf '[%s] dataset=tum_dynamic sequence=%s frames=10 pca=128 k=3 layers=%s\n' \
  "$(date '+%F %T')" "$SEQUENCE" "${GLOBAL_LAYERS[*]}" | tee -a "$LOG"

for layer in "${GLOBAL_LAYERS[@]}"; do
  OUTPUT="$RUN_ROOT/tum_dynamic/$SEQUENCE/global_block_${layer}"
  printf '[%s] starting global block %s\n' "$(date '+%F %T')" "$layer" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$ROOT" PYTHONNOUSERSITE=1 \
  HF_HOME="$ROOT/.cache/huggingface" TRANSFORMERS_CACHE="$ROOT/.cache/huggingface/transformers" \
  MPLCONFIGDIR="$ROOT/.cache/matplotlib" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PYTHON_BIN" "$ROOT/scripts/visualize_layer4_token_clusters.py" \
    --dataset tum_dynamic --dataset-root "$TUM_ROOT" --sequence "$SEQUENCE" --output-dir "$OUTPUT" \
    --feature-block "$layer" --max-source-frames 300 --num-frames 10 --processing-window-size 10 \
    --pca-dim 128 --clusters 3 --seed 0 --quality-sample-size 3000 --label-smoothing crf \
    --camera-attention --camera-attention-block "$layer" --camera-attention-global-dynamic \
    --save-visualizations --overwrite >> "$LOG" 2>&1
done

printf '[%s] complete\n' "$(date '+%F %T')" | tee -a "$LOG"
