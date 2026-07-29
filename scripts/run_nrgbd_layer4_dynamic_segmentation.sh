#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/data/mmc_syang/miniconda3/envs/fastvggt/bin/python}"
NRGBD_ROOT="${NRGBD_ROOT:-$ROOT/../dataset/NRGBD}"
RUN_ROOT="${RUN_ROOT:-$ROOT/new_results/1/layer4_dynamic_segmentation_crf_nrgbd300_pca128_k03_equalframe_attention}"
GPUS=(4 5 6 7)

mkdir -p "$RUN_ROOT/logs"
mapfile -t SEQUENCES < <(find -L "$NRGBD_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)

run_worker() {
  local gpu="$1" worker="$2" index=0 sequence output log
  log="$RUN_ROOT/logs/gpu${gpu}.log"
  for sequence in "${SEQUENCES[@]}"; do
    if (( index % ${#GPUS[@]} != worker )); then
      ((index += 1))
      continue
    fi
    output="$RUN_ROOT/nrgbd/$sequence"
    printf '[%s] gpu=%s sequence=nrgbd/%s\n' "$(date '+%F %T')" "$gpu" "$sequence" | tee -a "$log"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT" PYTHONNOUSERSITE=1 \
    HF_HOME="$ROOT/.cache/huggingface" TRANSFORMERS_CACHE="$ROOT/.cache/huggingface/transformers" \
    MPLCONFIGDIR="$ROOT/.cache/matplotlib" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PYTHON_BIN" "$ROOT/scripts/visualize_layer4_token_clusters.py" \
      --dataset nrgbd --dataset-root "$NRGBD_ROOT" --sequence "$sequence" --output-dir "$output" \
      --max-source-frames 300 --num-frames 10 --pca-dim 128 --clusters 3 --seed 0 \
      --quality-sample-size 3000 --label-smoothing crf --camera-attention \
      --camera-attention-global-dynamic --overwrite >> "$log" 2>&1
    ((index += 1))
  done
}

for worker in "${!GPUS[@]}"; do
  run_worker "${GPUS[$worker]}" "$worker" &
done
wait
