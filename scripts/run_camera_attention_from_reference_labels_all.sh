#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/data/mmc_syang/miniconda3/envs/fastvggt/bin/python}"
REFERENCE_ROOT="${REFERENCE_ROOT:-$ROOT/new_results/1/layer4_dynamic_segmentation_crf_tum_7scenes_test300_pca128_k02_k03}"
RUN_ROOT="${RUN_ROOT:-$ROOT/new_results/1/camera_attention_reference_kmeans_tum_7scenes_test300_pca128_k03}"
GPUS=(4 5 6 7)
DATASET_FILTER="${DATASET_FILTER:-}"

mkdir -p "$RUN_ROOT/logs"
TASKS="$RUN_ROOT/tasks.tsv"
: > "$TASKS"
while IFS= read -r labels; do
  relative="${labels#"$REFERENCE_ROOT"/}"
  dataset="${relative%%/*}"
  if [[ -n "$DATASET_FILTER" && "$dataset" != "$DATASET_FILTER" ]]; then
    continue
  fi
  rest="${relative#*/}"
  slug="${rest%%/pca128_k03/labels.npy}"
  if [[ "$dataset" == "tum_dynamic" ]]; then
    root="$ROOT/../dataset/TUM-Dynamics"
    sequence="$slug"
  else
    root="$ROOT/../dataset/7scenes/test"
    sequence="${slug//__//}"
  fi
  printf '%s\t%s\t%s\t%s\n' "$dataset" "$root" "$sequence" "$labels" >> "$TASKS"
done < <(find "$REFERENCE_ROOT" -path '*/pca128_k03/labels.npy' | sort)

run_worker() {
  local gpu="$1" worker="$2" index=0
  local dataset root sequence labels slug output log
  log="$RUN_ROOT/logs/gpu${gpu}.log"
  while IFS=$'\t' read -r dataset root sequence labels; do
    if (( index % ${#GPUS[@]} != worker )); then
      ((index += 1))
      continue
    fi
    slug="${sequence//\//__}"
    output="$RUN_ROOT/$dataset/$slug"
    printf '[%s] gpu=%s sequence=%s/%s\n' "$(date '+%F %T')" "$gpu" "$dataset" "$sequence" | tee -a "$log"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT" PYTHONNOUSERSITE=1 \
    HF_HOME="$ROOT/.cache/huggingface" TRANSFORMERS_CACHE="$ROOT/.cache/huggingface/transformers" \
    MPLCONFIGDIR="$ROOT/.cache/matplotlib" \
    "$PYTHON_BIN" "$ROOT/scripts/visualize_camera_attention_from_labels.py" \
      --dataset "$dataset" --dataset-root "$root" --sequence "$sequence" \
      --reference-labels "$labels" --output-dir "$output" >> "$log" 2>&1
    ((index += 1))
  done < "$TASKS"
}

for worker in "${!GPUS[@]}"; do
  run_worker "${GPUS[$worker]}" "$worker" &
done
wait
