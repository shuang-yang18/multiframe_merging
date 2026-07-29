#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/data/mmc_syang/miniconda3/envs/fastvggt/bin/python}"
TUM_ROOT="${TUM_ROOT:-$ROOT/../dataset/TUM-Dynamics}"
SEVEN_SCENES_ROOT="${SEVEN_SCENES_ROOT:-$ROOT/../dataset/7scenes/test}"
RUN_ROOT="${RUN_ROOT:-$ROOT/new_results/1/layer4_dynamic_segmentation_seqfirst_pca128_256_k236}"
PCA_DIMS="${PCA_DIMS:-128,192,256}"
CLUSTER_COUNTS="${CLUSTER_COUNTS:-2,3,6}"
GPUS=(4 5 6 7)

mkdir -p "$RUN_ROOT/logs"
TASKS_FILE="$RUN_ROOT/tasks.tsv"

if [[ ! -f "$TASKS_FILE" ]]; then
  : > "$TASKS_FILE"
  while IFS= read -r sequence; do
    printf 'tum_dynamic\t%s\t%s\n' "$TUM_ROOT" "$sequence" >> "$TASKS_FILE"
  done < <(find "$TUM_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'rgbd_dataset_*' -printf '%f\n' | sort)
  while IFS= read -r sequence_path; do
    if compgen -G "$sequence_path/*.color.png" > /dev/null; then
      sequence="${sequence_path#"$SEVEN_SCENES_ROOT"/}"
      printf '7scenes\t%s\t%s\n' "$SEVEN_SCENES_ROOT" "$sequence" >> "$TASKS_FILE"
    fi
  done < <(find -L "$SEVEN_SCENES_ROOT" -mindepth 2 -maxdepth 2 -type d -name 'seq-*' | sort)
fi

run_worker() {
  local gpu="$1"
  local worker_index="$2"
  local pca_dim="$3"
  local task_index=0
  local dataset root sequence slug output_dir log_file
  log_file="$RUN_ROOT/logs/pca${pca_dim}_gpu${gpu}.log"
  while IFS=$'\t' read -r dataset root sequence; do
    if (( task_index % ${#GPUS[@]} != worker_index )); then
      ((task_index += 1))
      continue
    fi
    slug="${sequence//\//__}"
    output_dir="$RUN_ROOT/$dataset/$slug"
    printf '[%s] pca=%s gpu=%s sequence=%s/%s\n' "$(date '+%F %T')" "$pca_dim" "$gpu" "$dataset" "$sequence" | tee -a "$log_file"
    CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONPATH="$ROOT" \
    PYTHONNOUSERSITE=1 \
    HF_HOME="$ROOT/.cache/huggingface" \
    TRANSFORMERS_CACHE="$ROOT/.cache/huggingface/transformers" \
    MPLCONFIGDIR="$ROOT/.cache/matplotlib" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PYTHON_BIN" "$ROOT/scripts/visualize_layer4_token_clusters.py" \
      --dataset "$dataset" \
      --dataset-root "$root" \
      --sequence "$sequence" \
      --output-dir "$output_dir" \
      --max-source-frames 300 \
      --num-frames 10 \
      --pca-dim "$pca_dim" \
      --cluster-counts "$CLUSTER_COUNTS" \
      --seed 0 \
      --quality-sample-size 3000 >> "$log_file" 2>&1
    ((task_index += 1))
  done < "$TASKS_FILE"
}

IFS=',' read -r -a PCA_VALUES <<< "$PCA_DIMS"
for pca_dim in "${PCA_VALUES[@]}"; do
  printf '[%s] starting PCA=%s for all sequences\n' "$(date '+%F %T')" "$pca_dim" | tee -a "$RUN_ROOT/logs/driver.log"
  for worker_index in "${!GPUS[@]}"; do
    run_worker "${GPUS[$worker_index]}" "$worker_index" "$pca_dim" &
  done
  wait
done
