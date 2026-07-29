#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/data/mmc_syang/miniconda3/envs/fastvggt/bin/python}"
TUM_ROOT="${TUM_ROOT:-$ROOT/../dataset/TUM-Dynamics}"
SEVEN_SCENES_ROOT="${SEVEN_SCENES_ROOT:-$ROOT/../dataset/7scenes/test}"
RUN_ROOT="${RUN_ROOT:-$ROOT/new_results/1/layer4_dynamic_segmentation_tum_7scenes_test300_uniform10}"
PCA_DIMS="${PCA_DIMS:-48,64,96,128}"
CLUSTER_COUNTS="${CLUSTER_COUNTS:-2,3,4,5,6,8}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-none}"
POTTS_SPATIAL_WEIGHT="${POTTS_SPATIAL_WEIGHT:-0.12}"
POTTS_TEMPORAL_WEIGHT="${POTTS_TEMPORAL_WEIGHT:-0.06}"
POTTS_ITERATIONS="${POTTS_ITERATIONS:-5}"
CRF_SPATIAL_WEIGHT="${CRF_SPATIAL_WEIGHT:-0.9}"
CRF_TEMPORAL_WEIGHT="${CRF_TEMPORAL_WEIGHT:-0.08}"
CRF_COLOR_SIGMA="${CRF_COLOR_SIGMA:-0.18}"
CRF_UNARY_TEMPERATURE="${CRF_UNARY_TEMPERATURE:-1.0}"
CRF_ITERATIONS="${CRF_ITERATIONS:-10}"
CAMERA_ATTENTION="${CAMERA_ATTENTION:-0}"
CAMERA_ATTENTION_GLOBAL_DYNAMIC="${CAMERA_ATTENTION_GLOBAL_DYNAMIC:-0}"
OVERWRITE="${OVERWRITE:-0}"
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

task_total=$(wc -l < "$TASKS_FILE")
printf 'Run root: %s\nTasks: %s sequences, PCA={%s}, K={%s}, smoothing=%s\n' "$RUN_ROOT" "$task_total" "$PCA_DIMS" "$CLUSTER_COUNTS" "$LABEL_SMOOTHING"

run_worker() {
  local gpu="$1"
  local worker_index="$2"
  local task_index=0
  local dataset root sequence slug output_dir log_file
  local camera_args=()
  local overwrite_args=()
  if [[ "$CAMERA_ATTENTION" == "1" ]]; then
    camera_args+=(--camera-attention)
  fi
  if [[ "$CAMERA_ATTENTION_GLOBAL_DYNAMIC" == "1" ]]; then
    camera_args+=(--camera-attention-global-dynamic)
  fi
  if [[ "$OVERWRITE" == "1" ]]; then
    overwrite_args+=(--overwrite)
  fi
  log_file="$RUN_ROOT/logs/gpu${gpu}.log"
  while IFS=$'\t' read -r dataset root sequence; do
    if (( task_index % ${#GPUS[@]} != worker_index )); then
      ((task_index += 1))
      continue
    fi
    slug="${sequence//\//__}"
    output_dir="$RUN_ROOT/$dataset/$slug"
    printf '[%s] gpu=%s sequence=%s/%s\n' "$(date '+%F %T')" "$gpu" "$dataset" "$sequence" | tee -a "$log_file"
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
      --pca-dims "$PCA_DIMS" \
      --cluster-counts "$CLUSTER_COUNTS" \
      --seed 0 \
      --quality-sample-size 3000 \
      --label-smoothing "$LABEL_SMOOTHING" \
      --potts-spatial-weight "$POTTS_SPATIAL_WEIGHT" \
      --potts-temporal-weight "$POTTS_TEMPORAL_WEIGHT" \
      --potts-iterations "$POTTS_ITERATIONS" \
      --crf-spatial-weight "$CRF_SPATIAL_WEIGHT" \
      --crf-temporal-weight "$CRF_TEMPORAL_WEIGHT" \
      --crf-color-sigma "$CRF_COLOR_SIGMA" \
      --crf-unary-temperature "$CRF_UNARY_TEMPERATURE" \
      --crf-iterations "$CRF_ITERATIONS" \
      "${camera_args[@]}" \
      "${overwrite_args[@]}" >> "$log_file" 2>&1
    ((task_index += 1))
  done < "$TASKS_FILE"
}

for worker_index in "${!GPUS[@]}"; do
  run_worker "${GPUS[$worker_index]}" "$worker_index" &
done
wait
