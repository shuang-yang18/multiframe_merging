#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/data/mmc_syang/miniconda3/envs/fastvggt/bin/python}"
TUM_ROOT="${TUM_ROOT:-$ROOT/../dataset/TUM-Dynamics}"
SEVEN_SCENES_ROOT="${SEVEN_SCENES_ROOT:-$ROOT/../dataset/7scenes/test}"
RUN_ROOT="${RUN_ROOT:-$ROOT/new_results/1/camera_attention_sequence10_global_tum_7scenes_test300_pca128_k03_crf}"
GPUS=(4 5 6 7)

mkdir -p "$RUN_ROOT/logs"
TASKS="$RUN_ROOT/tasks.tsv"
: > "$TASKS"
while IFS= read -r sequence; do
  printf 'tum_dynamic\t%s\t%s\n' "$TUM_ROOT" "$sequence" >> "$TASKS"
done < <(find "$TUM_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'rgbd_dataset_*' -printf '%f\n' | sort)
while IFS= read -r sequence_path; do
  if compgen -G "$sequence_path/*.color.png" > /dev/null; then
    printf '7scenes\t%s\t%s\n' "$SEVEN_SCENES_ROOT" "${sequence_path#"$SEVEN_SCENES_ROOT"/}" >> "$TASKS"
  fi
done < <(find -L "$SEVEN_SCENES_ROOT" -mindepth 2 -maxdepth 2 -type d -name 'seq-*' | sort)

run_worker() {
  local gpu="$1" worker="$2" index=0
  local dataset root sequence slug output log
  log="$RUN_ROOT/logs/gpu${gpu}.log"
  while IFS=$'\t' read -r dataset root sequence; do
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
    "$PYTHON_BIN" "$ROOT/scripts/visualize_camera_attention_sequence.py" \
      --dataset "$dataset" --dataset-root "$root" --sequence "$sequence" --output-dir "$output" \
      --max-frames 300 --num-frames 10 --chunk-frames 10 --pca-fit-samples 8192 --render-frames 10 >> "$log" 2>&1
    ((index += 1))
  done < "$TASKS"
}

for worker in "${!GPUS[@]}"; do
  run_worker "${GPUS[$worker]}" "$worker" &
done
wait
