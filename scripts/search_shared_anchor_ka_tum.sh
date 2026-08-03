#!/usr/bin/env bash
set -euo pipefail

# Search the shared-anchor K/A configuration for one temporal budget.  The
# fastest candidate satisfying all four TUM accuracy constraints is then
# evaluated on the other two datasets with the same frame budget.
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GPU="${1:?usage: search_shared_anchor_ka_tum.sh <gpu> <frames: 300|400|500> <run-root>}"
FRAMES="${2:?usage: search_shared_anchor_ka_tum.sh <gpu> <frames: 300|400|500> <run-root>}"
RUN_ROOT="${3:?usage: search_shared_anchor_ka_tum.sh <gpu> <frames: 300|400|500> <run-root>}"

FAST_PYTHON="${FAST_PYTHON:-/data/mmc_syang/miniconda3/envs/fastvggt/bin/python}"
CHECKPOINT="${CHECKPOINT:-$ROOT/checkpoints/vggt_omega_1b_512.pt}"
TUM_ROOT="${TUM_ROOT:-/data/mmc_syang/dataset/TUM-Dynamics}"
SEVEN_SCENES_ROOT="${SEVEN_SCENES_ROOT:-/data/mmc_syang/dataset/7scenes}"
NRGBD_ROOT="${NRGBD_ROOT:-/data/mmc_syang/dataset/NRGBD}"
MAX_K=$((FRAMES / 30))

case "$FRAMES" in
  300)
    BASELINE_SUMMARY="$ROOT/auc_eval_results/01/baseline/tum_dynamic/_summary_complete_scale_shift.json"
    ;;
  400|500)
    # The 500-frame baseline exceeds device memory; use the requested 400-frame
    # reference for both the four-metric gate and the 500-frame search.
    BASELINE_SUMMARY="$ROOT/auc_eval_results/01/400/baseline/tum_dynamic/_summary_complete_scale_shift.json"
    ;;
  *)
    echo "frames must be 300, 400, or 500; got $FRAMES" >&2
    exit 2
    ;;
esac

if [[ ! -f "$BASELINE_SUMMARY" ]]; then
  echo "Missing baseline summary: $BASELINE_SUMMARY" >&2
  exit 1
fi

json_number() {
  local key="$1"
  local file="$2"
  grep -m1 "\"$key\"" "$file" | sed -nE 's/^[[:space:]]*"[^"]+"[[:space:]]*:[[:space:]]*([-+0-9.eE]+),?[[:space:]]*$/\1/p' | head -1
}

BASE_ABS="$(json_number 'Abs Rel' "$BASELINE_SUMMARY")"
BASE_DELTA="$(json_number 'delta < 1.25' "$BASELINE_SUMMARY")"
BASE_AUC3="$(json_number 'AUC@3' "$BASELINE_SUMMARY")"
BASE_AUC30="$(json_number 'AUC@30' "$BASELINE_SUMMARY")"

SEARCH_ROOT="$RUN_ROOT/tum${FRAMES}"
RESULTS_CSV="$SEARCH_ROOT/results_all.csv"
VALID_CSV="$SEARCH_ROOT/results_valid.csv"
LOG_DIR="$SEARCH_ROOT/logs"
mkdir -p "$SEARCH_ROOT" "$LOG_DIR"

if [[ ! -f "$RESULTS_CSV" ]]; then
  printf 'frames,k,a,abs_rel,delta_125,auc3,auc30,fps,pass,status,output_dir\n' > "$RESULTS_CSV"
fi
if [[ ! -f "$VALID_CSV" ]]; then
  printf 'frames,k,a,abs_rel,delta_125,auc3,auc30,fps,output_dir\n' > "$VALID_CSV"
fi

# This precisely reproduces the original FastVGGT protection policy.
FASTVGGT_STANDARD_ARGS=(
  --enable-token-merging
  --token-merging-method spatial
  --token-merging-ratio 0.9
  --token-merging-layer-ratios '1-10:0.9,11-24:0.0'
  --token-merging-fastvggt-destination-policy grid_2x2
  --token-merging-fastvggt-destination-selector random
  --token-merging-fastvggt-uniform-protect-ratio 0.1
  --no-token-merging-fastvggt-exclusive-protection
  --no-token-merging-fastvggt-protect-anchor-frames
)

clean_reconstruction_artifacts() {
  local output_dir="$1"
  # Keep all evaluation JSON/manifests and remove only generated visual/depth artifacts.
  find "$output_dir" -type f \( -name '*.npy' -o -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' -o -name '*.ply' \) -delete
}

run_infer_eval() {
  local dataset="$1"
  local dataset_root="$2"
  local output_dir="$3"
  local log_file="$4"
  shift 4

  CUDA_VISIBLE_DEVICES="$GPU" \
  PYTHONPATH="$ROOT" \
  PYTHONNOUSERSITE=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$FAST_PYTHON" "$ROOT/inference/infer.py" \
    --dataset "$dataset" \
    --dataset-root "$dataset_root" \
    --output-dir "$output_dir" \
    --max-frames-per-seq "$FRAMES" \
    --window-size 0 \
    --checkpoint "$CHECKPOINT" \
    --overwrite \
    --eval \
    --eval-align scale_shift \
    --pose-eval-frames 0 \
    --pose-eval-seed 0 \
    --omega-accelerator shared_anchor_chunks \
    "${FASTVGGT_STANDARD_ARGS[@]}" \
    "$@" >> "$log_file" 2>&1
}

candidate_passes() {
  local abs_rel="$1"
  local delta="$2"
  local auc3="$3"
  local auc30="$4"
  awk -v abs_rel="$abs_rel" -v delta="$delta" -v auc3="$auc3" -v auc30="$auc30" \
      -v base_abs="$BASE_ABS" -v base_delta="$BASE_DELTA" -v base_auc3="$BASE_AUC3" -v base_auc30="$BASE_AUC30" \
    'BEGIN { exit !(abs_rel <= base_abs * 1.005 && delta >= base_delta * 0.995 && auc3 >= base_auc3 * 0.995 && auc30 >= base_auc30 * 0.995) }'
}

best_fps='-1'
best_k=''
best_a=''
best_dir=''

cd "$ROOT"
for k in $(seq 2 "$MAX_K"); do
  for a in $((k - 1)) "$k" $((k + 1)); do
    output_dir="$SEARCH_ROOT/k$(printf '%02d' "$k")_a$(printf '%02d' "$a")"
    summary="$output_dir/tum_dynamic/_summary_complete_scale_shift.json"
    log_file="$LOG_DIR/k$(printf '%02d' "$k")_a$(printf '%02d' "$a").log"
    echo "[$(date '+%F %T')] start T=$FRAMES K=$k A=$a" | tee -a "$log_file"

    if ! run_infer_eval tum_dynamic "$TUM_ROOT" "$output_dir" "$log_file" \
      --shared-anchor-num-chunks "$k" \
      --shared-anchor-count "$a"; then
      clean_reconstruction_artifacts "$output_dir"
      printf '%s,%s,%s,,,,,false,failed,%s\n' "$FRAMES" "$k" "$a" "$output_dir" >> "$RESULTS_CSV"
      echo "[$(date '+%F %T')] failed T=$FRAMES K=$k A=$a; continuing" | tee -a "$log_file"
      continue
    fi
    clean_reconstruction_artifacts "$output_dir"

    if [[ ! -f "$summary" ]]; then
      printf '%s,%s,%s,,,,,false,missing_summary,%s\n' "$FRAMES" "$k" "$a" "$output_dir" >> "$RESULTS_CSV"
      echo "[$(date '+%F %T')] missing summary T=$FRAMES K=$k A=$a; continuing" | tee -a "$log_file"
      continue
    fi

    abs_rel="$(json_number 'Abs Rel' "$summary")"
    delta="$(json_number 'delta < 1.25' "$summary")"
    auc3="$(json_number 'AUC@3' "$summary")"
    auc30="$(json_number 'AUC@30' "$summary")"
    fps="$(json_number 'fps' "$summary")"
    pass='false'
    if candidate_passes "$abs_rel" "$delta" "$auc3" "$auc30"; then
      pass='true'
      printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' "$FRAMES" "$k" "$a" "$abs_rel" "$delta" "$auc3" "$auc30" "$fps" "$output_dir" >> "$VALID_CSV"
      if awk -v current="$fps" -v best="$best_fps" 'BEGIN { exit !(current > best) }'; then
        best_fps="$fps"
        best_k="$k"
        best_a="$a"
        best_dir="$output_dir"
      fi
    fi
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,complete,%s\n' "$FRAMES" "$k" "$a" "$abs_rel" "$delta" "$auc3" "$auc30" "$fps" "$pass" "$output_dir" >> "$RESULTS_CSV"
    echo "[$(date '+%F %T')] complete T=$FRAMES K=$k A=$a pass=$pass fps=$fps" | tee -a "$log_file"
  done
done

{ head -1 "$VALID_CSV"; tail -n +2 "$VALID_CSV" | sort -t, -k8,8gr; } > "$SEARCH_ROOT/results_valid_by_fps.csv"
BEST_JSON="$SEARCH_ROOT/best_candidate.json"
if [[ -z "$best_k" ]]; then
  printf '{\n  "frames": %s,\n  "status": "no_candidate_passed",\n  "baseline_summary": "%s"\n}\n' "$FRAMES" "$BASELINE_SUMMARY" > "$BEST_JSON"
  echo "[$(date '+%F %T')] no candidate passed T=$FRAMES" | tee -a "$LOG_DIR/search.log"
  exit 0
fi

printf '{\n  "frames": %s,\n  "k": %s,\n  "a": %s,\n  "fps": %s,\n  "baseline_summary": "%s",\n  "output_dir": "%s"\n}\n' \
  "$FRAMES" "$best_k" "$best_a" "$best_fps" "$BASELINE_SUMMARY" "$best_dir" > "$BEST_JSON"

# Each temporal budget contributes its fastest passing candidate to validation.
for dataset in 7scenes nrgbd; do
  if [[ "$dataset" == '7scenes' ]]; then
    dataset_root="$SEVEN_SCENES_ROOT"
    extra_args=(--seven-scenes-split test)
  else
    dataset_root="$NRGBD_ROOT"
    extra_args=()
  fi
  validation_log="$LOG_DIR/best_k$(printf '%02d' "$best_k")_a$(printf '%02d' "$best_a")_${dataset}.log"
  echo "[$(date '+%F %T')] validate T=$FRAMES K=$best_k A=$best_a dataset=$dataset" | tee -a "$validation_log"
  if ! run_infer_eval "$dataset" "$dataset_root" "$best_dir" "$validation_log" \
      --shared-anchor-num-chunks "$best_k" \
      --shared-anchor-count "$best_a" \
      "${extra_args[@]}"; then
    echo "[$(date '+%F %T')] validation failed T=$FRAMES K=$best_k A=$best_a dataset=$dataset" | tee -a "$validation_log"
    continue
  fi
  clean_reconstruction_artifacts "$best_dir"
done

echo "[$(date '+%F %T')] search and validation complete T=$FRAMES K=$best_k A=$best_a fps=$best_fps" | tee -a "$LOG_DIR/search.log"
