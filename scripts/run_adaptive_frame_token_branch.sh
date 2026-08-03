#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?usage: $0 <gpu> <tum|7scenes> <baseline_omega_partial|branch>}"
DATASET="${2:?usage: $0 <gpu> <tum|7scenes> <baseline_omega_partial|branch>}"
METHOD="${3:?usage: $0 <gpu> <tum|7scenes> <baseline_omega_partial|branch>}"

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/data/mmc_syang/miniconda3/envs/fastvggt/bin/python}"
RUN_ROOT="${RUN_ROOT:-$ROOT/auc_eval_results/1new}"
CHECKPOINT="${CHECKPOINT:-$ROOT/checkpoints/vggt_omega_1b_512.pt}"
MATRIX_FILE="${MATRIX_FILE:-$ROOT/scripts/adaptive_frame_token_pairwise_matrix.tsv}"
ADAPTIVE_GROUP_SIMILARITY_THRESHOLD="${ADAPTIVE_GROUP_SIMILARITY_THRESHOLD:-0.998}"
ADAPTIVE_GROUP_MAX_SIZE="${ADAPTIVE_GROUP_MAX_SIZE:-3}"
ADAPTIVE_FRAME_TOKEN_SIMILARITY_THRESHOLD="${ADAPTIVE_FRAME_TOKEN_SIMILARITY_THRESHOLD:-0.995}"
ADAPTIVE_TOKEN_KEEP_RATIO="${ADAPTIVE_TOKEN_KEEP_RATIO:-0.4}"

case "$DATASET" in
  tum)
    DATASET_NAME="tum_dynamic"
    DATASET_ROOT="${TUM_ROOT:-/data/mmc_syang/dataset/TUM-Dynamics}"
    SEQUENCES=(rgbd_dataset_freiburg3_sitting_halfsphere rgbd_dataset_freiburg3_sitting_rpy)
    DATASET_ARGS=()
    ;;
  7scenes)
    DATASET_NAME="7scenes"
    DATASET_ROOT="${SEVEN_SCENES_ROOT:-/data/mmc_syang/dataset/7scenes}"
    SEQUENCES=(chess/seq-03 chess/seq-05)
    DATASET_ARGS=(--seven-scenes-split test)
    ;;
  *)
    echo "Unknown dataset key: $DATASET" >&2
    exit 2
    ;;
esac

COMMON_ADAPTIVE=(
  --enable-adaptive-frame-token-fusion
  # Preserve the released VGGT-Omega topology: these five 0-based blocks
  # remain register-only, and adaptive packing runs only on the other globals.
  --inter-frame-attention partial
  --adaptive-representation-pca-dim 128
  --adaptive-representation-clusters 3
  --adaptive-spatial-grid 4
  --adaptive-group-similarity-threshold "$ADAPTIVE_GROUP_SIMILARITY_THRESHOLD"
  --adaptive-group-max-size "$ADAPTIVE_GROUP_MAX_SIZE"
  --adaptive-parallel-window 10
  --adaptive-update-after-blocks 9,17
  --adaptive-frame-token-similarity-threshold "$ADAPTIVE_FRAME_TOKEN_SIMILARITY_THRESHOLD"
  --adaptive-token-keep-ratio "$ADAPTIVE_TOKEN_KEEP_RATIO"
  --adaptive-token-clusters 4
  --adaptive-token-kmeans-iterations 12
)

MATRIX_ROW="$(awk -F $'\t' -v id="$METHOD" '$1 == id { print; exit }' "$MATRIX_FILE")"
if [[ -n "$MATRIX_ROW" ]]; then
  IFS=$'\t' read -r _ FRAME_REPRESENTATION GROUPING REFERENCE REFERENCE_MODE UPDATE_POLICY FRAME_FUSION TOKEN_MERGING <<< "$MATRIX_ROW"
  METHOD_ARGS=(
    "${COMMON_ADAPTIVE[@]}"
    --adaptive-frame-representation "$FRAME_REPRESENTATION"
    --adaptive-grouping "$GROUPING"
    --adaptive-reference-selection "$REFERENCE"
    --adaptive-update-policy "$UPDATE_POLICY"
  )
  if [[ "$REFERENCE_MODE" == "excluded" ]]; then
    METHOD_ARGS+=(--adaptive-reference-excluded)
  fi
  case "$FRAME_FUSION" in
    direct_uniform)
      METHOD_ARGS+=(--adaptive-frame-fusion direct --adaptive-frame-fusion-weighting uniform)
      ;;
    direct_similarity)
      METHOD_ARGS+=(--adaptive-frame-fusion direct --adaptive-frame-fusion-weighting similarity)
      ;;
    token_wise)
      METHOD_ARGS+=(--adaptive-frame-fusion token_wise)
      ;;
    *) echo "Invalid frame fusion in matrix: $FRAME_FUSION" >&2; exit 2 ;;
  esac
  case "$TOKEN_MERGING" in
    fast_bipartite)
      METHOD_ARGS+=(--adaptive-token-merging fast_bipartite)
      ;;
    category_proportional)
      METHOD_ARGS+=(--adaptive-token-merging category_topk_norm --adaptive-token-cluster-budget proportional)
      ;;
    category_dispersion)
      METHOD_ARGS+=(--adaptive-token-merging category_topk_norm --adaptive-token-cluster-budget dispersion)
      ;;
    *) echo "Invalid token merging in matrix: $TOKEN_MERGING" >&2; exit 2 ;;
  esac
else
case "$METHOD" in
  baseline_omega_partial)
    METHOD_ARGS=(--inter-frame-attention partial)
    ;;
  default)
    METHOD_ARGS=(
      "${COMMON_ADAPTIVE[@]}"
      --adaptive-frame-representation global_pool
      --adaptive-grouping serial
      --adaptive-reference-selection first
      --adaptive-update-policy stage_update
      --adaptive-frame-fusion direct
      --adaptive-frame-fusion-weighting similarity
      --adaptive-token-merging fast_bipartite
    )
    ;;
  cluster_parallel_medoid)
    METHOD_ARGS=(
      "${COMMON_ADAPTIVE[@]}"
      --adaptive-frame-representation cluster_center
      --adaptive-grouping parallel
      --adaptive-reference-selection medoid
      --adaptive-update-policy stage_update
      --adaptive-frame-fusion direct
      --adaptive-frame-fusion-weighting similarity
      --adaptive-token-merging fast_bipartite
    )
    ;;
  spatial_serial_diverse)
    METHOD_ARGS=(
      "${COMMON_ADAPTIVE[@]}"
      --adaptive-frame-representation spatial_grid
      --adaptive-grouping serial
      --adaptive-reference-selection diverse
      --adaptive-update-policy stage_update
      --adaptive-frame-fusion direct
      --adaptive-frame-fusion-weighting similarity
      --adaptive-token-merging fast_bipartite
    )
    ;;
  reference_excluded)
    METHOD_ARGS=(
      "${COMMON_ADAPTIVE[@]}"
      --adaptive-frame-representation global_pool
      --adaptive-grouping serial
      --adaptive-reference-selection first
      --adaptive-reference-excluded
      --adaptive-update-policy stage_update
      --adaptive-frame-fusion direct
      --adaptive-frame-fusion-weighting similarity
      --adaptive-token-merging fast_bipartite
    )
    ;;
  initial_uniform)
    METHOD_ARGS=(
      "${COMMON_ADAPTIVE[@]}"
      --adaptive-frame-representation global_pool
      --adaptive-grouping serial
      --adaptive-reference-selection first
      --adaptive-update-policy initial_only
      --adaptive-frame-fusion direct
      --adaptive-frame-fusion-weighting uniform
      --adaptive-token-merging fast_bipartite
    )
    ;;
  per_layer_tokenwise)
    METHOD_ARGS=(
      "${COMMON_ADAPTIVE[@]}"
      --adaptive-frame-representation global_pool
      --adaptive-grouping serial
      --adaptive-reference-selection first
      --adaptive-update-policy per_layer_update
      --adaptive-frame-fusion token_wise
      --adaptive-token-merging fast_bipartite
    )
    ;;
  category_proportional)
    METHOD_ARGS=(
      "${COMMON_ADAPTIVE[@]}"
      --adaptive-frame-representation global_pool
      --adaptive-grouping serial
      --adaptive-reference-selection first
      --adaptive-update-policy stage_update
      --adaptive-frame-fusion direct
      --adaptive-frame-fusion-weighting similarity
      --adaptive-token-merging category_topk_norm
      --adaptive-token-cluster-budget proportional
    )
    ;;
  category_dispersion)
    METHOD_ARGS=(
      "${COMMON_ADAPTIVE[@]}"
      --adaptive-frame-representation global_pool
      --adaptive-grouping serial
      --adaptive-reference-selection first
      --adaptive-update-policy stage_update
      --adaptive-frame-fusion direct
      --adaptive-frame-fusion-weighting similarity
      --adaptive-token-merging category_topk_norm
      --adaptive-token-cluster-budget dispersion
    )
    ;;
  *)
    echo "Unknown method key: $METHOD" >&2
    exit 2
    ;;
esac
fi

METHOD_ROOT="$RUN_ROOT/$METHOD/$DATASET"
WORK_DIR="$METHOD_ROOT/_work"
SUMMARY_DIR="$METHOD_ROOT/summaries"
LOG="$METHOD_ROOT/run.log"

# Baselines establish the shared reference first. Branch jobs wait for an
# explicit release marker so an experiment matrix can be revised safely.
if [[ "$METHOD" != "baseline_omega_partial" && -z "$MATRIX_ROW" ]]; then
  while [[ ! -f "$RUN_ROOT/.release_branch_jobs" ]]; do
    sleep 30
  done
fi

mkdir -p "$SUMMARY_DIR"
rm -rf "$WORK_DIR"

CUDA_VISIBLE_DEVICES="$GPU" \
PYTHONPATH="$ROOT" \
PYTHONNOUSERSITE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
HF_HOME="$ROOT/.cache/huggingface" \
TRANSFORMERS_CACHE="$ROOT/.cache/huggingface/hub" \
"$PYTHON_BIN" "$ROOT/inference/infer.py" \
  --dataset "$DATASET_NAME" \
  --dataset-root "$DATASET_ROOT" \
  "${DATASET_ARGS[@]}" \
  --sequences "${SEQUENCES[@]}" \
  --max-frames-per-seq 300 \
  --window-size 0 \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$WORK_DIR" \
  --overwrite \
  --eval \
  --eval-align scale_shift \
  "${METHOD_ARGS[@]}" \
  > "$LOG" 2>&1

for artifact in \
  "$WORK_DIR/$DATASET_NAME/_summary_complete_scale_shift.json" \
  "$WORK_DIR/$DATASET_NAME/_summary_scale_shift.json" \
  "$WORK_DIR/$DATASET_NAME/_sequence_metrics_scale_shift.csv" \
  "$WORK_DIR/$DATASET_NAME/_summary_pose_auc.json"; do
  if [[ -f "$artifact" ]]; then
    cp "$artifact" "$SUMMARY_DIR/$(basename "$artifact")"
  fi
done

# Keep sequence-level evaluation evidence even though dense prediction arrays
# and previews under _work are disposable.  This preserves per-sequence pose
# AUC and inference timing for later analysis.
SEQUENCE_WORK_ROOT="$WORK_DIR/$DATASET_NAME"
SEQUENCE_SUMMARY_ROOT="$SUMMARY_DIR/sequences"
if [[ -d "$SEQUENCE_WORK_ROOT" ]]; then
  while IFS= read -r -d '' artifact; do
    relative_path="${artifact#"$SEQUENCE_WORK_ROOT/"}"
    destination="$SEQUENCE_SUMMARY_ROOT/$relative_path"
    mkdir -p "$(dirname "$destination")"
    cp "$artifact" "$destination"
  done < <(
    find "$SEQUENCE_WORK_ROOT" -type f \( -name '_pose_auc.json' -o -name '_time.json' \) -print0
  )
fi

rm -rf "$WORK_DIR"
touch "$METHOD_ROOT/.done"
