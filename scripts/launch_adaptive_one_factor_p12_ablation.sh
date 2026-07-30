#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/auc_eval_results/1new_one_factor_pairwise12_k080}"
RUNNER="$ROOT/scripts/run_adaptive_frame_token_branch.sh"
MATRIX_FILE="$ROOT/scripts/adaptive_frame_token_one_factor_p12_matrix.tsv"
ADAPTIVE_GROUP_SIMILARITY_THRESHOLD="${ADAPTIVE_GROUP_SIMILARITY_THRESHOLD:-0.998}"
ADAPTIVE_GROUP_MAX_SIZE="${ADAPTIVE_GROUP_MAX_SIZE:-3}"
ADAPTIVE_FRAME_TOKEN_SIMILARITY_THRESHOLD="${ADAPTIVE_FRAME_TOKEN_SIMILARITY_THRESHOLD:-0.995}"
ADAPTIVE_TOKEN_KEEP_RATIO="${ADAPTIVE_TOKEN_KEEP_RATIO:-0.8}"

mkdir -p "$RUN_ROOT/logs"
cp "$MATRIX_FILE" "$RUN_ROOT/one_factor_matrix.tsv"

launch_worker() {
  local gpu="$1"
  shift
  local session="adaptive_onefactor_p12_gpu${gpu}"
  screen -S "$session" -X quit >/dev/null 2>&1 || true
  screen -dmS "$session" bash -lc \
    "cd '$ROOT' && run_worker() { local gpu=\"\$1\"; shift; for method in \"\$@\"; do MATRIX_FILE='$MATRIX_FILE' RUN_ROOT='$RUN_ROOT' ADAPTIVE_GROUP_SIMILARITY_THRESHOLD='$ADAPTIVE_GROUP_SIMILARITY_THRESHOLD' ADAPTIVE_GROUP_MAX_SIZE='$ADAPTIVE_GROUP_MAX_SIZE' ADAPTIVE_FRAME_TOKEN_SIMILARITY_THRESHOLD='$ADAPTIVE_FRAME_TOKEN_SIMILARITY_THRESHOLD' ADAPTIVE_TOKEN_KEEP_RATIO='$ADAPTIVE_TOKEN_KEEP_RATIO' bash '$RUNNER' \"\$gpu\" tum \"\$method\"; MATRIX_FILE='$MATRIX_FILE' RUN_ROOT='$RUN_ROOT' ADAPTIVE_GROUP_SIMILARITY_THRESHOLD='$ADAPTIVE_GROUP_SIMILARITY_THRESHOLD' ADAPTIVE_GROUP_MAX_SIZE='$ADAPTIVE_GROUP_MAX_SIZE' ADAPTIVE_FRAME_TOKEN_SIMILARITY_THRESHOLD='$ADAPTIVE_FRAME_TOKEN_SIMILARITY_THRESHOLD' ADAPTIVE_TOKEN_KEEP_RATIO='$ADAPTIVE_TOKEN_KEEP_RATIO' bash '$RUNNER' \"\$gpu\" 7scenes \"\$method\"; done; }; run_worker '$gpu' $* > '$RUN_ROOT/logs/gpu${gpu}.log' 2>&1"
  echo "Started $session: $*"
}

launch_worker 4 pairwise_ofa12_00_baseline pairwise_ofa12_01_rep_cluster pairwise_ofa12_02_rep_spatial pairwise_ofa12_03_group_parallel
launch_worker 5 pairwise_ofa12_04_ref_medoid pairwise_ofa12_05_ref_diverse pairwise_ofa12_06_ref_excluded
launch_worker 6 pairwise_ofa12_07_update_initial pairwise_ofa12_08_update_stage pairwise_ofa12_09_fusion_uniform
launch_worker 7 pairwise_ofa12_10_fusion_tokenwise pairwise_ofa12_11_token_proportional pairwise_ofa12_12_token_dispersion
