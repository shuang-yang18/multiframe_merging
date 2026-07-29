#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/auc_eval_results/1new}"
RUNNER="$ROOT/scripts/run_adaptive_frame_token_branch.sh"
MATRIX_FILE="$ROOT/scripts/adaptive_frame_token_pairwise_matrix.tsv"
SCRIPT_PATH="$ROOT/scripts/launch_adaptive_frame_token_pairwise_matrix.sh"

run_task() {
  local gpu="$1"
  local method="$2"
  bash "$RUNNER" "$gpu" tum "$method"
  bash "$RUNNER" "$gpu" 7scenes "$method"
}

run_worker() {
  local gpu="$1"
  shift
  for method in "$@"; do
    run_task "$gpu" "$method"
  done
}

wait_for_baselines() {
  until [[ -f "$RUN_ROOT/baseline_omega_partial/tum/.done" && -f "$RUN_ROOT/baseline_omega_partial/7scenes/.done" ]]; do
    sleep 30
  done
}

if [[ "${1:-}" == "--worker" ]]; then
  case "${2:-}" in
    gpu5) run_worker 5 pairwise_01 pairwise_02 pairwise_03 pairwise_04 pairwise_05 ;;
    gpu6) run_worker 6 pairwise_06 pairwise_07 pairwise_08 pairwise_09 pairwise_10 ;;
    gpu7) run_worker 7 pairwise_11 pairwise_12 pairwise_13 pairwise_14 pairwise_15 ;;
    *) echo "usage: $0 --worker <gpu5|gpu6|gpu7>" >&2; exit 2 ;;
  esac
  exit 0
fi

wait_for_baselines
for session in adaptive1new_gpu5 adaptive1new_gpu6 adaptive1new_gpu7; do
  screen -S "$session" -X quit 2>/dev/null || true
done

mkdir -p "$RUN_ROOT/logs"
cp "$MATRIX_FILE" "$RUN_ROOT/pairwise_matrix.tsv"
screen -dmS adaptive1new_pairwise_gpu5 bash -lc "cd '$ROOT' && bash '$SCRIPT_PATH' --worker gpu5 > '$RUN_ROOT/logs/pairwise_gpu5.log' 2>&1"
screen -dmS adaptive1new_pairwise_gpu6 bash -lc "cd '$ROOT' && bash '$SCRIPT_PATH' --worker gpu6 > '$RUN_ROOT/logs/pairwise_gpu6.log' 2>&1"
screen -dmS adaptive1new_pairwise_gpu7 bash -lc "cd '$ROOT' && bash '$SCRIPT_PATH' --worker gpu7 > '$RUN_ROOT/logs/pairwise_gpu7.log' 2>&1"

printf 'Started pairwise matrix screens after both baselines completed.\n'
