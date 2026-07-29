#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/auc_eval_results/1new}"
RUNNER="$ROOT/scripts/run_adaptive_frame_token_branch.sh"
SCRIPT_PATH="$ROOT/scripts/launch_adaptive_frame_token_branch_ablation.sh"

mkdir -p "$RUN_ROOT/logs"
rm -f "$RUN_ROOT"/baseline_omega_partial/{tum,7scenes}/.done

run_task() {
  local gpu="$1"
  local dataset="$2"
  local method="$3"
  bash "$RUNNER" "$gpu" "$dataset" "$method"
}

wait_for_baselines() {
  until [[ -f "$RUN_ROOT/baseline_omega_partial/tum/.done" && -f "$RUN_ROOT/baseline_omega_partial/7scenes/.done" ]]; do
    sleep 30
  done
}

launch_gpu5() {
  run_task 5 tum baseline_omega_partial
  wait_for_baselines
  run_task 5 tum default
  run_task 5 tum cluster_parallel_medoid
  run_task 5 tum spatial_serial_diverse
  run_task 5 tum reference_excluded
  run_task 5 7scenes initial_uniform
}

launch_gpu6() {
  run_task 6 7scenes baseline_omega_partial
  wait_for_baselines
  run_task 6 7scenes default
  run_task 6 7scenes cluster_parallel_medoid
  run_task 6 7scenes spatial_serial_diverse
  run_task 6 7scenes reference_excluded
  run_task 6 tum initial_uniform
}

launch_gpu7() {
  wait_for_baselines
  run_task 7 tum per_layer_tokenwise
  run_task 7 7scenes per_layer_tokenwise
  run_task 7 tum category_proportional
  run_task 7 7scenes category_proportional
  run_task 7 tum category_dispersion
  run_task 7 7scenes category_dispersion
}

if [[ "${1:-}" == "--worker" ]]; then
  case "${2:-}" in
    gpu5) launch_gpu5 ;;
    gpu6) launch_gpu6 ;;
    gpu7) launch_gpu7 ;;
    *) echo "usage: $0 --worker <gpu5|gpu6|gpu7>" >&2; exit 2 ;;
  esac
  exit 0
fi

screen -dmS adaptive1new_gpu5 bash -lc "cd '$ROOT' && bash '$SCRIPT_PATH' --worker gpu5 > '$RUN_ROOT/logs/gpu5.log' 2>&1"
screen -dmS adaptive1new_gpu6 bash -lc "cd '$ROOT' && bash '$SCRIPT_PATH' --worker gpu6 > '$RUN_ROOT/logs/gpu6.log' 2>&1"
screen -dmS adaptive1new_gpu7 bash -lc "cd '$ROOT' && bash '$SCRIPT_PATH' --worker gpu7 > '$RUN_ROOT/logs/gpu7.log' 2>&1"

printf 'Started screens: adaptive1new_gpu5, adaptive1new_gpu6, adaptive1new_gpu7\n'
