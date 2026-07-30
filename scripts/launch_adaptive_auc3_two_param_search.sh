#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/auc_eval_results/1new_auc3_two_param_search}"
SEARCHER="$ROOT/scripts/search_adaptive_auc3_two_params.py"

mkdir -p "$RESULT_ROOT/logs"

launch_worker() {
  local gpu="$1"
  local method="$2"
  local session="adaptive_auc3_${method}_gpu${gpu}"
  screen -S "$session" -X quit >/dev/null 2>&1 || true
  screen -dmS "$session" bash -lc \
    "cd '$ROOT' && exec python '$SEARCHER' --gpu '$gpu' --method '$method' --result-root '$RESULT_ROOT' > '$RESULT_ROOT/logs/${method}_gpu${gpu}.log' 2>&1"
  echo "Started $session on GPU $gpu: $method"
}

launch_worker 4 auc3_p01_initial_uniform
launch_worker 5 auc3_p01_stage_uniform
launch_worker 6 auc3_p12_perlayer_similarity
launch_worker 7 auc3_p12_perlayer_uniform
