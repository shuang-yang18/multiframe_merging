#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/auc_eval_results/1new_token_ratio_target034_038}"
SEARCHER="$ROOT/scripts/search_adaptive_token_ratio_branches.py"

mkdir -p "$RUN_ROOT/logs"

launch_worker() {
  local gpu="$1"
  shift
  local session="adaptive_ratio_tune_gpu${gpu}"
  screen -S "$session" -X quit >/dev/null 2>&1 || true
  screen -dmS "$session" bash -lc \
    "cd '$ROOT' && exec python '$SEARCHER' --worker --gpu '$gpu' --result-root '$RUN_ROOT' --methods $* > '$RUN_ROOT/logs/gpu${gpu}.log' 2>&1"
  echo "Started $session on GPU $gpu: $*"
}

launch_worker 4 pairwise_01 pairwise_02 pairwise_03 pairwise_04
launch_worker 5 pairwise_05 pairwise_06 pairwise_07 pairwise_08
launch_worker 6 pairwise_09 pairwise_10 pairwise_11 pairwise_12
launch_worker 7 pairwise_13 pairwise_14 pairwise_15
