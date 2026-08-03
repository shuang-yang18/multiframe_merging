#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/auc_eval_results/1new_stage_update_wide_01_search}"
SEARCHER="$ROOT/scripts/search_adaptive_auc3_two_params.py"
METHOD="auc3_p01_stage_uniform"
KEEP_RATIOS="0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"

mkdir -p "$RESULT_ROOT/logs"

launch_worker() {
  local gpu="$1"
  local thresholds="$2"
  local worker_root="$RESULT_ROOT/gpu${gpu}"
  local session="adaptive_stage_wide_gpu${gpu}"
  screen -S "$session" -X quit >/dev/null 2>&1 || true
  screen -dmS "$session" bash -lc \
    "cd '$ROOT' && exec python '$SEARCHER' --gpu '$gpu' --method '$METHOD' --result-root '$worker_root' --group-thresholds $thresholds --token-keep-ratios $KEEP_RATIOS > '$RESULT_ROOT/logs/gpu${gpu}.log' 2>&1"
  echo "Started $session: thresholds=[$thresholds]"
}

launch_worker 4 "0.0 0.1 0.2"
launch_worker 5 "0.3 0.4 0.5"
launch_worker 6 "0.6 0.7 0.8"
launch_worker 7 "0.9 1.0"
