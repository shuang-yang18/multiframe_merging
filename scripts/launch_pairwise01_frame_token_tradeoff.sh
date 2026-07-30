#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
METHOD="${METHOD:-pairwise_01}"
BASELINE_FRAME_TOKEN_RATIO="${BASELINE_FRAME_TOKEN_RATIO:-0.44248179803735344}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/auc_eval_results/1new_pairwise01_frame_up_token_down}"
SEARCHER="$ROOT/scripts/search_pairwise01_frame_token_tradeoff.py"
KEEPS=(0.75 0.70 0.65 0.60 0.55 0.50 0.45 0.40 0.35)

mkdir -p "$RESULT_ROOT/logs"

launch_worker() {
  local gpu="$1"
  local threshold="$2"
  local session="${METHOD}_tradeoff_gpu${gpu}"
  screen -S "$session" -X quit >/dev/null 2>&1 || true
  screen -dmS "$session" bash -lc \
    "cd '$ROOT' && exec python '$SEARCHER' --gpu '$gpu' --method '$METHOD' --baseline-frame-token-ratio '$BASELINE_FRAME_TOKEN_RATIO' --group-threshold '$threshold' --keep-ratios ${KEEPS[*]} --result-root '$RESULT_ROOT' > '$RESULT_ROOT/logs/gpu${gpu}.log' 2>&1"
  echo "Started $session: group_threshold=$threshold"
}

launch_worker 4 0.9982
launch_worker 5 0.9985
launch_worker 6 0.9988
launch_worker 7 0.9990
