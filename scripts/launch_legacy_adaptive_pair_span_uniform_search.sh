#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/auc_eval_results/01/ours_adaptive_pair_span_uniform300_search}"
SEARCHER="$ROOT/scripts/search_legacy_adaptive_pair_span_uniform.py"

mkdir -p "$RESULT_ROOT/logs"
/data/mmc_syang/miniconda3/envs/fastvggt/bin/python "$SEARCHER" \
  --initialize \
  --result-root "$RESULT_ROOT" \
  --max-fine-trials "${MAX_FINE_TRIALS:-0}"

for gpu in 5 6 7; do
  session="adaptive_pair_span_uniform_g${gpu}"
  screen -S "$session" -X quit >/dev/null 2>&1 || true
  screen -dmS "$session" bash -lc \
    "cd '$ROOT' && RESULT_ROOT='$RESULT_ROOT' MAX_FINE_TRIALS='${MAX_FINE_TRIALS:-0}' bash scripts/wait_and_run_legacy_adaptive_pair_span_search.sh '$gpu'"
  echo "started ${session}; it waits for the current uniform300 job on GPU ${gpu}"
done
