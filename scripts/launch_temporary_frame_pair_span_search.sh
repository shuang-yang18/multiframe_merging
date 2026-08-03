#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/auc_eval_results/01/temporary_frame_pair_span_search_p0960_s0950}"
PYTHON_BIN="${PYTHON_BIN:-/data/mmc_syang/miniconda3/envs/fastvggt/bin/python}"

mkdir -p "$RESULT_ROOT/logs"
"$PYTHON_BIN" "$ROOT/scripts/search_temporary_frame_pair_span.py" --initialize --result-root "$RESULT_ROOT"

for gpu in 5 6 7; do
  session="temporary_frame_pair_span_g${gpu}"
  screen -S "$session" -X quit >/dev/null 2>&1 || true
  screen -dmS "$session" bash -lc \
    "cd '$ROOT' && exec '$PYTHON_BIN' scripts/search_temporary_frame_pair_span.py --worker --gpu '$gpu' --result-root '$RESULT_ROOT' >> '$RESULT_ROOT/logs/worker_gpu${gpu}.log' 2>&1"
  echo "started $session"
done
