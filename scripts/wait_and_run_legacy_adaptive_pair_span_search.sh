#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GPU="${1:?usage: wait_and_run_legacy_adaptive_pair_span_search.sh <gpu: 5|6|7>}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/auc_eval_results/01/ours_adaptive_pair_span_uniform300_search}"
SEARCHER="$ROOT/scripts/search_legacy_adaptive_pair_span_uniform.py"

case "$GPU" in
  5) UPSTREAM_SESSION="uniform300_sparse_g5" ;;
  6) UPSTREAM_SESSION="uniform300_da_g6" ;;
  7) UPSTREAM_SESSION="uniform300_da_fastvggt_g7" ;;
  *) echo "Expected GPU 5, 6, or 7; got $GPU" >&2; exit 2 ;;
esac

mkdir -p "$RESULT_ROOT/logs"
while screen -ls 2>/dev/null | grep -q "[.]${UPSTREAM_SESSION}"; do
  echo "[$(date '+%F %T')] waiting for ${UPSTREAM_SESSION} before using GPU ${GPU}" >> "$RESULT_ROOT/logs/worker_gpu${GPU}.log"
  sleep 60
done

echo "[$(date '+%F %T')] starting pair/span search on GPU ${GPU}" >> "$RESULT_ROOT/logs/worker_gpu${GPU}.log"
exec /data/mmc_syang/miniconda3/envs/fastvggt/bin/python "$SEARCHER" \
  --worker \
  --gpu "$GPU" \
  --result-root "$RESULT_ROOT" \
  --max-fine-trials "${MAX_FINE_TRIALS:-0}" \
  >> "$RESULT_ROOT/logs/worker_gpu${GPU}.log" 2>&1
