#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-outputs/tum_bonn300_accel_8groups_20260722}"
LOG_DIR="${LOG_DIR:-$ROOT/outputs/logs}"

cd "$ROOT"
mkdir -p "$LOG_DIR" "$RUN_ROOT"
export RUN_ROOT
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"

start_worker() {
  local gpu="$1"
  local session="tum_bonn300_accel_8groups_gpu${gpu}"
  local log_path="$LOG_DIR/tum_bonn300_accel_8groups_gpu${gpu}.log"

  if tmux has-session -t "$session" 2>/dev/null; then
    echo "GPU ${gpu} worker already running: tmux=${session}, log=${log_path}"
    return
  fi

  tmux new-session -d -s "$session" \
    "cd '$ROOT' && RUN_ROOT='$RUN_ROOT' bash '$ROOT/scripts/run_tum_bonn300_accel_8groups.sh' '$gpu' > '$log_path' 2>&1"
  echo "started GPU ${gpu} worker: tmux=${session}, log=${log_path}"
}

start_worker 5
start_worker 6
start_worker 7

echo
echo "Monitor:"
echo "  bash scripts/status_tum_bonn300_accel_8groups.sh"
echo
echo "Collect metrics after all workers finish:"
echo "  python scripts/collect_tum_bonn300_accel_8groups_metrics.py --run-root '$RUN_ROOT'"

