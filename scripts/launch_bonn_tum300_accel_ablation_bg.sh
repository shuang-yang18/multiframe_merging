#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-outputs/bonn_tum300_accel_20260721}"
LOG_DIR="${LOG_DIR:-$ROOT/outputs/logs}"

cd "$ROOT"
mkdir -p "$LOG_DIR" "$RUN_ROOT"

export RUN_ROOT
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"

start_worker() {
  local gpu="$1"
  local session="bonn_tum300_accel_gpu${gpu}"
  local log_path="$LOG_DIR/bonn_tum300_accel_gpu${gpu}.log"
  local pid_path="$LOG_DIR/bonn_tum300_accel_gpu${gpu}.pid"

  if tmux has-session -t "$session" 2>/dev/null; then
    echo "GPU ${gpu} worker already running: tmux=${session}, log=${log_path}"
    return
  fi

  tmux new-session -d -s "$session" \
    "cd '$ROOT' && RUN_ROOT='$RUN_ROOT' bash '$ROOT/scripts/run_bonn_tum300_accel_ablation.sh' '$gpu' > '$log_path' 2>&1"
  echo "$session" > "$pid_path"
  echo "started GPU ${gpu} worker: tmux=${session}, log=${log_path}"
}

start_worker 6
start_worker 7

echo
echo "Monitor:"
echo "  bash scripts/status_bonn_tum300_accel_ablation.sh"
echo
echo "Collect metrics after both workers finish:"
echo "  python scripts/collect_bonn_tum300_accel_metrics.py"
