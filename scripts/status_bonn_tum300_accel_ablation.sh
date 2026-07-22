#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOG_DIR="${LOG_DIR:-$ROOT/outputs/logs}"

for gpu in 6 7; do
  session="bonn_tum300_accel_gpu${gpu}"
  pid_path="$LOG_DIR/bonn_tum300_accel_gpu${gpu}.pid"
  log_path="$LOG_DIR/bonn_tum300_accel_gpu${gpu}.log"
  echo "== GPU ${gpu} =="
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "running tmux=${session}"
  elif [[ -f "$pid_path" ]]; then
    echo "not running; last marker=$(cat "$pid_path")"
  else
    echo "not launched"
  fi
  if [[ -f "$log_path" ]]; then
    tail -20 "$log_path"
  else
    echo "no log yet: $log_path"
  fi
  echo
done
