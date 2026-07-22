#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOG_DIR="${LOG_DIR:-$ROOT/outputs/logs}"

for gpu in 5 6 7; do
  session="tum_bonn300_accel_8groups_gpu${gpu}"
  log_path="$LOG_DIR/tum_bonn300_accel_8groups_gpu${gpu}.log"
  echo "== GPU ${gpu} =="
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "running tmux=${session}"
  else
    echo "not running"
  fi
  if [[ -f "$log_path" ]]; then
    tail -30 "$log_path"
  else
    echo "no log yet: $log_path"
  fi
  echo
done

