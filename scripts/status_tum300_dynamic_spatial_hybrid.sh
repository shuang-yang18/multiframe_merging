#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/new_results/1/tum300_dynamic_spatial_hybrid_20260724}"

for spec in "4 all" "5 middle" "6 late" "7 middle_late"; do
  read -r gpu schedule <<< "$spec"
  session="vggt_dynamic_hybrid_${schedule}_gpu${gpu}"
  log_file="$RUN_ROOT/logs/${schedule}_gpu${gpu}.log"
  echo "== GPU $gpu: $schedule =="
  if screen -list | grep -q "[.]${session}[[:space:]]"; then
    echo "running: $session"
  else
    echo "screen finished or not started"
  fi
  [[ -f "$log_file" ]] && tail -20 "$log_file" || echo "no log yet"
  echo
done
