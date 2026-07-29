#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/new_results/1/tum300_dynamic_spatial_fastvggt_20260724}"
LOG_DIR="$RUN_ROOT/logs"
mkdir -p "$LOG_DIR"

launch() {
  local gpu="$1"
  local schedule="$2"
  local session="vggt_dynamic_spatial_${schedule}_gpu${gpu}"
  local log_file="$LOG_DIR/${schedule}_gpu${gpu}.log"
  screen -dmS "$session" bash -lc \
    "cd '$ROOT' && while nvidia-smi -i '$gpu' --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -Eq '[0-9]'; do printf '[%s] GPU %s occupied; waiting\\n' \"\$(date '+%F %T')\" '$gpu' >> '$log_file'; sleep 60; done; bash scripts/run_tum300_dynamic_spatial_fastvggt.sh '$gpu' '$schedule' >> '$log_file' 2>&1"
  echo "$session -> GPU $gpu, schedule=$schedule, log=$log_file"
}

launch 4 all
launch 5 middle
launch 6 late
launch 7 middle_late
