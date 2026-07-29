#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/new_results/frame_anchor_adaptive_tum300_20260723}"
LOG_DIR="$RUN_ROOT/logs"
mkdir -p "$LOG_DIR"

launch() {
  local gpu="$1"
  local variant="$2"
  local session="vggt_anchor_adaptive_${variant}_gpu${gpu}"
  local log_file="$LOG_DIR/${variant}_gpu${gpu}.log"
  screen -dmS "$session" bash -lc \
    "cd '$ROOT' && while nvidia-smi -i '$gpu' --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -Eq '[0-9]'; do printf '[%s] GPU %s occupied; waiting\\n' \"\$(date '+%F %T')\" '$gpu' >> '$log_file'; sleep 60; done; bash scripts/run_tum300_frame_anchor_adaptive.sh '$gpu' '$variant' >> '$log_file' 2>&1"
  echo "$session -> waits for GPU $gpu, then runs $variant; log: $log_file"
}

launch 6 plain
launch 7 fastvggt
