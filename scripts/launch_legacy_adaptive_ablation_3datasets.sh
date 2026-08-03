#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/auc_eval_results/legacy_adaptive_ablation_20260731}"
ABLATION_MODE="${ABLATION_MODE:-all}"
mkdir -p "$RUN_ROOT/logs"

launch() {
  local gpu="$1"
  local dataset="$2"
  local session="legacy_ablation_${dataset}_g${gpu}"
  local log="$RUN_ROOT/logs/${dataset}_gpu${gpu}.log"
  screen -S "$session" -X quit >/dev/null 2>&1 || true
  screen -dmS "$session" bash -lc "cd '$ROOT' && RUN_ROOT='$RUN_ROOT' ABLATION_MODE='$ABLATION_MODE' bash scripts/run_legacy_adaptive_ablation_eval.sh '$gpu' '$dataset' > '$log' 2>&1"
  echo "started $session; log: $log"
}

launch 5 tum
launch 6 7scenes
launch 7 nrgbd
