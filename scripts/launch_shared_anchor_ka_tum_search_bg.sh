#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
STAMP="${STAMP:-$(date '+%Y%m%d_%H%M%S')}"
RUN_ROOT="${RUN_ROOT:-$ROOT/auc_eval_results/01/shared_anchor_ka_search_${STAMP}}"
mkdir -p "$RUN_ROOT"
printf '%s\n' "$RUN_ROOT" > "$RUN_ROOT/run_root.txt"

for spec in '5 300' '6 400' '7 500'; do
  read -r gpu frames <<< "$spec"
  session="ka_search_t${frames}_gpu${gpu}_${STAMP}"
  log="$RUN_ROOT/launcher_t${frames}_gpu${gpu}.log"
  screen -dmS "$session" bash -lc "cd '$ROOT' && exec bash scripts/search_shared_anchor_ka_tum.sh '$gpu' '$frames' '$RUN_ROOT' > '$log' 2>&1"
  printf '%s\t%s\t%s\n' "$session" "$gpu" "$frames" >> "$RUN_ROOT/sessions.tsv"
done

echo "RUN_ROOT=$RUN_ROOT"
