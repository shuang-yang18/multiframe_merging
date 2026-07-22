#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOG="$ROOT/new_results/logs/stop_search_pair_span_2330.log"
PGID="${PGID:-4148475}"
TARGET="${TARGET:-2026-07-18 23:30:00 UTC}"

mkdir -p "$(dirname "$LOG")"
target_ts=$(date -d "$TARGET" +%s)
now_ts=$(date +%s)
delay=$((target_ts - now_ts))
if [[ "$delay" -lt 0 ]]; then
  delay=0
fi

echo "[$(date '+%F %T %Z')] scheduled stop for PGID ${PGID} in ${delay}s at ${TARGET}" >> "$LOG"
sleep "$delay"
echo "[$(date '+%F %T %Z')] sending TERM to -${PGID}" >> "$LOG"
kill -TERM "-${PGID}" 2>> "$LOG" || true
sleep 30
if ps -p "$PGID" >/dev/null 2>&1; then
  echo "[$(date '+%F %T %Z')] sending KILL to -${PGID}" >> "$LOG"
  kill -KILL "-${PGID}" 2>> "$LOG" || true
else
  echo "[$(date '+%F %T %Z')] PGID ${PGID} stopped after TERM" >> "$LOG"
fi
