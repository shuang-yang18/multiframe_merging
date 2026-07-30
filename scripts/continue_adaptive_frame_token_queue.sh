#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?usage: $0 <gpu> <predecessor-method> <next-method> [next-method ...]}"
PREDECESSOR="${2:?usage: $0 <gpu> <predecessor-method> <next-method> [next-method ...]}"
shift 2

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/auc_eval_results/1new}"
RUNNER="$ROOT/scripts/run_adaptive_frame_token_branch.sh"

wait_for_method() {
  local method="$1"
  until [[ -f "$RUN_ROOT/$method/tum/.done" && -f "$RUN_ROOT/$method/7scenes/.done" ]]; do
    sleep 30
  done
}

wait_for_method "$PREDECESSOR"
# The predecessor writes .done immediately before its shell exits. Leave a
# brief handoff gap so the next inference never overlaps on the same GPU.
sleep 10

for method in "$@"; do
  bash "$RUNNER" "$GPU" tum "$method"
  bash "$RUNNER" "$GPU" 7scenes "$method"
done
