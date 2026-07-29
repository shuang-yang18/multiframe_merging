#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/new_results/1/layer4_dynamic_segmentation_seqfirst_pca128_256_k236}"
SESSION="${SESSION:-vggt_layer4_cluster_seqfirst}"

mkdir -p "$RUN_ROOT/logs"
screen -dmS "$SESSION" bash -lc "cd '$ROOT' && bash scripts/run_layer4_token_cluster_sequence_first.sh"
printf 'screen session: %s\nlogs: %s/logs\n' "$SESSION" "$RUN_ROOT"
