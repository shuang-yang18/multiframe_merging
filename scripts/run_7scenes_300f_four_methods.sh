#!/usr/bin/env bash
# Run one 7Scenes sequence (8 frames, fixed stride 3) serially on GPU 0.
# LiteVGGT's FP8 kernels require the flattened token count to be divisible by
# eight.  At 504x504 each frame has 1301 tokens, therefore 8 (not 10) is the
# smallest valid common smoke-test length.  The production 500/1000-frame
# settings are already valid.
# Baseline is deliberately last: it is the unmodified reference check after
# every acceleration backend has exercised its own required environment.
set -euo pipefail

WORKSPACE_ROOT=${WORKSPACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
VGGT_ROOT=${VGGT_ROOT:-${WORKSPACE_ROOT}/vggt}
DATASET_ROOT=${DATASET_ROOT:-${WORKSPACE_ROOT}/dataset/7scenes}
SEQUENCE=${SEQUENCE:-chess/seq-03}
OUTPUT_ROOT=${OUTPUT_ROOT:-${VGGT_ROOT}/outputs/7scenes_8f_four_methods}
VGGT_ENV=${VGGT_ENV:-vggt28}
LITEVGGT_ENV=${LITEVGGT_ENV:-litevggt}
COME_ENV=${COME_ENV:-come}
SPARSE_VGGT_ROOT=${SPARSE_VGGT_ROOT:-${WORKSPACE_ROOT}/sparse-vggt}
SPARGEATTN_ROOT=${SPARGEATTN_ROOT:-${WORKSPACE_ROOT}/SpargeAttn-sm120}

mkdir -p "${OUTPUT_ROOT}/logs"
failures=0

run_case() {
  local env_name=$1 method=$2
  local -a conda_selector
  if [[ "${env_name}" = /* ]]; then
    conda_selector=(-p "${env_name}")
  else
    conda_selector=(-n "${env_name}")
  fi

  echo "[$(date '+%F %T')] GPU 0: starting ${method} in ${env_name}"
  if conda run --no-capture-output "${conda_selector[@]}" env \
      CUDA_VISIBLE_DEVICES=0 \
      PYTHONNOUSERSITE=1 \
      OMP_NUM_THREADS=1 \
      MMC_SYANG_ROOT=${WORKSPACE_ROOT} \
      SPARSE_VGGT_ROOT=${SPARSE_VGGT_ROOT} \
      SPARGEATTN_ROOT=${SPARGEATTN_ROOT} \
      PYTHONPATH="${SPARGEATTN_ROOT}:${VGGT_ROOT}:${WORKSPACE_ROOT}:${WORKSPACE_ROOT}/Pi3${PYTHONPATH:+:${PYTHONPATH}}" \
      python "${VGGT_ROOT}/scripts/benchmark_7scenes_vggt_methods.py" \
        --method "${method}" \
        --dataset-root "${DATASET_ROOT}" \
        --sequence "${SEQUENCE}" \
        --num-frames 8 \
        --sampling-stride 3 \
        --image-size 504 \
        --timing-repeats 1 \
        --checkpoint "${VGGT_ROOT}/ckpts/model.pt" \
        --output "${OUTPUT_ROOT}/${method}/metrics.json" \
      >"${OUTPUT_ROOT}/logs/${method}_gpu0.log" 2>&1; then
    echo "[$(date '+%F %T')] GPU 0: completed ${method}"
  else
    echo "[$(date '+%F %T')] GPU 0: FAILED ${method}; continuing with remaining methods"
    failures=$((failures + 1))
  fi
}

run_case "${LITEVGGT_ENV}" litevggt
run_case "${COME_ENV}" come
run_case "${VGGT_ENV}" fastvggt
run_case "${VGGT_ENV}" da-vggt
run_case "${VGGT_ENV}" sparse-vggt
run_case "${VGGT_ENV}" u-m
run_case "${VGGT_ENV}" avggt
run_case "${VGGT_ENV}" baseline

echo "Completed. JSON results: ${OUTPUT_ROOT}/{baseline,fastvggt,da-vggt,sparse-vggt,u-m,avggt,litevggt,come}/metrics.json"
exit "${failures}"
