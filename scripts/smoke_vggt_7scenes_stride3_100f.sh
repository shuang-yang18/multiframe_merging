#!/usr/bin/env bash
# Smoke-test the unmodified VGGT and U-M paths on every official 7Scenes test
# sequence. Each sequence is evaluated with 100 frames sampled at stride 3.
set -euo pipefail

ROOT=${ROOT:-/data/mmc_syang}
VGGT_ROOT=${VGGT_ROOT:-${ROOT}/vggt}
VGGT_PYTHON=${VGGT_PYTHON:-${ROOT}/miniconda3/envs/fastvggt/bin/python}
VGGT_CHECKPOINT=${VGGT_CHECKPOINT:-${VGGT_ROOT}/ckpts/model.pt}
SEVEN_SCENES_ROOT=${SEVEN_SCENES_ROOT:-${ROOT}/dataset/7scenes}
OUTPUT_ROOT=${OUTPUT_ROOT:-${VGGT_ROOT}/outputs/smoke_vggt_7scenes_stride3_100f}
GPU=${1:-0}
METHOD_FILTER=${2:-all}

[[ ${GPU} =~ ^[0-9]+$ ]] || { echo "GPU ID must be a non-negative integer: ${GPU}" >&2; exit 2; }
[[ ${METHOD_FILTER} == all || ${METHOD_FILTER} == baseline || ${METHOD_FILTER} == u-m ]] || {
  echo "Method must be baseline, u-m, or all: ${METHOD_FILTER}" >&2
  exit 2
}
[[ -x ${VGGT_PYTHON} ]] || { echo "Python executable not found: ${VGGT_PYTHON}" >&2; exit 2; }
[[ -f ${VGGT_CHECKPOINT} ]] || { echo "Checkpoint not found: ${VGGT_CHECKPOINT}" >&2; exit 2; }
[[ -d ${SEVEN_SCENES_ROOT} ]] || { echo "7Scenes root not found: ${SEVEN_SCENES_ROOT}" >&2; exit 2; }

export CUDA_VISIBLE_DEVICES=${GPU}
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export VGGT_UM_TRITON=1
export MMC_SYANG_ROOT=${ROOT}
export PYTHONPATH=${VGGT_ROOT}:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}

cd "${VGGT_ROOT}"
for method in baseline u-m; do
  [[ ${METHOD_FILTER} == all || ${METHOD_FILTER} == ${method} ]] || continue
  if [[ ${method} == baseline ]]; then
    acceleration_method=none
    extra_args=(--merge-ratio 0)
  else
    acceleration_method=u-m
    extra_args=(--merge-ratio 0 --um-lambda 0.04 --um-spatial-radius 2 --um-temporal-window 4 --um-refresh-layers 0,9,21)
  fi

  echo "[$(date -u -Is)] VGGT ${method}: 7Scenes test, 100 frames, stride 3"
  "${VGGT_PYTHON}" scripts/eval_standard_tum_7scenes.py \
    --dataset 7scenes \
    --dataset-root "${SEVEN_SCENES_ROOT}" \
    --checkpoint "${VGGT_CHECKPOINT}" \
    --output-dir "${OUTPUT_ROOT}/${method}" \
    --device cuda:0 \
    --num-frames 100 \
    --sampling-stride 3 \
    --image-resolution 518 \
    --timing-repeats 1 \
    --acceleration-method "${acceleration_method}" \
    --overwrite \
    "${extra_args[@]}"
done
