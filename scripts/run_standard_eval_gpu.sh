#!/usr/bin/env bash
set -euo pipefail
dataset=$1
gpu=$2
root=/data/mmc_syang/Pi3
out=${root}/outputs/standard_${dataset}_300
python_bin=/data/mmc_syang/miniconda3/envs/flow3r/bin/python
mkdir -p "${out}/logs"
exec env PYTHONUNBUFFERED=1 PYTHONPATH="${root}" CUDA_VISIBLE_DEVICES="${gpu}" \
  "${python_bin}" "${root}/scripts/eval_standard_tum_7scenes.py" \
  --dataset "${dataset}" \
  --dataset-root "/data/mmc_syang/dataset/$([[ ${dataset} == 7scenes ]] && echo 7scenes || echo TUM-Dynamics)" \
  --pretrained "${root}/checkpoints/Pi3" \
  --output-dir "${out}" --device cuda:0 --max-frames-per-seq 300 \
  --frame-sample-mode uniform --timing-repeats 3 \
  > "${out}/logs/${dataset}_gpu${gpu}.log" 2>&1
