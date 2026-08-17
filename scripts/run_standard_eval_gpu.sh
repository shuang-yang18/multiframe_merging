#!/usr/bin/env bash
set -euo pipefail

dataset=$1
gpu=$2
root=/data/mmc_syang/vggt
out=${root}/outputs/standard_${dataset}_300
if [[ ${dataset} == tum_dynamic ]]; then
    data_root=/data/mmc_syang/dataset/TUM-Dynamics
else
    data_root=/data/mmc_syang/dataset/7scenes
fi
mkdir -p "${out}/logs"
exec env PYTHONUNBUFFERED=1 PYTHONPATH="${root}" CUDA_VISIBLE_DEVICES="${gpu}" \
  /data/mmc_syang/miniconda3/envs/fastvggt/bin/python \
  "${root}/scripts/eval_standard_tum_7scenes.py" \
  --dataset "${dataset}" --dataset-root "${data_root}" \
  --checkpoint "${root}/ckpts/model.pt" --output-dir "${out}" \
  --device cuda:0 --num-frames 300 --timing-repeats 3 --image-resolution 518 \
  > "${out}/logs/${dataset}_gpu${gpu}.log" 2>&1
