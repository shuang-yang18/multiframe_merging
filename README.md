# multiframe_merging

This repository contains an accelerated VGGT-Omega variant for long-sequence
4D reconstruction.  The main idea is to reduce redundant computation inside the
VGGT-Omega aggregator with two complementary mechanisms:

- **Multi-frame merging**: merge visually redundant frames before global
  inter-frame attention, keep an inverse map, and restore the full frame count
  at a configurable layer.
- **FastVGGT-style spatial token merging**: at each global inter-frame attention
  layer, merge patch tokens and unmerge them after attention.

The implementation keeps the original VGGT-Omega model interface and adds
runtime switches for frame merging, global-cluster frame grouping, token merging
statistics, and frame-group export.

## Repository Layout

```text
vggt_omega/models/aggregator.py        # frame merging and aggregator logic
vggt_omega/models/layers/attention.py  # FastVGGT-style token merge/unmerge
inference/infer.py                     # TUM / 7scenes / Sintel / Bonn evaluation CLI
scripts/run_multiframe_merging_eval.sh # main reproducible evaluation entry
scripts/run_global_cluster_sweep.sh    # global-cluster frame-merging ablation
scripts/export_frame_merge_groups.py   # export merged frame groups to JSON/CSV
```

Generated outputs, checkpoints, and experiment logs are ignored by git.

## Installation

```bash
git clone https://github.com/shuang-yang18/multiframe_merging.git
cd multiframe_merging
pip install -r requirements.txt
pip install -e .
```

Download the VGGT-Omega checkpoint and place it at:

```text
checkpoints/vggt_omega_1b_512.pt
```

You can also pass another checkpoint with `CHECKPOINT=/path/to/model.pt`.

## Datasets

The evaluation scripts expect these default dataset locations:

```text
/data/mmc_syang/dataset/TUM-Dynamics
/data/mmc_syang/dataset/7scenes/test
```

For 7Scenes, the code uses only the prepared test split directory.  Override it
with:

```bash
SEVEN_SCENES_ROOT=/path/to/7scenes/test
```

All reported long-sequence experiments use `MAX_FRAMES=300`.

## Main Method

Run multi-frame merging plus FastVGGT-style spatial token merging:

```bash
GPU=0 PYTHON=/path/to/python \
RESTORE_LAYER=24 \
PAIR_THRESHOLD=0.98 \
SPAN_THRESHOLD=0.95 \
MAX_GROUP_SIZE=4 \
scripts/run_multiframe_merging_eval.sh \
  tum_dynamic \
  tum300_multiframe_max4_pair098_span095_restore24
```

For 7Scenes:

```bash
GPU=0 PYTHON=/path/to/python \
RESTORE_LAYER=24 \
PAIR_THRESHOLD=0.98 \
SPAN_THRESHOLD=0.95 \
MAX_GROUP_SIZE=4 \
scripts/run_multiframe_merging_eval.sh \
  7scenes \
  7scenes_test300_multiframe_max4_pair098_span095_restore24
```

Useful environment variables:

```text
GPU                  CUDA device id
PYTHON               Python executable
CHECKPOINT           VGGT-Omega checkpoint path
MAX_FRAMES           frames per sequence, default 300
TOKEN_MERGING_RATIO  FastVGGT merge-away ratio, default 0.9
RESTORE_LAYER        layer where merged frames are restored
PAIR_THRESHOLD       adjacent-frame similarity threshold for multi-frame groups
SPAN_THRESHOLD       first-last similarity threshold for multi-frame groups
MAX_GROUP_SIZE       maximum frames per multi-frame group
MAX_WINDOW           max segment length, default 20
POOL_STRIDE          pooling stride for frame-similarity descriptors
```

## Global Cluster Ablation

The global-cluster ablation ignores temporal order and clusters visually similar
frames subject to a minimum similarity threshold and maximum cluster size:

```bash
GPU=0 PYTHON=/path/to/python scripts/run_global_cluster_sweep.sh
```

By default this runs three TUM settings:

```text
threshold=0.98, max_group_size=4
threshold=0.95, max_group_size=3
threshold=0.98, max_group_size=3
```

Run a single setting:

```bash
GPU=0 scripts/run_global_cluster_sweep.sh 0.98 3
```

Each run exports the concrete frame clusters to:

```text
outputs/<run>/<dataset>/_frame_merge_groups.json
outputs/<run>/<dataset>/_frame_merge_groups.csv
```

The CSV contains rows such as:

```text
sequence,event,block,strategy,batch,group,size,frames
rgbd_dataset_freiburg3_walking_xyz,0,0,global_cluster,0,12,3,"4 19 82"
```

## Important CLI Options

The core implementation is exposed through `inference/infer.py`:

```bash
python inference/infer.py \
  --dataset tum_dynamic \
  --output-dir outputs/example \
  --max-frames-per-seq 300 \
  --window-size 0 \
  --checkpoint checkpoints/vggt_omega_1b_512.pt \
  --eval \
  --enable-token-merging \
  --token-merging-method frame_persistent_spatial \
  --token-merging-ratio 0.9 \
  --token-merging-start 0 \
  --token-merging-frame-restore-layer 24 \
  --token-merging-frame-group-strategy local \
  --token-merging-frame-multi-max-group-size 4 \
  --token-merging-frame-multi-pair-threshold 0.98 \
  --token-merging-frame-multi-span-threshold 0.95
```

Frame grouping strategies:

```text
local            streaming segmentation + local adjacent multi-frame groups
segment_middle   merge the middle frames inside each segment
global_cluster   cluster similar frames without temporal-order constraints
```

Token merging methods of interest:

```text
spatial                   FastVGGT-style spatial token merging only
frame_persistent          persistent frame merging only
frame_persistent_spatial  persistent frame merging + FastVGGT spatial merging
```

## Metrics

Evaluation writes summary files under each output directory:

```text
_summary_scale_shift.json
_summary_pose_auc.json
_summary_complete_scale_shift.json
```

The main metrics used in our experiments are:

```text
Abs Rel
delta < 1.25
AUC@3
AUC@30
FPS
```

Per-sequence timing files also include frame-merge statistics:

```text
frame_merge_active_frames_mean
frame_merge_merge_ratio_mean
frame_merge_stats
token_merging_full_attention_token_ratio_mean
```

When group export is enabled, `merge_groups` records exactly which original
frame indices were fused into each active frame.

## Notes

- Checkpoints, datasets, and generated outputs are not included in this repo.
- The code is based on VGGT-Omega and keeps the original license file.
- `outputs/`, `checkpoints/`, and model weight files are ignored by `.gitignore`.
