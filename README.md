# multiframe_merging

This repository contains an accelerated VGGT-Omega variant for long-sequence
4D reconstruction. The primary, reproducible interface is the legacy
`frame_persistent_spatial` path: persistent multi-frame fusion combined with
FastVGGT-style spatial token merging inside global attention.

- **Multi-frame merging**: build local groups from frame similarity, merge each
  group into an active-frame representation, retain an inverse map, and restore
  the full frame count at a configurable layer.
- **FastVGGT-style spatial token merging**: at each global inter-frame attention
  layer, merge patch tokens and unmerge them after attention.

The implementation keeps the original VGGT-Omega model interface and records
frame/token retention statistics and concrete merged frame groups.

## Repository Layout

```text
vggt_omega/models/aggregator.py        # frame merging and aggregator logic
vggt_omega/models/layers/attention.py  # FastVGGT-style token merge/unmerge
inference/infer.py                     # TUM / 7scenes / Sintel / Bonn evaluation CLI
scripts/run_multiframe_merging_eval.sh # main reproducible evaluation entry
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

You can also pass another checkpoint with `CHECKPOINT=checkpoints/your_model.pt`.

## Datasets

The evaluation scripts use repository-relative dataset defaults:

```text
datasets/TUM-Dynamics
datasets/7scenes/test
```

For 7Scenes, the code uses only the prepared test split directory.  Override it
with:

```bash
SEVEN_SCENES_ROOT=datasets/7scenes/test
```

All reported long-sequence experiments use `MAX_FRAMES=300`.

Pose AUC follows the official VGGT relative-pose protocol: convert camera poses
to world-to-camera SE(3), evaluate all frame pairs, use the maximum of rotation
and translation-direction errors, and compute histogram cumulative AUC. For
long-sequence experiments, `--pose-eval-frames 0` keeps all inferred frames. To
run the paper's 10-view setting, sample 10 input frames before inference.

## Main Method

The default command reproduces the `Ours: layerwise p=.986/s=.948` setting:

- `frame_persistent_spatial` persistent multi-frame fusion;
- pair/span thresholds `0.986 / 0.948`, maximum group size `4`;
- restore all frames only after block `24`;
- FastVGGT merge-away ratio `r=0.9` on 0-based global blocks `0-9` and
  `18-23`; blocks `10-17` do not apply token merging.

Run TUM:

```bash
GPU=0 \
scripts/run_multiframe_merging_eval.sh \
  tum_dynamic \
  tum300_ours_layerwise_p0986_s0948
```

Run the prepared 7Scenes test split:

```bash
GPU=0 \
scripts/run_multiframe_merging_eval.sh \
  7scenes \
  7scenes_test300_ours_layerwise_p0986_s0948
```

The script defaults to the settings above. To run an ablation, override only
the relevant variables, for example:

```bash
GPU=0 PAIR_THRESHOLD=0.990 SPAN_THRESHOLD=0.960 \
TOKEN_MERGING_RATIO=0.8 \
scripts/run_multiframe_merging_eval.sh tum_dynamic tum300_ablation
```

Useful environment variables:

```text
GPU                  CUDA device id
PYTHON               Python executable
CHECKPOINT           VGGT-Omega checkpoint path
MAX_FRAMES           frames per sequence, default 300
TOKEN_MERGING_RATIO  FastVGGT merge-away ratio, default 0.9
TOKEN_MERGING_LAYER_RATIOS  1-based FastVGGT ratio schedule, default 1-10:0.9,11-18:0.0,19-24:0.9
RESTORE_LAYER        layer where merged frames are restored
PAIR_THRESHOLD       adjacent-frame similarity threshold, default 0.986
SPAN_THRESHOLD       first-last similarity threshold, default 0.948
MAX_GROUP_SIZE       maximum frames per multi-frame group
MAX_WINDOW           max segment length, default 20
POOL_STRIDE          pooling stride for frame-similarity descriptors
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
  --token-merging-layer-ratios '1-10:0.9,11-18:0.0,19-24:0.9' \
  --token-merging-frame-multi-pair-threshold 0.986 \
  --token-merging-frame-multi-span-threshold 0.948
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
