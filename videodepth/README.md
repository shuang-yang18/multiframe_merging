# Video Depth Estimation

See configs in `configs/evaluation/videodepth.yaml`.

```bash
python videodepth/infer.py
python videodepth/eval.py              # align=scale&shift by default
python videodepth/eval.py align=scale  # override with align=scale
```

## Infer to Generate .npy & .png Depth Files

`infer.py` will generate folders like:

```
recons-eval
├── ...
├── outputs
|   └── videodepth
|       ├── bonn
|       |   ├── sequence_1
|       |   |   ├── _time.json
|       |   |   ├── xxxxxx.npy
|       |   |   ├── xxxxxx.png
|       |   |   └── ...
|       |   ├── sequence_2
|       |   └── ...
|       ├── hydra (runtime configs)
|       ├── kitti
|       ├── nyu-v2
|       └── sintel
└── ...
```

## Eval with Generated Depth Files

After `infer.py` finishes, you can run `eval.py` to evaluate the results.

Then the videodepth metrics will be generated in `outputs/videodepth/{dataset_name}-metric-{align}.csv`. For example, if `align=scale`, then

```
recons-eval
├── ...
├── outputs
|   └── videodepth
|       ├── bonn
|       |   ├── sequence_1
|       |   |   ├── _time.json
|       |   |   ├── xxxxxx.npy
|       |   |   ├── xxxxxx.png
|       |   |   └── ...
|       |   ├── sequence_2
|       |   └── ...
|       ├── hydra (runtime configs)
|       ├── kitti
|       ├── nyu-v2
|       ├── sintel
|       ├── bonn-metric-scale.csv
|       ├── kitti-metric-scale.csv
|       └── sintel-metric-scale.csv
└── ...
```