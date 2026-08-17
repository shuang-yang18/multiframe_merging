# Monocular Depth Estimation

See configs in `configs/evaluation/monodepth.yaml`.

```bash
python monodepth/infer.py
python monodepth/eval.py
```

## Infer to Generate .npy & .png Depth Files

`infer.py` will generate folders like:

```
recons-eval
├── ...
├── outputs
|   └── monodepth
|       ├── bonn
|       |   ├── sequence_1
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

Then the monodepth metrics will be generated in `outputs/monodepth/{dataset_name}-metric.csv`.

```
recons-eval
├── ...
├── outputs
|   └── monodepth
|       ├── bonn
|       |   ├── sequence_1
|       |   |   ├── xxxxxx.npy
|       |   |   ├── xxxxxx.png
|       |   |   └── ...
|       |   ├── sequence_2
|       |   └── ...
|       ├── hydra (runtime configs)
|       ├── kitti
|       ├── nyu-v2
|       ├── sintel
|       ├── bonn-metric.csv
|       ├── kitti-metric.csv
|       ├── nyu-v2-metric.csv
|       └── sintel-metric.csv
└── ...
```