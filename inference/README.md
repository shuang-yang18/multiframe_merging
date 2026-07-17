# Video Depth Evaluation

Generate VGGT-Omega depth predictions, then evaluate them:

```bash
pip install -e ".[evaluation]"
python video_depth/infer.py --checkpoint checkpoints/vggt_omega_1b_512.pt
python video_depth/eval.py
python video_depth/infer.py --dataset bonn --checkpoint checkpoints/vggt_omega_1b_512.pt
python video_depth/eval.py --dataset bonn
```

`infer.py` supports `--dataset all --all-scenes`, `--window-size 0` for
joint full-sequence inference, and `--input-mode max_size` for lower memory
use. Prediction arrays, previews, timing files, and metric CSVs are written
below `outputs/video_depth/`.
