# Video Depth Evaluation

Generate VGGT-Omega depth predictions, then evaluate them:

```bash
pip install -e ".[evaluation]"
python inference/infer.py --dataset tum_dynamic --checkpoint checkpoints/vggt_omega_1b_512.pt --eval
python inference/infer.py --dataset bonn --checkpoint checkpoints/vggt_omega_1b_512.pt --eval
```

`infer.py` supports `--dataset all --all-scenes`, `--dataset-root`, `--window-size 0`
for joint full-sequence inference, and `--input-mode max_size` for lower memory
use. Prediction arrays, previews, timing files, and metric CSVs are written below
the selected `--output-dir`.
