#!/usr/bin/env python3
"""Pi3 NRGBD standard evaluation entry point.

Uses the unified Pi3 evaluator with the NRGBD reader, formal CUDA-event timing,
and the same depth, pose, reconstruction and VRAM metrics as TUM/7Scenes.
"""
from __future__ import annotations
import sys
from run_pi3_vggt_omega_eval import main

if __name__ == "__main__":
    if "--dataset" not in sys.argv:
        sys.argv.extend(["--dataset", "nrgbd"])
    raise SystemExit(main())
