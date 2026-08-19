#!/usr/bin/env python3
"""Pi3 ScanNet evaluation with the unified depth, pose, geometry, VRAM and latency metrics."""
from __future__ import annotations
import sys
from pathlib import Path
from run_pi3_vggt_omega_eval import main

DEFAULT_ROOT = Path("/data/mmc_syang/dataset/scannet30/raw")

if __name__ == "__main__":
    if "--dataset" not in sys.argv:
        sys.argv.extend(["--dataset", "scannet"])
    if "--dataset-root" not in sys.argv:
        sys.argv.extend(["--dataset-root", str(DEFAULT_ROOT)])
    raise SystemExit(main())
