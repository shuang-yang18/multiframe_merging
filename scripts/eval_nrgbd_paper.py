#!/usr/bin/env python3
"""VGGT NRGBD standard evaluation entry point."""
from __future__ import annotations
import sys
from eval_standard_tum_7scenes import main

if __name__ == "__main__":
    if "--dataset" not in sys.argv:
        sys.argv.extend(["--dataset", "nrgbd"])
    main()
