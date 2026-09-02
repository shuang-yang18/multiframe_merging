"""VGGT entry points for the shared Omega-compatible unified U-M core.

The core is shared with Pi3 so the two adapter implementations cannot drift:
both use the same protected-frame cube graph, exact whole-group merge rule,
pre-QKV group averaging, and Triton fused edge-cost fallback.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_PI3_ROOT_CANDIDATES = (
    os.environ.get("PI3_ROOT"),
    "/data/mmc_syang/Pi3",
    Path(__file__).resolve().parents[3] / "Pi3",
)
for _pi3_root in _PI3_ROOT_CANDIDATES:
    if _pi3_root is not None and Path(_pi3_root).is_dir():
        _pi3_root = str(_pi3_root)
        if _pi3_root not in sys.path:
            sys.path.insert(0, _pi3_root)
        break

from pi3.models.um import UMPlan, build_um_plan as _build_um_plan, um_attention as _um_attention


def build_um_plan(
    tokens, *, num_frames: int, patch_start: int, grid_size: tuple[int, int],
    spatial_radius: int = 2, temporal_window: int = 4, lambda_cost: float = 0.04,
) -> UMPlan:
    """Build the unified U-M plan from raw VGGT global tokens."""
    return _build_um_plan(
        tokens, num_frames, patch_start, grid_size,
        spatial_radius, temporal_window, lambda_cost,
    )


def um_attention(attention, tokens, pos, plan: UMPlan, *, norm1):
    """Run mean-token U-M attention and restore its full residual."""
    return _um_attention(attention, tokens, plan, norm1=norm1, xpos=pos)


__all__ = ["UMPlan", "build_um_plan", "um_attention"]
