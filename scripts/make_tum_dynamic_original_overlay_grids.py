#!/usr/bin/env python3
"""Create paired original/dynamic-overlay grids from saved TUM segmentation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(
            "new_results/1/layer4_dynamic_segmentation_crf_tum300_allframes_pca128_k03_equalframe_attention/tum_dynamic"
        ),
    )
    parser.add_argument("--variant", default="pca128_k03")
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--pairs-per-row", type=int, default=4)
    parser.add_argument("--cell-width", type=int, default=320)
    return parser.parse_args()


def contain(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    rendered = image.convert("RGB")
    rendered.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (14, 14, 14))
    offset = ((size[0] - rendered.width) // 2, (size[1] - rendered.height) // 2)
    canvas.paste(rendered, offset)
    return canvas


def overlay_for(dynamic_dir: Path, source_index: int) -> Path:
    matches = sorted(dynamic_dir.glob(f"*_source_{source_index:04d}_overlay.png"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one dynamic overlay for source index {source_index}: {dynamic_dir}")
    return matches[0]


def build_grid(sequence_dir: Path, variant: str, frames: int, pairs_per_row: int, cell_width: int) -> Path:
    result_dir = sequence_dir / variant
    config = json.loads((result_dir / "config.json").read_text())
    source_images = [Path(path) for path in config["selected_images"]]
    source_indices = np.asarray(config["selected_indices"], dtype=np.int64)
    if len(source_images) != len(source_indices):
        raise ValueError(f"{sequence_dir.name}: config image/index lengths differ")
    if frames < 1 or frames > len(source_images):
        raise ValueError(f"{sequence_dir.name}: --frames must be in [1, {len(source_images)}]")

    chosen_positions = np.linspace(0, len(source_images) - 1, frames).round().astype(int)
    cell_height = round(cell_width * 0.75)
    title_height = 26
    pair_width = cell_width * 2
    rows = (frames + pairs_per_row - 1) // pairs_per_row
    grid = Image.new("RGB", (pairs_per_row * pair_width, rows * (cell_height + title_height)), (12, 12, 12))
    draw = ImageDraw.Draw(grid)
    dynamic_dir = result_dir / "camera_attention_global_dynamic"

    for pair_index, position in enumerate(chosen_positions):
        source_index = int(source_indices[position])
        original = contain(Image.open(source_images[position]), (cell_width, cell_height))
        dynamic = contain(Image.open(overlay_for(dynamic_dir, source_index)), (cell_width, cell_height))
        row, col = divmod(pair_index, pairs_per_row)
        x = col * pair_width
        y = row * (cell_height + title_height)
        grid.paste(original, (x, y + title_height))
        grid.paste(dynamic, (x + cell_width, y + title_height))
        draw.text((x + 5, y + 5), f"frame {source_index:03d}: original | dynamic segmentation", fill=(240, 240, 240))

    output_dir = result_dir / "pair_grids"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"original_dynamic_pairs_{frames:02d}frames.png"
    grid.save(output_path, quality=95)
    return output_path


def main() -> None:
    args = parse_args()
    for sequence_dir in sorted(path for path in args.results_root.iterdir() if path.is_dir()):
        output = build_grid(sequence_dir, args.variant, args.frames, args.pairs_per_row, args.cell_width)
        print(output)


if __name__ == "__main__":
    main()
