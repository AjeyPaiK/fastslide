# Copyright 2026 Ajey Pai Karkala. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Heatmap overlays for the FastSlide XYZ viewer.

A heatmap is a low-resolution intensity grid that overlays a region of a slide.
Each heatmap cell covers ``size_per_pixel`` x ``size_per_pixel`` level-0 slide
pixels, with the grid's top-left corner at ``(x_offset, y_offset)`` (also in
level-0 pixels).

Storage format (efficient, O(1) to serve)
-----------------------------------------
A heatmap is stored as two files that share a base name:

- ``<name>.png``  -- a greyscale-with-alpha (``LA``) image, one pixel per cell.
  The grey channel is the intensity normalised to ``0..255``; the alpha channel
  is ``0`` for empty cells (intensity ``0``) and ``255`` otherwise, so the slide
  shows through the background.
- ``<name>.json`` -- georeferencing metadata::

      {"x_offset": 0, "y_offset": 0, "size_per_pixel": 16,
       "width": 5792, "height": 12638, "max_value": 92.0}

The viewer draws the PNG as a single georeferenced image, so rendering cost is
independent of the number of cells.

Legacy TSV input
----------------
The original generator emitted a tab-separated text file::

    Heatmap <x_offset> <y_offset> <size_per_pixel>
    x1  y1  value1
    x2  y2  value2
    ...

These can be huge (one line per cell). :func:`convert_tsv` turns such a file
into the PNG + JSON pair once; afterwards the server only touches the PNG.
"""

from __future__ import annotations

import functools
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

#: Source extensions a heatmap can be discovered from, most efficient first.
HEATMAP_SOURCE_EXTENSIONS = (".png", ".tsv")

#: Default colormap applied when serving a heatmap (a standard "heatmap" look).
DEFAULT_COLORMAP = "jet"


@functools.lru_cache(maxsize=32)
def _colormap_lut(cmap_name: str) -> np.ndarray:
    """Returns a ``(256, 3)`` uint8 RGB lookup table for a matplotlib colormap.

    Falls back to :data:`DEFAULT_COLORMAP` for an unknown name.
    """
    from matplotlib import colormaps

    try:
        cmap = colormaps[cmap_name]
    except KeyError:
        cmap = colormaps[DEFAULT_COLORMAP]
    rgba = cmap(np.linspace(0.0, 1.0, 256))  # (256, 4) floats in 0..1
    return (rgba[:, :3] * 255.0).round().astype(np.uint8)


@functools.lru_cache(maxsize=8)
def colorize_png(png_path_str: str, cmap_name: str = DEFAULT_COLORMAP) -> bytes:
    """Colorizes a stored intensity heatmap PNG with a standard colormap.

    The stored ``<name>.png`` keeps raw intensity (greyscale ``LA``); this maps
    the grey channel through ``cmap_name`` to RGB while preserving the alpha
    channel (so empty cells stay transparent). Returns encoded PNG bytes; the
    result is cached so repeated requests for the same colormap are O(1).
    """
    image = Image.open(png_path_str).convert("LA")
    arr = np.asarray(image)  # (H, W, 2): [grey, alpha]
    grey = arr[..., 0]
    alpha = arr[..., 1]
    rgb = _colormap_lut(cmap_name)[grey]  # (H, W, 3)
    rgba = np.dstack([rgb, alpha])
    buffer = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def meta_path_for(png_path: Path) -> Path:
    """Returns the JSON sidecar path for a rendered heatmap PNG."""
    return png_path.with_suffix(".json")


def _read_tsv_header(tsv_path: Path) -> tuple[int, int, float]:
    """Reads ``x_offset``, ``y_offset`` and ``size_per_pixel`` from the header."""
    with open(tsv_path, "r", encoding="utf-8") as fh:
        first = fh.readline()
    parts = first.replace(",", " ").split()
    if not parts or parts[0].lower() != "heatmap":
        raise ValueError(f"{tsv_path}: first token of the header must be 'Heatmap'")
    if len(parts) < 4:
        raise ValueError(f"{tsv_path}: header needs 'Heatmap x_offset y_offset size_per_pixel'")
    x_offset = int(float(parts[1]))
    y_offset = int(float(parts[2]))
    size_per_pixel = float(parts[3])
    if size_per_pixel <= 0:
        raise ValueError(f"{tsv_path}: size_per_pixel must be positive")
    return x_offset, y_offset, size_per_pixel


def _grid_from_tsv(tsv_path: Path) -> np.ndarray:
    """Loads the sparse ``x y value`` rows of a TSV into a dense 2-D array.

    Uses pandas' C parser, which reads the (potentially very large) file far
    faster than pure-Python parsing. Missing cells default to ``0``.
    """
    import pandas as pd  # imported lazily; only needed for TSV conversion

    # Tab-separated so the fast C parser is used (a regex separator would force
    # pandas onto its much slower Python engine for these multi-million-row files).
    frame = pd.read_csv(
        tsv_path,
        sep="\t",
        skiprows=1,
        header=None,
        usecols=[0, 1, 2],
        names=["x", "y", "v"],
        dtype={"x": np.int32, "y": np.int32, "v": np.float32},
    )
    width = int(frame["x"].max()) + 1
    height = int(frame["y"].max()) + 1
    grid = np.zeros((height, width), dtype=np.float32)
    grid[frame["y"].to_numpy(), frame["x"].to_numpy()] = frame["v"].to_numpy()
    return grid


def _write_png_and_meta(
    grid: np.ndarray,
    x_offset: int,
    y_offset: int,
    size_per_pixel: float,
    png_path: Path,
) -> dict[str, Any]:
    """Writes ``grid`` as an ``LA`` PNG plus its JSON sidecar; returns the meta."""
    height, width = grid.shape
    max_value = float(grid.max()) if grid.size else 0.0

    grey = np.zeros((height, width), dtype=np.uint8)
    if max_value > 0:
        grey = np.clip(grid / max_value * 255.0, 0, 255).astype(np.uint8)
    # Empty cells (intensity 0) are fully transparent so the slide shows through.
    alpha = np.where(grid > 0, 255, 0).astype(np.uint8)

    la = np.dstack([grey, alpha])
    Image.fromarray(la, mode="LA").save(png_path, format="PNG", optimize=True)

    meta = {
        "x_offset": int(x_offset),
        "y_offset": int(y_offset),
        "size_per_pixel": float(size_per_pixel),
        "width": int(width),
        "height": int(height),
        "max_value": max_value,
    }
    meta_path_for(png_path).write_text(json.dumps(meta))
    return meta


def convert_tsv(tsv_path: Path, png_path: Path | None = None) -> tuple[Path, dict[str, Any]]:
    """Converts a heatmap TSV into the PNG + JSON pair.

    Args:
        tsv_path: Source ``.tsv`` file.
        png_path: Destination PNG (defaults to ``tsv_path`` with a ``.png`` suffix).

    Returns:
        ``(png_path, meta)``.
    """
    tsv_path = Path(tsv_path)
    png_path = Path(png_path) if png_path is not None else tsv_path.with_suffix(".png")
    x_offset, y_offset, size_per_pixel = _read_tsv_header(tsv_path)
    grid = _grid_from_tsv(tsv_path)
    meta = _write_png_and_meta(grid, x_offset, y_offset, size_per_pixel, png_path)
    return png_path, meta


def ensure_rendered(source: Path) -> tuple[Path, dict[str, Any]]:
    """Returns the rendered ``(png_path, meta)`` for a heatmap source.

    If ``source`` is already a PNG with a JSON sidecar, it is used directly. If
    it is a TSV, it is converted once (and cached next to it as ``<name>.png`` /
    ``<name>.json``); a stale cache (older than the TSV) is rebuilt.
    """
    source = Path(source)
    if source.suffix.lower() == ".png":
        meta = json.loads(meta_path_for(source).read_text())
        return source, meta

    if source.suffix.lower() == ".tsv":
        png_path = source.with_suffix(".png")
        meta_path = meta_path_for(png_path)
        fresh = (
            png_path.is_file()
            and meta_path.is_file()
            and png_path.stat().st_mtime >= source.stat().st_mtime
        )
        if fresh:
            return png_path, json.loads(meta_path.read_text())
        return convert_tsv(source, png_path)

    raise ValueError(f"unsupported heatmap source: {source}")


def map_extent(meta: dict[str, Any]) -> list[float]:
    """Returns the OpenLayers image extent for a heatmap in slide map coords.

    The viewer's map uses level-0 pixels with y pointing down (so map y is the
    negated slide pixel y). The extent is ``[minX, minY, maxX, maxY]``.

    TSV ``x`` / ``y`` columns are local cell indices; ``(x_offset, y_offset)``
    is the level-0 slide pixel position of cell ``(0, 0)``.
    """
    x_offset = meta["x_offset"]
    y_offset = meta["y_offset"]
    size = meta["size_per_pixel"]
    width = meta["width"]
    height = meta["height"]
    left = x_offset
    right = x_offset + width * size
    top = -y_offset
    bottom = -(y_offset + height * size)
    return [left, bottom, right, top]


def _main() -> None:
    """CLI: convert heatmap TSV file(s) into PNG + JSON pairs.

    Usage:
        python heatmap.py path/to/heatmap.tsv [more.tsv ...]
        python heatmap.py path/to/dir        # converts every *.tsv under dir
    """
    import argparse

    parser = argparse.ArgumentParser(description="Convert heatmap TSV files to PNG + JSON.")
    parser.add_argument("paths", nargs="+", help="TSV file(s) or directory to convert.")
    parser.add_argument("--force", action="store_true", help="Re-render even if a fresh PNG exists.")
    args = parser.parse_args()

    sources: list[Path] = []
    for raw in args.paths:
        p = Path(raw)
        if p.is_dir():
            sources.extend(sorted(p.rglob("*.tsv")))
        else:
            sources.append(p)

    for tsv in sources:
        png = tsv.with_suffix(".png")
        if not args.force and png.is_file() and png.stat().st_mtime >= tsv.stat().st_mtime:
            print(f"skip (up to date): {png}")
            continue
        print(f"converting {tsv} ...", flush=True)
        out, meta = convert_tsv(tsv, png)
        print(f"  wrote {out} ({meta['width']}x{meta['height']}, max={meta['max_value']})")


if __name__ == "__main__":
    _main()
