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
"""Colormap helpers and georeferencing utilities for the FastSlide XYZ viewer."""

from __future__ import annotations

import functools
import io
from typing import Any

import numpy as np
from PIL import Image

#: Default colormap applied when serving a heatmap (a standard "heatmap" look).
DEFAULT_COLORMAP = "jet"


@functools.lru_cache(maxsize=32)
def _colormap_lut(cmap_name: str) -> np.ndarray:
    """Return a ``(256, 3)`` uint8 RGB lookup table for a matplotlib colormap."""
    from matplotlib import colormaps

    try:
        cmap = colormaps[cmap_name]
    except KeyError:
        cmap = colormaps[DEFAULT_COLORMAP]
    rgba = cmap(np.linspace(0.0, 1.0, 256))
    return (rgba[:, :3] * 255.0).round().astype(np.uint8)


@functools.lru_cache(maxsize=32)
def colorbar_png(
    cmap_name: str = DEFAULT_COLORMAP,
    width: int = 160,
    height: int = 24,
) -> bytes:
    """Return a horizontal colorbar PNG for a matplotlib colormap.

    Parameters
    ----------
    cmap_name : str, optional
        Matplotlib colormap name.
    width, height : int, optional
        Maximum bounding box of the colorbar strip in pixels.

    Returns
    -------
    bytes
        Encoded PNG bytes for the colorbar image.
    """
    bar_w = max(32, min(int(width), 512))
    bar_h = max(4, min(int(height), 128))
    lut = _colormap_lut(cmap_name)
    if bar_w == 1:
        indices = np.array([0], dtype=np.uint8)
    else:
        indices = np.linspace(0, 255, bar_w, dtype=np.uint8)
    strip = lut[indices]
    rgb = np.repeat(strip[np.newaxis, :, :], bar_h, axis=0)
    buffer = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def colormap_lut_bytes(cmap_name: str = DEFAULT_COLORMAP) -> bytes:
    """Return a packed ``256 x 3`` uint8 RGB lookup table.

    Parameters
    ----------
    cmap_name : str, optional
        Matplotlib colormap name.

    Returns
    -------
    bytes
        Raw LUT bytes consumed by the viewer for client-side tile coloring.
    """
    return _colormap_lut(cmap_name).tobytes()


def meta_title(meta: dict[str, Any]) -> str | None:
    """Return the optional human-readable label stored in heatmap metadata.

    Parameters
    ----------
    meta : dict
        Heatmap metadata dictionary.

    Returns
    -------
    str or None
        Stripped title string when present, otherwise ``None``.
    """
    title = meta.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    return None


def map_extent(meta: dict[str, Any]) -> list[float]:
    """Return the OpenLayers image extent for a heatmap in slide map coords.

    The viewer map uses level-0 pixels with ``y`` pointing down, so map ``y`` is
    the negated slide pixel ``y``.

    Parameters
    ----------
    meta : dict
        Heatmap metadata with keys ``x_offset``, ``y_offset``, ``size_per_pixel``,
        ``width``, and ``height``.

    Returns
    -------
    list of float
        Extent as ``[minX, minY, maxX, maxY]`` in slide map coordinates.
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
