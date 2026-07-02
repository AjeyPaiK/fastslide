# Copyright 2026 Jonas Teuwen. All Rights Reserved.
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
"""FastAPI XYZ tile server for a single FastSlide whole-slide image.

This is the FastSlide analogue of the OpenSlide ``deepzoom_server.py`` example,
but it serves a standard XYZ (slippy-map) tile pyramid that an OpenLayers
viewer can consume directly, and uses FastAPI/uvicorn instead of Flask.

Usage:
    python server.py /path/to/slide.svs
    python server.py /path/to/slide.svs --host 0.0.0.0 --port 8000 --tile-size 256

Then open http://127.0.0.1:8000 in a browser.

Endpoints:
    GET /                                  -> the OpenLayers viewer (index.html)
    GET /info                              -> slide + per-image metadata as JSON
    GET /annotations                       -> GeoJSON annotations (if configured)
    GET /heatmaps                          -> heatmap overlays (if configured)
    GET /heatmap-tiles/{name}/{z}/{x}/{y}.png -> one heatmap tile (SQLite pyramid)
    GET /heatmap-lut                       -> colormap LUT bytes
    GET /tiles/{image}/{z}/{x}/{y}.{ext}   -> a single tile (ext: jpg | jpeg | png)

Annotations are optional GeoJSON whose coordinates are in level-0 slide pixels
(origin top-left, y pointing down). ``--annotations`` may point to a single file
or to a folder; when it is a folder, *every* ``*.json`` / ``*.geojson`` inside
is served as a separate, individually toggleable layer. ``--heatmaps-db`` points
to a ``heatmaps.sqlite`` file or a directory of per-slide databases.
"""

from __future__ import annotations

import argparse
import functools
import json
import sqlite3
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response

import fastslide
from fastslide.xyz_pyramid import XYZPyramid

import heatmap as heatmap_lib
from heatmap_db import HeatmapDatabase
import geojson_labels

_HERE = Path(__file__).resolve().parent
_INDEX_HTML = _HERE / "index.html"

_MEDIA_TYPES = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
}

_ANNOTATION_EXTENSIONS = (".json", ".geojson")


def create_app(
    slide_path: str | Path,
    tile_size: int = 256,
    jpeg_quality: int = 85,
    annotations_path: str | Path | None = None,
    heatmaps_db: str | Path | None = None,
) -> FastAPI:
    """Builds the FastAPI app serving XYZ tiles for a single slide.

    Args:
        slide_path: Path to the whole-slide image to serve.
        tile_size: Tile edge length in pixels.
        jpeg_quality: Quality used when encoding JPEG tiles.
        annotations_path: Optional path to a GeoJSON file (or folder) with
            annotations in level-0 slide pixel coordinates.
        heatmaps_db: Optional path to a ``heatmaps.sqlite`` file or a directory
            of per-slide heatmap databases.

    Returns:
        A configured :class:`fastapi.FastAPI` instance.
    """
    slide = fastslide.FastSlide.from_file_path(str(slide_path))
    annotations_file = Path(annotations_path).resolve() if annotations_path else None

    heatmaps_db_path = Path(heatmaps_db).resolve() if heatmaps_db else None
    heatmaps_db_file: Path | None = None
    heatmaps_db_root: Path | None = None
    if heatmaps_db_path is not None:
        if heatmaps_db_path.is_dir():
            heatmaps_db_root = heatmaps_db_path
        elif heatmaps_db_path.is_file():
            heatmaps_db_file = heatmaps_db_path
        else:
            raise FileNotFoundError(f"heatmap database not found: {heatmaps_db_path}")

    slide_name = Path(slide_path).name

    # One XYZ pyramid per navigable image in the file. Most slides expose a
    # single image; some (e.g. Olympus VSI) expose a navigator plus one or more
    # region images.
    images = slide.images
    pyramids = [XYZPyramid(images[i], tile_size=tile_size) for i in range(len(images))]

    def heatmap_db() -> HeatmapDatabase | None:
        if heatmaps_db_file is not None:
            return HeatmapDatabase.for_path(heatmaps_db_file)
        if heatmaps_db_root is not None:
            db_path = HeatmapDatabase.resolve_path(heatmaps_db_root, slide_name)
            if db_path is not None:
                return HeatmapDatabase.for_path(db_path)
        return None

    def slide_pyramid_info() -> tuple[int, int, list[float]] | None:
        primary = pyramids[images.primary_index]
        width, height = primary.level0_dimensions
        return width, height, primary.resolutions

    app = FastAPI(title="FastSlide XYZ Viewer")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(_INDEX_HTML, media_type="text/html")

    @app.get("/info")
    def info() -> JSONResponse:
        image_infos = []
        for i, pyramid in enumerate(pyramids):
            entry = pyramid.info()
            entry["index"] = i
            entry["name"] = images[i].name
            image_infos.append(entry)
        return JSONResponse(
            {
                "name": Path(slide_path).name,
                "format": slide.format,
                "primary_index": images.primary_index,
                "images": image_infos,
            }
        )

    def _annotation_titles() -> dict[str, str]:
        """Maps annotation keys to GeoJSON display titles."""
        if annotations_file is None:
            return {}
        files: list[Path] = []
        if annotations_file.is_dir():
            for ext in _ANNOTATION_EXTENSIONS:
                files.extend(p for p in annotations_file.rglob(f"*{ext}") if p.is_file())
            files.sort()
        elif annotations_file.is_file():
            files = [annotations_file]
        titles: dict[str, str] = {}
        for path in files:
            try:
                geojson = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            label = geojson_labels.geojson_title(geojson)
            if not label:
                continue
            if annotations_file.is_dir():
                key = path.relative_to(annotations_file).with_suffix("").as_posix()
            else:
                key = path.stem
            titles[key] = label
        return titles

    def _heatmap_display_title(heatmap_key: str, ann_titles: dict[str, str]) -> str | None:
        if heatmap_key in ann_titles:
            return ann_titles[heatmap_key]
        stem = Path(heatmap_key).name
        if stem in ann_titles:
            return ann_titles[stem]
        return None

    @app.get("/annotations")
    def annotations() -> JSONResponse:
        files: list[Path] = []
        if annotations_file is not None:
            if annotations_file.is_dir():
                for ext in _ANNOTATION_EXTENSIONS:
                    files.extend(p for p in annotations_file.rglob(f"*{ext}") if p.is_file())
                files.sort()
            elif annotations_file.is_file():
                files = [annotations_file]
        items: list[dict[str, object]] = []
        for path in files:
            try:
                geojson = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            items.append({"name": path.stem, "geojson": geojson})
        return JSONResponse({"annotations": items})

    @app.get("/heatmaps")
    def heatmaps() -> JSONResponse:
        db = heatmap_db()
        if db is None:
            return JSONResponse({"heatmaps": []})

        ann_titles = _annotation_titles()
        items: list[dict[str, object]] = []
        records = db.try_list_heatmaps()
        if records is None:
            return JSONResponse({"heatmaps": items, "pending": True})

        pyramid = slide_pyramid_info()
        for record in records:
            if pyramid is not None:
                sw, sh, resolutions = pyramid
                if not record.meta.get("resolutions"):
                    db.configure_pyramid(
                        record.id,
                        slide_width=sw,
                        slide_height=sh,
                        tile_size=tile_size,
                        resolutions=resolutions,
                    )
                    refreshed = db.get_by_name(record.name)
                    if refreshed is not None:
                        record = refreshed
            title = (
                heatmap_lib.meta_title(record.meta)
                or _heatmap_display_title(record.name, ann_titles)
                or record.name
            )
            entry = HeatmapDatabase.heatmap_list_entry(record)
            entry["title"] = title
            items.append(entry)
        return JSONResponse({"heatmaps": items})

    @app.get("/heatmap-tiles/{name}/{z}/{x}/{y}.png")
    def heatmap_tile_png(name: str, z: int, x: int, y: int) -> Response:
        db = heatmap_db()
        if db is None:
            raise HTTPException(status_code=404, detail="heatmap database not found")
        record = db.get_by_name(name)
        if record is None:
            raise HTTPException(status_code=404, detail="heatmap not found")
        pyramid = slide_pyramid_info()
        kwargs: dict[str, object] = {}
        if pyramid is not None:
            sw, sh, resolutions = pyramid
            kwargs = {
                "slide_width": sw,
                "slide_height": sh,
                "tile_size": tile_size,
                "resolutions": resolutions,
            }
            if not record.meta.get("resolutions"):
                db.configure_pyramid(
                    record.id,
                    slide_width=sw,
                    slide_height=sh,
                    tile_size=tile_size,
                    resolutions=resolutions,
                )
        try:
            data = db.get_tile_png(record.id, z, x, y, **kwargs)
        except sqlite3.OperationalError as exc:
            if HeatmapDatabase._is_locked_error(exc):
                raise HTTPException(
                    status_code=503,
                    detail="heatmap database temporarily locked",
                ) from exc
            raise
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return Response(content=data, media_type="image/png")

    @app.get("/heatmap-lut")
    def heatmap_lut(cmap: str = heatmap_lib.DEFAULT_COLORMAP) -> Response:
        try:
            data = heatmap_lib.colormap_lut_bytes(cmap)
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return Response(content=data, media_type="application/octet-stream")

    @app.get("/heatmap-colorbar.png")
    def heatmap_colorbar(
        cmap: str = heatmap_lib.DEFAULT_COLORMAP,
        width: int = 160,
        height: int = 24,
    ) -> Response:
        try:
            data = heatmap_lib.colorbar_png(cmap, width=width, height=height)
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return Response(content=data, media_type="image/png")

    @app.get("/tiles/{image}/{z}/{x}/{y}.{ext}")
    def tile(image: int, z: int, x: int, y: int, ext: str) -> Response:
        media_type = _MEDIA_TYPES.get(ext.lower())
        if media_type is None:
            raise HTTPException(status_code=404, detail=f"unsupported extension: {ext}")
        if not 0 <= image < len(pyramids):
            raise HTTPException(status_code=404, detail=f"no image {image}")
        try:
            data = pyramids[image].get_tile_bytes(z, x, y, fmt=ext, quality=jpeg_quality)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return Response(content=data, media_type=media_type)

    return app


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a slide as an XYZ tile pyramid.")
    parser.add_argument("slide", help="Path to the whole-slide image file.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000).")
    parser.add_argument("--tile-size", type=int, default=256, help="Tile size in px (default: 256).")
    parser.add_argument("--jpeg-quality", type=int, default=85, help="JPEG quality 1-100 (default: 85).")
    parser.add_argument(
        "--annotations",
        default=None,
        help="GeoJSON file, or a folder of GeoJSON files, with level-0 annotations.",
    )
    parser.add_argument(
        "--heatmaps-db",
        default=None,
        help="heatmaps.sqlite file or directory of per-slide heatmap databases.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = _parse_args()
    app = create_app(
        args.slide,
        tile_size=args.tile_size,
        jpeg_quality=args.jpeg_quality,
        annotations_path=args.annotations,
        heatmaps_db=args.heatmaps_db,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
