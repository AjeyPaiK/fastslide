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
"""FastAPI XYZ tile server for a *folder* of whole-slide images.

This is the multi-slide counterpart to ``server.py`` and the FastSlide
analogue of OpenSlide's ``deepzoom_multiserver.py``. It scans a directory for
slides whose format FastSlide supports, presents a file browser to pick one,
and serves the same OpenLayers XYZ viewer (``index.html``) per slide.

Usage:
    python server_multiimage.py /path/to/slides
    python server_multiimage.py /path/to/slides --host 0.0.0.0 --port 8000

Then open http://127.0.0.1:8000 to browse the folder.

Endpoints:
    GET /                                  -> the file browser (browser.html)
    GET /api/slides                        -> JSON list of discovered slides
    GET /viewer?slide=<relpath>            -> the OpenLayers viewer (index.html)
    GET /info?slide=<relpath>              -> slide + per-image metadata as JSON
    GET /annotations?slide=<relpath>       -> GeoJSON annotations for the slide
    GET /tiles/{image}/{z}/{x}/{y}.{ext}?slide=<relpath>  -> a single tile

Annotations are optional GeoJSON files whose coordinates are given in level-0
slide pixels (origin top-left, y pointing down). Pass ``--annotations-dir`` to
serve them. *All* GeoJSON files belonging to a slide are returned at once, so a
slide can carry several annotation layers that the viewer toggles individually.
For a slide ``sub/4104 T2.mrxs`` the server collects, under the annotations dir:

- every ``*.json`` / ``*.geojson`` inside a per-slide subfolder named after the
  slide stem (``<dir>/4104 T2/`` or ``<dir>/sub/4104 T2/``), and
- flat files whose name is the slide stem or the stem followed by a separator,
  e.g. ``4104 T2.json`` and ``4104 T2_tissue_foreground.json``.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sqlite3
from collections.abc import Callable
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response

import fastslide
from fastslide.xyz_pyramid import XYZPyramid

import heatmap as heatmap_lib
from heatmap_db import HeatmapDatabase
import geojson_labels

_HERE = Path(__file__).resolve().parent
_INDEX_HTML = _HERE / "index.html"
_BROWSER_HTML = _HERE / "browser.html"

_MEDIA_TYPES = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
}


def _supported_extensions() -> tuple[str, ...]:
    """Returns supported extensions, longest first (so ``.ome.tif`` wins)."""
    exts = {e.lower() for e in fastslide.get_supported_extensions()}
    return tuple(sorted(exts, key=len, reverse=True))


def _matches_extension(name: str, extensions: tuple[str, ...]) -> bool:
    return name.lower().endswith(extensions)


def _entry(path: Path, root: Path) -> dict[str, object]:
    # Directory-based formats (e.g. OME-Zarr) report no single file size.
    size = path.stat().st_size if path.is_file() else None
    return {
        "path": path.relative_to(root).as_posix(),
        "name": path.name,
        "size_bytes": size,
        "is_dir": path.is_dir(),
    }


def _list_slides(root: Path, extensions: tuple[str, ...]) -> list[dict[str, object]]:
    """Returns supported slides under ``root`` as relative-path entries.

    Matching is purely by extension against ``get_supported_extensions()``;
    this is cheap and faithful to the advertised formats. Directory-based
    formats (``.zarr``/``.ome.zarr``) are matched as a unit and not descended
    into, which also avoids walking their many chunk files.
    """
    slides: list[dict[str, object]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        keep: list[str] = []
        for name in sorted(dirnames):
            if _matches_extension(name, extensions):
                slides.append(_entry(here / name, root))
            else:
                keep.append(name)
        dirnames[:] = keep  # prune matched directories from the walk
        for name in sorted(filenames):
            if _matches_extension(name, extensions):
                slides.append(_entry(here / name, root))
    slides.sort(key=lambda s: str(s["path"]))
    return slides


_ANNOTATION_EXTENSIONS = (".json", ".geojson")
# Characters that may join a slide stem to an annotation suffix in flat layouts,
# e.g. ``4104 T2_tissue_foreground.json`` -> stem ``4104 T2`` + ``_`` + suffix.
# Requiring a separator avoids matching a different slide whose stem merely
# starts with this one (``4104 T2`` must not match ``4104 T21``).
_STEM_SEPARATORS = ("_", "-", " ", ".", "(")


def _collect_geojson(directory: Path) -> list[Path]:
    """Returns every GeoJSON file under ``directory`` (recursively)."""
    files: list[Path] = []
    for ext in _ANNOTATION_EXTENSIONS:
        files.extend(p for p in directory.rglob(f"*{ext}") if p.is_file())
    return files


def _stem_matches_slide(file_stem: str, slide_stem: str) -> bool:
    """True when a flat overlay filename belongs to ``slide_stem``."""
    return file_stem == slide_stem or (
        file_stem.startswith(slide_stem)
        and file_stem[len(slide_stem) : len(slide_stem) + 1] in _STEM_SEPARATORS
    )


def _overlay_layer_name(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).with_suffix("").as_posix()
    except ValueError:
        return path.stem


def _overlay_summary(paths: list[Path], root: Path) -> dict[str, object]:
    layers = sorted({_overlay_layer_name(root, path) for path in paths})
    return {"count": len(layers), "layers": layers}


def _attach_overlay_metadata(
    slides: list[dict[str, object]],
    *,
    annotations_root: Path | None,
    heatmap_db_path_for_slide: Callable[[str], Path | None],
    find_annotation_files: Callable[[str], list[Path]],
) -> None:
    for slide in slides:
        rel = str(slide["path"])
        overlays: dict[str, dict[str, object]] = {}
        if annotations_root is not None:
            overlays["annotations"] = _overlay_summary(find_annotation_files(rel), annotations_root)
        db_path = heatmap_db_path_for_slide(rel)
        if db_path is not None:
            layers = HeatmapDatabase.try_list_heatmaps_at(db_path)
            if layers is None:
                overlays["heatmaps"] = {"count": 0, "layers": [], "pending": True}
            else:
                names = [r.name for r in layers]
                overlays["heatmaps"] = {"count": len(names), "layers": sorted(names)}
        if overlays:
            slide["overlays"] = overlays


def create_app(
    root: str | Path,
    tile_size: int = 256,
    jpeg_quality: int = 85,
    annotations_dir: str | Path | None = None,
    heatmaps_dir: str | Path | None = None,
    heatmaps_db: str | Path | None = None,
) -> FastAPI:
    """Builds the FastAPI app serving XYZ tiles for every slide under ``root``.

    Args:
        root: Directory to scan for supported slides (searched recursively).
        tile_size: Tile edge length in pixels.
        jpeg_quality: Quality used when encoding JPEG tiles.
        annotations_dir: Optional directory holding per-slide GeoJSON files,
            matched by the slide's file stem (and mirrored sub-path).
        heatmaps_dir: Optional directory of per-slide ``heatmaps.sqlite`` databases,
            matched like annotations (per-slide subfolder or stem-named file).
        heatmaps_db: Optional SQLite database file or directory of per-slide
            ``heatmaps.sqlite`` databases. When both are set, either root may
            supply a database for a slide.

    Returns:
        A configured :class:`fastapi.FastAPI` instance.
    """
    root_dir = Path(root).resolve()
    if not root_dir.is_dir():
        raise NotADirectoryError(f"not a directory: {root_dir}")

    annotations_root = Path(annotations_dir).resolve() if annotations_dir else None
    if annotations_root is not None and not annotations_root.is_dir():
        raise NotADirectoryError(f"not a directory: {annotations_root}")

    heatmaps_dir_path = Path(heatmaps_dir).resolve() if heatmaps_dir else None
    if heatmaps_dir_path is not None and not heatmaps_dir_path.is_dir():
        raise NotADirectoryError(f"not a directory: {heatmaps_dir_path}")

    heatmaps_db_path = Path(heatmaps_db).resolve() if heatmaps_db else None
    heatmaps_db_file: Path | None = None
    heatmaps_search_roots: list[Path] = []
    if heatmaps_db_path is not None:
        if heatmaps_db_path.is_dir():
            heatmaps_search_roots.append(heatmaps_db_path)
        elif heatmaps_db_path.is_file():
            heatmaps_db_file = heatmaps_db_path
        else:
            raise FileNotFoundError(f"heatmap database not found: {heatmaps_db_path}")
    if heatmaps_dir_path is not None and heatmaps_dir_path not in heatmaps_search_roots:
        heatmaps_search_roots.append(heatmaps_dir_path)

    extensions = _supported_extensions()

    def heatmap_db_path_for_slide(rel: str) -> Path | None:
        if heatmaps_db_file is not None:
            return heatmaps_db_file
        for root in heatmaps_search_roots:
            db_path = HeatmapDatabase.resolve_path(root, rel)
            if db_path is not None:
                return db_path
        return None

    def heatmap_db_for_slide(rel: str) -> HeatmapDatabase | None:
        db_path = heatmap_db_path_for_slide(rel)
        if db_path is None:
            return None
        return HeatmapDatabase.for_path(db_path)

    @functools.lru_cache(maxsize=256)
    def _flat_overlay_files(directory: str, extensions_key: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
        """Cached non-recursive listing of overlay files in a directory."""
        here = Path(directory)
        if not here.is_dir():
            return ()
        rows: list[tuple[str, str]] = []
        for path in here.iterdir():
            if path.is_file() and path.suffix.lower() in extensions_key:
                rows.append((path.stem, str(path.resolve())))
        return tuple(sorted(rows))

    def resolve_slide(rel: str) -> Path:
        """Resolves a slide relative path, guarding against traversal."""
        candidate = (root_dir / rel).resolve()
        try:
            candidate.relative_to(root_dir)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="slide not found") from exc
        if not candidate.exists() or not _matches_extension(candidate.name, extensions):
            raise HTTPException(status_code=404, detail="slide not found")
        return candidate

    def find_annotation_files(rel: str) -> list[Path]:
        """Returns every GeoJSON annotation file belonging to a slide.

        Two layouts are supported and combined: a per-slide subfolder named
        after the slide stem (all GeoJSON inside it), and flat files whose name
        is the slide stem (optionally followed by a separator and a suffix).
        """
        if annotations_root is None:
            return []
        rel_path = Path(rel)
        stem = rel_path.stem
        found: dict[Path, None] = {}

        # 1) Per-slide subfolder holding all of that slide's annotation layers.
        for subdir in (annotations_root / stem, annotations_root / rel_path.parent / stem):
            if subdir.is_dir():
                for f in _collect_geojson(subdir):
                    found[f.resolve()] = None

        # 2) Flat files matching the stem exactly or stem + separator + suffix.
        for directory in (annotations_root, annotations_root / rel_path.parent):
            if not directory.is_dir():
                continue
            for file_stem, resolved_str in _flat_overlay_files(
                str(directory.resolve()), _ANNOTATION_EXTENSIONS
            ):
                if _stem_matches_slide(file_stem, stem):
                    found[Path(resolved_str)] = None

        # Guard against path traversal and return a stable, sorted list.
        result: list[Path] = []
        for resolved in found:
            try:
                resolved.relative_to(annotations_root)
            except ValueError:
                continue
            result.append(resolved)
        return sorted(result)

    # Cache the opened slide per relative path so repeated tile requests reuse
    # the same reader. The cached tuple keeps the FastSlide handle alive, which
    # matters because each SlideImageView only holds a weak reader handle.
    @functools.lru_cache(maxsize=16)
    def open_slide(rel: str) -> tuple[fastslide.FastSlide, tuple[XYZPyramid, ...]]:
        path = resolve_slide(rel)
        slide = fastslide.FastSlide.from_file_path(str(path))
        images = slide.images
        pyramids = tuple(XYZPyramid(images[i], tile_size=tile_size) for i in range(len(images)))
        return slide, pyramids

    def slide_pyramid_info(rel: str) -> tuple[int, int, list[float]] | None:
        try:
            _, pyramids = open_slide(rel)
            primary = pyramids[0]
            width, height = primary.level0_dimensions
            return width, height, primary.resolutions
        except HTTPException:
            return None

    app = FastAPI(title="FastSlide XYZ Multi-Slide Viewer")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(_BROWSER_HTML, media_type="text/html")

    @app.get("/viewer")
    def viewer() -> FileResponse:
        return FileResponse(_INDEX_HTML, media_type="text/html")

    @app.get("/api/slides")
    def api_slides() -> JSONResponse:
        slides = _list_slides(root_dir, extensions)
        _attach_overlay_metadata(
            slides,
            annotations_root=annotations_root,
            heatmap_db_path_for_slide=heatmap_db_path_for_slide,
            find_annotation_files=find_annotation_files,
        )
        overlay_sources: dict[str, bool] = {}
        if annotations_root is not None:
            overlay_sources["annotations"] = True
        if heatmaps_db_file is not None or heatmaps_search_roots:
            overlay_sources["heatmaps"] = True
        payload: dict[str, object] = {
            "root": str(root_dir),
            "extensions": sorted(extensions),
            "slides": slides,
        }
        if overlay_sources:
            payload["overlay_sources"] = overlay_sources
        return JSONResponse(payload)

    @app.get("/info")
    def info(slide: str = Query(...)) -> JSONResponse:
        reader, pyramids = open_slide(slide)
        image_infos = []
        for i, pyramid in enumerate(pyramids):
            entry = pyramid.info()
            entry["index"] = i
            view = pyramid.slide
            entry["name"] = view.name if hasattr(view, "name") else f"Image {i}"
            image_infos.append(entry)
        return JSONResponse(
            {
                "name": Path(slide).name,
                "format": reader.format,
                "primary_index": reader.images.primary_index,
                "images": image_infos,
            }
        )

    @app.get("/annotations")
    def annotations(slide: str = Query(...)) -> JSONResponse:
        resolve_slide(slide)  # validate the slide reference (404 otherwise)
        items: list[dict[str, object]] = []
        for path in find_annotation_files(slide):
            try:
                geojson = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            try:
                name = path.relative_to(annotations_root).with_suffix("").as_posix()
            except ValueError:
                name = path.stem
            items.append({"name": name, "geojson": geojson})
        return JSONResponse({"annotations": items})

    def _annotation_titles(slide: str) -> dict[str, str]:
        """Maps annotation relative keys to GeoJSON display titles."""
        if annotations_root is None:
            return {}
        titles: dict[str, str] = {}
        for path in find_annotation_files(slide):
            try:
                geojson = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            label = geojson_labels.geojson_title(geojson)
            if not label:
                continue
            try:
                key = path.relative_to(annotations_root).with_suffix("").as_posix()
            except ValueError:
                key = path.stem
            titles[key] = label
        return titles

    def _heatmap_display_title(heatmap_key: str, ann_titles: dict[str, str]) -> str | None:
        """Looks up a heatmap's label from matching annotation GeoJSON."""
        if heatmap_key in ann_titles:
            return ann_titles[heatmap_key]
        stem = Path(heatmap_key).name
        if stem in ann_titles:
            return ann_titles[stem]
        return None

    @app.get("/heatmaps")
    def heatmaps(slide: str = Query(...)) -> JSONResponse:
        resolve_slide(slide)  # validate the slide reference (404 otherwise)
        ann_titles = _annotation_titles(slide)
        items: list[dict[str, object]] = []

        db = heatmap_db_for_slide(slide)
        if db is None:
            return JSONResponse({"heatmaps": items})

        records = db.try_list_heatmaps()
        if records is None:
            return JSONResponse({"heatmaps": items, "pending": True})

        pyramid = slide_pyramid_info(slide)
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
    def heatmap_tile_png(
        name: str,
        z: int,
        x: int,
        y: int,
        slide: str = Query(...),
    ) -> Response:
        resolve_slide(slide)
        db = heatmap_db_for_slide(slide)
        if db is None:
            raise HTTPException(status_code=404, detail="heatmap database not found")
        record = db.get_by_name(name)
        if record is None:
            raise HTTPException(status_code=404, detail="heatmap not found")
        pyramid = slide_pyramid_info(slide)
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
    def heatmap_lut(
        slide: str = Query(...),
        cmap: str = Query(heatmap_lib.DEFAULT_COLORMAP),
    ) -> Response:
        resolve_slide(slide)
        try:
            data = heatmap_lib.colormap_lut_bytes(cmap)
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return Response(content=data, media_type="application/octet-stream")

    @app.get("/heatmap-colorbar.png")
    def heatmap_colorbar(
        slide: str = Query(...),
        cmap: str = Query(heatmap_lib.DEFAULT_COLORMAP),
        width: int = 160,
        height: int = 24,
    ) -> Response:
        resolve_slide(slide)
        try:
            data = heatmap_lib.colorbar_png(cmap, width=width, height=height)
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return Response(content=data, media_type="image/png")

    @app.get("/tiles/{image}/{z}/{x}/{y}.{ext}")
    def tile(image: int, z: int, x: int, y: int, ext: str, slide: str = Query(...)) -> Response:
        media_type = _MEDIA_TYPES.get(ext.lower())
        if media_type is None:
            raise HTTPException(status_code=404, detail=f"unsupported extension: {ext}")
        _, pyramids = open_slide(slide)
        if not 0 <= image < len(pyramids):
            raise HTTPException(status_code=404, detail=f"no image {image}")
        try:
            data = pyramids[image].get_tile_bytes(z, x, y, fmt=ext, quality=jpeg_quality)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return Response(content=data, media_type=media_type)

    return app


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a folder of slides as XYZ pyramids.")
    parser.add_argument("root", help="Directory to scan for supported slides.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000).")
    parser.add_argument("--tile-size", type=int, default=256, help="Tile size in px (default: 256).")
    parser.add_argument("--jpeg-quality", type=int, default=85, help="JPEG quality 1-100 (default: 85).")
    parser.add_argument(
        "--annotations-dir",
        default=None,
        help="Directory of per-slide GeoJSON files (matched by slide stem).",
    )
    parser.add_argument(
        "--heatmaps-dir",
        default=None,
        help="Directory of per-slide heatmaps.sqlite databases (matched by slide stem).",
    )
    parser.add_argument(
        "--heatmaps-db",
        default=None,
        help=(
            "SQLite heatmap database file or directory of per-slide heatmaps.sqlite "
            "(XYZ tile pyramid)."
        ),
    )
    return parser.parse_args()


def _warn_fd_limit() -> None:
    """Warn when the process soft open-file limit is likely too low for the viewer."""
    import resource
    import sys

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < 4096:
        print(
            f"WARNING: open-file limit is {soft} (hard {hard}). "
            "Heavy viewer use may fail with 'Too many open files'. "
            "Run: ulimit -n 65536",
            file=sys.stderr,
        )


def main() -> None:
    """CLI entry point."""
    args = _parse_args()
    _warn_fd_limit()
    app = create_app(
        args.root,
        tile_size=args.tile_size,
        jpeg_quality=args.jpeg_quality,
        annotations_dir=args.annotations_dir,
        heatmaps_dir=args.heatmaps_dir,
        heatmaps_db=args.heatmaps_db,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
