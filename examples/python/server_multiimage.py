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
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response

import fastslide
from fastslide.xyz_pyramid import XYZPyramid

import heatmap as heatmap_lib

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


def _dedupe_heatmap_sources(paths: list[Path]) -> list[Path]:
    """Collapses heatmap sources sharing a base name, preferring ``.png``.

    A heatmap may exist as a rendered ``.png`` (with JSON sidecar) and/or the
    original ``.tsv``; both describe the same heatmap, so keep only one source
    per ``(folder, base name)`` and prefer the already-rendered PNG.
    """
    best: dict[tuple[str, str], Path] = {}
    for p in paths:
        key = (str(p.parent), p.stem.lower())
        current = best.get(key)
        if current is None or (
            current.suffix.lower() != ".png" and p.suffix.lower() == ".png"
        ):
            best[key] = p
    return sorted(best.values())


_HEATMAP_EXTENSIONS = (".png", ".tsv")


def create_app(
    root: str | Path,
    tile_size: int = 256,
    jpeg_quality: int = 85,
    annotations_dir: str | Path | None = None,
    heatmaps_dir: str | Path | None = None,
) -> FastAPI:
    """Builds the FastAPI app serving XYZ tiles for every slide under ``root``.

    Args:
        root: Directory to scan for supported slides (searched recursively).
        tile_size: Tile edge length in pixels.
        jpeg_quality: Quality used when encoding JPEG tiles.
        annotations_dir: Optional directory holding per-slide GeoJSON files,
            matched by the slide's file stem (and mirrored sub-path).
        heatmaps_dir: Optional directory holding per-slide heatmaps (PNG + JSON
            pairs, or legacy TSVs), matched the same way as annotations.

    Returns:
        A configured :class:`fastapi.FastAPI` instance.
    """
    root_dir = Path(root).resolve()
    if not root_dir.is_dir():
        raise NotADirectoryError(f"not a directory: {root_dir}")

    annotations_root = Path(annotations_dir).resolve() if annotations_dir else None
    if annotations_root is not None and not annotations_root.is_dir():
        raise NotADirectoryError(f"not a directory: {annotations_root}")

    heatmaps_root = Path(heatmaps_dir).resolve() if heatmaps_dir else None
    if heatmaps_root is not None and not heatmaps_root.is_dir():
        raise NotADirectoryError(f"not a directory: {heatmaps_root}")

    extensions = _supported_extensions()

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
            for f in directory.iterdir():
                if not f.is_file() or f.suffix.lower() not in _ANNOTATION_EXTENSIONS:
                    continue
                name = f.stem
                if name == stem or (
                    name.startswith(stem) and name[len(stem) : len(stem) + 1] in _STEM_SEPARATORS
                ):
                    found[f.resolve()] = None

        # Guard against path traversal and return a stable, sorted list.
        result: list[Path] = []
        for resolved in found:
            try:
                resolved.relative_to(annotations_root)
            except ValueError:
                continue
            result.append(resolved)
        return sorted(result)

    def find_heatmap_files(rel: str) -> list[Path]:
        """Returns the heatmap source files belonging to a slide.

        Mirrors :func:`find_annotation_files` (per-slide subfolder + stem-matched
        flat files), then collapses duplicate ``.png`` / ``.tsv`` sources.
        """
        if heatmaps_root is None:
            return []
        rel_path = Path(rel)
        stem = rel_path.stem
        found: dict[Path, None] = {}

        for subdir in (heatmaps_root / stem, heatmaps_root / rel_path.parent / stem):
            if subdir.is_dir():
                for ext in _HEATMAP_EXTENSIONS:
                    for f in subdir.rglob(f"*{ext}"):
                        if f.is_file():
                            found[f.resolve()] = None

        for directory in (heatmaps_root, heatmaps_root / rel_path.parent):
            if not directory.is_dir():
                continue
            for f in directory.iterdir():
                if not f.is_file() or f.suffix.lower() not in _HEATMAP_EXTENSIONS:
                    continue
                name = f.stem
                if name == stem or (
                    name.startswith(stem) and name[len(stem) : len(stem) + 1] in _STEM_SEPARATORS
                ):
                    found[f.resolve()] = None

        result: list[Path] = []
        for resolved in found:
            try:
                resolved.relative_to(heatmaps_root)
            except ValueError:
                continue
            result.append(resolved)
        return _dedupe_heatmap_sources(result)

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

    app = FastAPI(title="FastSlide XYZ Multi-Slide Viewer")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(_BROWSER_HTML, media_type="text/html")

    @app.get("/viewer")
    def viewer() -> FileResponse:
        return FileResponse(_INDEX_HTML, media_type="text/html")

    @app.get("/api/slides")
    def api_slides() -> JSONResponse:
        return JSONResponse(
            {
                "root": str(root_dir),
                "extensions": sorted(extensions),
                "slides": _list_slides(root_dir, extensions),
            }
        )

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

    def _heatmap_name(source: Path) -> str:
        try:
            return source.relative_to(heatmaps_root).with_suffix("").as_posix()
        except ValueError:
            return source.stem

    @app.get("/heatmaps")
    def heatmaps(slide: str = Query(...)) -> JSONResponse:
        resolve_slide(slide)  # validate the slide reference (404 otherwise)
        items: list[dict[str, object]] = []
        for source in find_heatmap_files(slide):
            try:
                _, meta = heatmap_lib.ensure_rendered(source)
            except (OSError, ValueError):
                continue
            items.append(
                {
                    "name": _heatmap_name(source),
                    "extent": heatmap_lib.map_extent(meta),
                    "width": meta["width"],
                    "height": meta["height"],
                    "max_value": meta["max_value"],
                }
            )
        return JSONResponse({"heatmaps": items})

    @app.get("/heatmap.png")
    def heatmap_png(
        slide: str = Query(...),
        name: str = Query(...),
        cmap: str = Query(heatmap_lib.DEFAULT_COLORMAP),
    ) -> Response:
        resolve_slide(slide)
        for source in find_heatmap_files(slide):
            if _heatmap_name(source) == name:
                try:
                    png_path, _ = heatmap_lib.ensure_rendered(source)
                    data = heatmap_lib.colorize_png(str(png_path), cmap)
                except (OSError, ValueError) as exc:
                    raise HTTPException(status_code=500, detail=str(exc)) from exc
                return Response(content=data, media_type="image/png")
        raise HTTPException(status_code=404, detail="heatmap not found")

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
        help="Directory of per-slide heatmaps (PNG+JSON or TSV, matched by slide stem).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = _parse_args()
    app = create_app(
        args.root,
        tile_size=args.tile_size,
        jpeg_quality=args.jpeg_quality,
        annotations_dir=args.annotations_dir,
        heatmaps_dir=args.heatmaps_dir,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
