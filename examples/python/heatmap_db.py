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
"""SQLite-backed heatmap storage with an XYZ tile pyramid and R-tree index.

Each slide can have a ``heatmaps.sqlite`` database (typically inside the slide's
heatmap subfolder). A heatmap stores its low-resolution intensity grid plus
pre-rendered LA PNG tiles aligned to the slide's XYZ tile grid for smooth
pan/zoom in the FastSlide viewer.

Notes
-----
Tile coordinates follow the same XYZ convention as :class:`fastslide.xyz_pyramid.XYZPyramid`.
Intensity is stored as greyscale-with-alpha (``LA``) PNG bytes; the viewer applies
colormaps client-side via :mod:`heatmap`.
"""

from __future__ import annotations

import io
import json
import math
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

import heatmap as heatmap_lib

HEATMAPS_DB_FILENAME = "heatmaps.sqlite"
_DEFAULT_BUSY_TIMEOUT_MS = 250

_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS heatmaps (
    id              INTEGER PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,
    title           TEXT,
    x_offset        INTEGER NOT NULL,
    y_offset        INTEGER NOT NULL,
    size_per_pixel  REAL NOT NULL,
    width           INTEGER NOT NULL,
    height          INTEGER NOT NULL,
    max_value       REAL NOT NULL,
    slide_width     INTEGER,
    slide_height    INTEGER,
    tile_size       INTEGER NOT NULL DEFAULT 256,
    max_zoom        INTEGER NOT NULL DEFAULT 0,
    resolutions     TEXT,
    source_path     TEXT,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS heatmap_rasters (
    heatmap_id      INTEGER PRIMARY KEY REFERENCES heatmaps(id) ON DELETE CASCADE,
    intensity       BLOB NOT NULL,
    alpha           BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS heatmap_tiles (
    id              INTEGER PRIMARY KEY,
    heatmap_id      INTEGER NOT NULL REFERENCES heatmaps(id) ON DELETE CASCADE,
    z               INTEGER NOT NULL,
    x               INTEGER NOT NULL,
    y               INTEGER NOT NULL,
    width           INTEGER NOT NULL,
    height          INTEGER NOT NULL,
    png             BLOB NOT NULL,
    UNIQUE(heatmap_id, z, x, y)
);

CREATE VIRTUAL TABLE IF NOT EXISTS heatmap_tile_rtree USING rtree(
    id,
    min_x, max_x,
    min_y, max_y
);

CREATE INDEX IF NOT EXISTS idx_heatmap_tiles_lookup
    ON heatmap_tiles(heatmap_id, z, x, y);
"""


@dataclass(frozen=True)
class HeatmapRecord:
    """One heatmap layer loaded from a ``heatmaps.sqlite`` database.

    Attributes
    ----------
    id : int
        Primary key in the ``heatmaps`` table.
    name : str
        Unique layer identifier used in tile URLs.
    title : str or None
        Optional human-readable label.
    meta : dict
        Georeferencing and raster metadata (offsets, size, resolutions, etc.).
    tiled : bool
        ``True`` when the record is served as an XYZ tile pyramid.
    """

    id: int
    name: str
    title: str | None
    meta: dict[str, Any]
    tiled: bool


class HeatmapDatabase:
    """Read/write access to a per-slide ``heatmaps.sqlite`` database.

    Parameters
    ----------
    db_path : str or pathlib.Path
        Path to the SQLite database file.

    Attributes
    ----------
    DB_FILENAME : str
        Default database basename (``heatmaps.sqlite``).
    DEFAULT_TILE_SIZE : int
        Default XYZ tile edge length in pixels.
    """

    DB_FILENAME = HEATMAPS_DB_FILENAME
    DB_BASENAMES = (HEATMAPS_DB_FILENAME,)
    DEFAULT_TILE_SIZE = 256

    _TRANSPARENT_LA_PNG: bytes | None = None
    _thread_local = threading.local()

    def __init__(self, db_path: str | Path, *, create: bool = False) -> None:
        self._path = Path(db_path)
        self._conn: sqlite3.Connection | None = None
        self.open(create=create)

    @property
    def path(self) -> Path:
        """Path to the SQLite database file."""
        return self._path

    @property
    def conn(self) -> sqlite3.Connection:
        """Open SQLite connection.

        Raises
        ------
        RuntimeError
            If :meth:`open` has not been called and the database is closed.
        """
        if self._conn is None:
            raise RuntimeError("HeatmapDatabase is not open")
        return self._conn

    @classmethod
    def for_path(cls, db_path: str | Path, *, create: bool = False) -> HeatmapDatabase:
        """Return a per-thread cached database handle.

        FastAPI serves requests from a worker thread pool, so SQLite connections
        must not be shared across threads.
        """
        resolved = str(Path(db_path).resolve())
        key = (resolved, create)
        caches: dict[tuple[str, bool], HeatmapDatabase] | None = getattr(
            cls._thread_local, "instances", None
        )
        if caches is None:
            caches = {}
            cls._thread_local.instances = caches
        db = caches.get(key)
        if db is None:
            db = cls(db_path, create=create)
            caches[key] = db
        return db

    def __enter__(self) -> HeatmapDatabase:
        if self._conn is None:
            self.open()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()

    def open(self, *, create: bool = False) -> None:
        """Open the database, optionally creating it and the schema.

        Parameters
        ----------
        create : bool, optional
            When ``True``, create parent directories and initialise the schema if
            the file does not exist. When ``False``, the file must already exist.
        """
        if self._conn is not None:
            return
        if create:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        elif not self._path.is_file():
            raise FileNotFoundError(f"heatmap database not found: {self._path}")
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout = {_DEFAULT_BUSY_TIMEOUT_MS}")
        if create:
            conn.executescript(_SCHEMA)
            conn.execute("PRAGMA journal_mode=WAL")
        else:
            try:
                conn.execute("PRAGMA journal_mode=WAL")
            except sqlite3.OperationalError:
                pass
        self._conn = conn

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @classmethod
    def resolve_path(cls, root: Path, slide_rel: str) -> Path | None:
        """Locate ``heatmaps.sqlite`` for a slide.

        Mirrors the annotation folder layout used by the viewer servers: a
        per-slide subfolder named after the slide stem, or a stem-named file in
        the heatmaps root.

        Parameters
        ----------
        root : pathlib.Path
            Directory that contains per-slide heatmap subfolders or flat DB files.
        slide_rel : str
            Slide path relative to the served slides root.

        Returns
        -------
        pathlib.Path or None
            Resolved database path when found, otherwise ``None``.
        """
        root = Path(root)
        rel_path = Path(slide_rel)
        stem = rel_path.stem
        seen: set[Path] = set()

        def pick(path: Path) -> Path | None:
            resolved = path.resolve()
            if resolved in seen:
                return None
            seen.add(resolved)
            return resolved if resolved.is_file() else None

        for subdir in (root / stem, root / rel_path.parent / stem):
            for basename in cls.DB_BASENAMES:
                found = pick(subdir / basename)
                if found is not None:
                    return found

        for directory in (root, root / rel_path.parent):
            if not directory.is_dir():
                continue
            for ext in (".sqlite", ".db"):
                found = pick(directory / f"{stem}{ext}")
                if found is not None:
                    return found
        return None

    def list_heatmaps(self) -> list[HeatmapRecord]:
        """Return all heatmap layers in the database, sorted by name.

        Returns
        -------
        list of HeatmapRecord
        """
        rows = self.conn.execute("SELECT * FROM heatmaps ORDER BY name").fetchall()
        return [
            HeatmapRecord(
                id=int(row["id"]),
                name=row["name"],
                title=row["title"],
                meta=self._row_to_meta(row),
                tiled=True,
            )
            for row in rows
        ]

    def try_list_heatmaps(self) -> list[HeatmapRecord] | None:
        """Return heatmap layers, or ``None`` when the database is temporarily locked.

        Returns
        -------
        list of HeatmapRecord or None
            Layer list on success; ``None`` when another process holds a write lock
            (for example while a pipeline is still building ``heatmaps.sqlite``).
        """
        try:
            return self.list_heatmaps()
        except sqlite3.OperationalError as exc:
            if self._is_locked_error(exc):
                return None
            raise

    @classmethod
    def try_list_heatmaps_at(cls, db_path: str | Path) -> list[HeatmapRecord] | None:
        """List heatmap layers once and close the SQLite connection.

        Use this for bulk metadata scans (for example ``/api/slides``) instead of
        :meth:`for_path`, which caches one open connection per database per thread.
        """
        path = Path(db_path)
        if not path.is_file():
            return None
        db = cls(path, create=False)
        try:
            return db.try_list_heatmaps()
        finally:
            db.close()

    @staticmethod
    def _is_locked_error(exc: sqlite3.OperationalError) -> bool:
        message = str(exc).lower()
        return "locked" in message or "busy" in message

    def get_by_name(self, name: str) -> HeatmapRecord | None:
        """Return one heatmap layer by its unique name.

        Parameters
        ----------
        name : str
            Layer identifier stored in the ``heatmaps`` table.

        Returns
        -------
        HeatmapRecord or None
            The matching record, or ``None`` when no layer with that name exists.
        """
        row = self.conn.execute("SELECT * FROM heatmaps WHERE name = ?", (name,)).fetchone()
        if row is None:
            return None
        return HeatmapRecord(
            id=int(row["id"]),
            name=row["name"],
            title=row["title"],
            meta=self._row_to_meta(row),
            tiled=True,
        )

    def configure_pyramid(
        self,
        heatmap_id: int,
        *,
        slide_width: int,
        slide_height: int,
        tile_size: int,
        resolutions: list[float],
    ) -> None:
        """Store slide pyramid parameters used to build XYZ heatmap tiles.

        Parameters
        ----------
        heatmap_id : int
            Primary key of the heatmap row to update.
        slide_width, slide_height : int
            Level-0 slide dimensions in pixels.
        tile_size : int
            XYZ tile edge length in pixels.
        resolutions : list of float
            Downsample factors per zoom level, coarsest first (matching
            :class:`fastslide.xyz_pyramid.XYZPyramid`).
        """
        self.conn.execute(
            """
            UPDATE heatmaps
            SET slide_width = ?, slide_height = ?, tile_size = ?, max_zoom = ?, resolutions = ?
            WHERE id = ?
            """,
            (
                int(slide_width),
                int(slide_height),
                int(tile_size),
                len(resolutions) - 1,
                json.dumps([float(r) for r in resolutions]),
                heatmap_id,
            ),
        )
        self.conn.commit()

    def upsert_heatmap(
        self,
        *,
        name: str,
        meta: dict[str, Any],
        intensity: np.ndarray,
        alpha: np.ndarray,
        title: str | None = None,
        source_path: str | None = None,
    ) -> int:
        """Insert or replace a heatmap raster and clear cached tiles.

        Parameters
        ----------
        name : str
            Unique layer name.
        meta : dict
            Georeferencing metadata with keys ``x_offset``, ``y_offset``,
            ``size_per_pixel``, ``width``, ``height``, and ``max_value``.
        intensity, alpha : numpy.ndarray
            ``uint8`` arrays of shape ``(height, width)``.
        title : str or None, optional
            Human-readable label; inferred from ``meta`` when omitted.
        source_path : str or None, optional
            Provenance path stored in the database.

        Returns
        -------
        int
            Primary key of the upserted heatmap row.
        """
        now = self._utc_now()
        row = self.conn.execute("SELECT id FROM heatmaps WHERE name = ?", (name,)).fetchone()
        if row is None:
            cur = self.conn.execute(
                """
                INSERT INTO heatmaps (
                    name, title, x_offset, y_offset, size_per_pixel,
                    width, height, max_value, source_path, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    name,
                    title or heatmap_lib.meta_title(meta),
                    int(meta["x_offset"]),
                    int(meta["y_offset"]),
                    float(meta["size_per_pixel"]),
                    int(meta["width"]),
                    int(meta["height"]),
                    float(meta["max_value"]),
                    source_path,
                    now,
                ),
            )
            heatmap_id = int(cur.lastrowid)
        else:
            heatmap_id = int(row["id"])
            self.conn.execute(
                """
                UPDATE heatmaps SET
                    title = ?, x_offset = ?, y_offset = ?, size_per_pixel = ?,
                    width = ?, height = ?, max_value = ?, source_path = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    title or heatmap_lib.meta_title(meta),
                    int(meta["x_offset"]),
                    int(meta["y_offset"]),
                    float(meta["size_per_pixel"]),
                    int(meta["width"]),
                    int(meta["height"]),
                    float(meta["max_value"]),
                    source_path,
                    now,
                    heatmap_id,
                ),
            )
            self.conn.execute(
                "DELETE FROM heatmap_tile_rtree WHERE id IN (SELECT id FROM heatmap_tiles WHERE heatmap_id = ?)",
                (heatmap_id,),
            )
            self.conn.execute(
                "DELETE FROM heatmap_tiles WHERE heatmap_id = ?",
                (heatmap_id,),
            )

        self.conn.execute(
            """
            INSERT INTO heatmap_rasters (heatmap_id, intensity, alpha)
            VALUES (?, ?, ?)
            ON CONFLICT(heatmap_id) DO UPDATE SET
                intensity = excluded.intensity,
                alpha = excluded.alpha
            """,
            (heatmap_id, intensity.tobytes(), alpha.tobytes()),
        )
        self.conn.commit()
        return heatmap_id

    def build_pyramid(self, heatmap_id: int) -> int:
        """Pre-render all heatmap tiles that intersect the heatmap extent.

        Parameters
        ----------
        heatmap_id : int
            Primary key of the heatmap to tile.

        Returns
        -------
        int
            Number of LA PNG tiles written.

        Raises
        ------
        ValueError
            If the heatmap is missing or its pyramid is not configured.
        """
        row = self.conn.execute("SELECT * FROM heatmaps WHERE id = ?", (heatmap_id,)).fetchone()
        if row is None:
            raise ValueError(f"heatmap {heatmap_id} not found")
        meta = self._row_to_meta(row)
        if not row["slide_width"] or not row["slide_height"] or not row["resolutions"]:
            raise ValueError(f"heatmap {heatmap_id}: pyramid not configured")

        slide_width = int(row["slide_width"])
        slide_height = int(row["slide_height"])
        tile_size = int(row["tile_size"])
        resolutions = json.loads(row["resolutions"])
        intensity, alpha = self._load_raster(heatmap_id)
        hm_x0, hm_y0, hm_x1, hm_y1 = self._heatmap_bbox_l0(meta)

        self.conn.execute(
            "DELETE FROM heatmap_tile_rtree WHERE id IN (SELECT id FROM heatmap_tiles WHERE heatmap_id = ?)",
            (heatmap_id,),
        )
        self.conn.execute("DELETE FROM heatmap_tiles WHERE heatmap_id = ?", (heatmap_id,))

        written = 0
        for z in range(len(resolutions)):
            level_w = max(1, int(math.ceil(slide_width / float(resolutions[z]))))
            level_h = max(1, int(math.ceil(slide_height / float(resolutions[z]))))
            cols = math.ceil(level_w / tile_size)
            rows = math.ceil(level_h / tile_size)
            for x in range(cols):
                for y in range(rows):
                    bounds = self._tile_level0_bounds(
                        z, x, y, slide_width, slide_height, tile_size, resolutions
                    )
                    if bounds is None:
                        continue
                    l0_x0, l0_y0, l0_x1, l0_y1, w, h = bounds
                    if not self._intersects(l0_x0, l0_y0, l0_x1, l0_y1, hm_x0, hm_y0, hm_x1, hm_y1):
                        continue
                    out_i, out_a = self._sample_tile(
                        intensity, alpha, meta, l0_x0, l0_y0, l0_x1, l0_y1, w, h
                    )
                    if not out_a.any():
                        continue
                    png = self._la_png_bytes(out_i, out_a)
                    cur = self.conn.execute(
                        """
                        INSERT INTO heatmap_tiles
                            (heatmap_id, z, x, y, width, height, png)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (heatmap_id, z, x, y, w, h, png),
                    )
                    self._insert_tile_spatial(int(cur.lastrowid), l0_x0, l0_y0, l0_x1, l0_y1)
                    written += 1
        self.conn.commit()
        return written

    def get_tile_png(
        self,
        heatmap_id: int,
        z: int,
        x: int,
        y: int,
        *,
        slide_width: int | None = None,
        slide_height: int | None = None,
        tile_size: int | None = None,
        resolutions: list[float] | None = None,
    ) -> bytes:
        """Return an LA PNG tile, building and caching it on demand when needed.

        Parameters
        ----------
        heatmap_id : int
            Primary key of the heatmap layer.
        z, x, y : int
            XYZ tile indices aligned to the slide pyramid.
        slide_width, slide_height : int or None, optional
            Level-0 slide dimensions; read from the database when omitted.
        tile_size : int or None, optional
            Tile edge length in pixels; read from the database when omitted.
        resolutions : list of float or None, optional
            Downsample factors per zoom level; read from the database when omitted.

        Returns
        -------
        bytes
            Encoded LA PNG tile bytes (possibly fully transparent outside the
            heatmap extent).

        Raises
        ------
        ValueError
            If ``heatmap_id`` does not exist.
        """
        row = self.conn.execute("SELECT * FROM heatmaps WHERE id = ?", (heatmap_id,)).fetchone()
        if row is None:
            raise ValueError(f"heatmap {heatmap_id} not found")

        cached = self.conn.execute(
            """
            SELECT png, width, height FROM heatmap_tiles
            WHERE heatmap_id = ? AND z = ? AND x = ? AND y = ?
            """,
            (heatmap_id, z, x, y),
        ).fetchone()
        if cached is not None:
            return bytes(cached["png"])

        sw = slide_width or row["slide_width"]
        sh = slide_height or row["slide_height"]
        ts = tile_size or row["tile_size"] or self.DEFAULT_TILE_SIZE
        res_json = row["resolutions"]
        if resolutions is None and res_json:
            resolutions = json.loads(res_json)
        if not sw or not sh or not resolutions:
            return self._transparent_tile_png(ts, ts)

        meta = self._row_to_meta(row)
        bounds = self._tile_level0_bounds(z, x, y, int(sw), int(sh), int(ts), resolutions)
        if bounds is None:
            return self._transparent_tile_png(ts, ts)
        l0_x0, l0_y0, l0_x1, l0_y1, w, h = bounds
        hm_x0, hm_y0, hm_x1, hm_y1 = self._heatmap_bbox_l0(meta)
        if not self._intersects(l0_x0, l0_y0, l0_x1, l0_y1, hm_x0, hm_y0, hm_x1, hm_y1):
            return self._transparent_tile_png(w, h)

        intensity, alpha = self._load_raster(heatmap_id)
        out_i, out_a = self._sample_tile(intensity, alpha, meta, l0_x0, l0_y0, l0_x1, l0_y1, w, h)
        png = self._la_png_bytes(out_i, out_a)

        if out_a.any():
            cur = self.conn.execute(
                """
                INSERT OR IGNORE INTO heatmap_tiles
                    (heatmap_id, z, x, y, width, height, png)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (heatmap_id, z, x, y, w, h, png),
            )
            if cur.lastrowid:
                self._insert_tile_spatial(cur.lastrowid, l0_x0, l0_y0, l0_x1, l0_y1)
                self.conn.commit()
        return png

    @staticmethod
    def heatmap_list_entry(record: HeatmapRecord) -> dict[str, object]:
        """Build the JSON object returned by the viewer ``/heatmaps`` endpoint.

        Parameters
        ----------
        record : HeatmapRecord
            Heatmap metadata loaded from the database.

        Returns
        -------
        dict
            Serializable heatmap descriptor including extent and tile parameters.
        """
        meta = record.meta
        return {
            "name": record.name,
            "title": record.title or record.name,
            "pending": False,
            "tiled": True,
            "extent": heatmap_lib.map_extent(meta),
            "width": meta["width"],
            "height": meta["height"],
            "max_value": meta["max_value"],
            "tile_size": meta.get("tile_size", HeatmapDatabase.DEFAULT_TILE_SIZE),
            "max_zoom": meta.get("max_zoom"),
            "resolutions": meta.get("resolutions"),
        }

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    @classmethod
    def _row_to_meta(cls, row: sqlite3.Row) -> dict[str, Any]:
        meta = {
            "x_offset": int(row["x_offset"]),
            "y_offset": int(row["y_offset"]),
            "size_per_pixel": float(row["size_per_pixel"]),
            "width": int(row["width"]),
            "height": int(row["height"]),
            "max_value": float(row["max_value"]),
        }
        if row["title"]:
            meta["title"] = row["title"]
        if row["tile_size"]:
            meta["tile_size"] = int(row["tile_size"])
        if row["max_zoom"] is not None:
            meta["max_zoom"] = int(row["max_zoom"])
        if row["resolutions"]:
            meta["resolutions"] = json.loads(row["resolutions"])
        return meta

    def _load_raster(self, heatmap_id: int) -> tuple[np.ndarray, np.ndarray]:
        row = self.conn.execute(
            "SELECT intensity, alpha FROM heatmap_rasters WHERE heatmap_id = ?",
            (heatmap_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"heatmap {heatmap_id}: raster missing")
        hm = self.conn.execute(
            "SELECT width, height FROM heatmaps WHERE id = ?",
            (heatmap_id,),
        ).fetchone()
        width = int(hm["width"])
        height = int(hm["height"])
        intensity = np.frombuffer(row["intensity"], dtype=np.uint8).reshape(height, width)
        alpha = np.frombuffer(row["alpha"], dtype=np.uint8).reshape(height, width)
        return intensity, alpha

    @staticmethod
    def _la_png_bytes(intensity: np.ndarray, alpha: np.ndarray) -> bytes:
        la = np.dstack([intensity, alpha])
        buffer = io.BytesIO()
        Image.fromarray(la, mode="LA").save(buffer, format="PNG", optimize=True)
        return buffer.getvalue()

    @classmethod
    def _transparent_tile_png(cls, width: int, height: int) -> bytes:
        if width == cls.DEFAULT_TILE_SIZE and height == cls.DEFAULT_TILE_SIZE and cls._TRANSPARENT_LA_PNG is not None:
            return cls._TRANSPARENT_LA_PNG
        data = cls._la_png_bytes(
            np.zeros((height, width), dtype=np.uint8),
            np.zeros((height, width), dtype=np.uint8),
        )
        if width == cls.DEFAULT_TILE_SIZE and height == cls.DEFAULT_TILE_SIZE:
            cls._TRANSPARENT_LA_PNG = data
        return data

    @staticmethod
    def _heatmap_bbox_l0(meta: dict[str, Any]) -> tuple[float, float, float, float]:
        x0 = float(meta["x_offset"])
        y0 = float(meta["y_offset"])
        x1 = x0 + float(meta["width"]) * float(meta["size_per_pixel"])
        y1 = y0 + float(meta["height"]) * float(meta["size_per_pixel"])
        return x0, y0, x1, y1

    @staticmethod
    def _tile_level0_bounds(
        z: int,
        x: int,
        y: int,
        slide_width: int,
        slide_height: int,
        tile_size: int,
        resolutions: list[float],
    ) -> tuple[int, int, int, int, int, int] | None:
        if z < 0 or z >= len(resolutions):
            return None
        down = float(resolutions[z])
        level_w = max(1, int(math.ceil(slide_width / down)))
        level_h = max(1, int(math.ceil(slide_height / down)))
        px = x * tile_size
        py = y * tile_size
        if px >= level_w or py >= level_h:
            return None
        w = min(tile_size, level_w - px)
        h = min(tile_size, level_h - py)
        l0_x0 = int(px * down)
        l0_y0 = int(py * down)
        l0_x1 = int(min((px + w) * down, slide_width))
        l0_y1 = int(min((py + h) * down, slide_height))
        return l0_x0, l0_y0, l0_x1, l0_y1, w, h

    @staticmethod
    def _intersects(
        ax0: float,
        ay0: float,
        ax1: float,
        ay1: float,
        bx0: float,
        by0: float,
        bx1: float,
        by1: float,
    ) -> bool:
        return ax0 < bx1 and ax1 > bx0 and ay0 < by1 and ay1 > by0

    @staticmethod
    def _sample_tile(
        intensity: np.ndarray,
        alpha: np.ndarray,
        meta: dict[str, Any],
        l0_x0: int,
        l0_y0: int,
        l0_x1: int,
        l0_y1: int,
        out_w: int,
        out_h: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        x_off = float(meta["x_offset"])
        y_off = float(meta["y_offset"])
        spp = float(meta["size_per_pixel"])
        grid_h, grid_w = intensity.shape

        if out_w <= 0 or out_h <= 0 or l0_x1 <= l0_x0 or l0_y1 <= l0_y0:
            z = np.zeros((max(out_h, 1), max(out_w, 1)), dtype=np.uint8)
            return z, z.copy()

        xs = np.linspace(l0_x0 + 0.5, l0_x1 - 0.5, out_w, dtype=np.float64)
        ys = np.linspace(l0_y0 + 0.5, l0_y1 - 0.5, out_h, dtype=np.float64)
        gx = np.floor((xs - x_off) / spp).astype(np.int32)
        gy = np.floor((ys - y_off) / spp).astype(np.int32)
        gx_grid, gy_grid = np.meshgrid(gx, gy)
        valid = (gx_grid >= 0) & (gx_grid < grid_w) & (gy_grid >= 0) & (gy_grid < grid_h)
        out_i = np.zeros((out_h, out_w), dtype=np.uint8)
        out_a = np.zeros((out_h, out_w), dtype=np.uint8)
        out_i[valid] = intensity[gy_grid[valid], gx_grid[valid]]
        out_a[valid] = alpha[gy_grid[valid], gx_grid[valid]]
        return out_i, out_a

    def _insert_tile_spatial(
        self,
        tile_id: int,
        l0_x0: int,
        l0_y0: int,
        l0_x1: int,
        l0_y1: int,
    ) -> None:
        self.conn.execute(
            "INSERT INTO heatmap_tile_rtree (id, min_x, max_x, min_y, max_y) VALUES (?, ?, ?, ?, ?)",
            (tile_id, l0_x0, l0_x1, l0_y0, l0_y1),
        )
