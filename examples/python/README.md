# FastSlide XYZ pyramid viewer

A minimal example that serves a whole-slide image as a standard
XYZ (slippy-map) tile pyramid over HTTP and views it in the browser with
[OpenLayers](https://openlayers.org/).

It is the FastSlide counterpart to the OpenSlide `deepzoom_server.py` example,
with two differences:

- it serves an **XYZ pyramid** (`/tiles/{z}/{x}/{y}.jpg`) instead of Deep Zoom
  / DZI, so any XYZ-capable web map library can consume it directly, and
- it uses **FastAPI + uvicorn** instead of Flask.

The tiling itself is produced by `fastslide.XYZPyramid`, which ships with the
`fastslide` package, so you can reuse it outside this example.

Two servers are provided:

- `server.py` -- serves a **single slide** passed on the command line.
- `server_multiimage.py` -- scans a **folder** for supported slides and shows a
  file browser to pick one; clicking a slide opens the same viewer.

## Run with Bazel (from the fastslide module)

The examples are wired into the build, so from the `aifo/fastslide` module root
you can run them directly without installing anything.

Single slide:

```bash
bazelisk run //examples/python:viewer -- /abs/path/to/slide.svs
```

A whole folder (with a file browser at `/`):

```bash
bazelisk run //examples/python:viewer_multiimage -- /abs/path/to/slides
```

Extra flags are forwarded after the path, for example:

```bash
bazelisk run //examples/python:viewer -- /abs/path/to/slide.svs --port 8080 --tile-size 256
```

Then open <http://127.0.0.1:8000> (or the port you chose).

## Run with a plain Python environment

```bash
pip install -r requirements.txt
```

`fastslide` is listed in `requirements.txt`. If you are working from a source
checkout, install your locally built wheel instead, for example:

```bash
pip install fastapi "uvicorn[standard]" pillow numpy fastslide
```

Then run a server:

```bash
# single slide
python server.py /path/to/slide.svs

# a folder of slides (file browser at http://127.0.0.1:8000)
python server_multiimage.py /path/to/slides
```

Then open <http://127.0.0.1:8000>.

Options (both servers accept `--host`, `--port`, `--tile-size`, `--jpeg-quality`):

```bash
python server.py /path/to/slide.svs --host 0.0.0.0 --port 8000 --tile-size 256 --jpeg-quality 85
```

## GeoJSON annotations (overlay)

Both servers can overlay GeoJSON annotations on top of the slide. Coordinates
must be in **level-0 slide pixels** (origin top-left, y pointing down) -- the
same convention QuPath exports. A slide can carry **several annotation files at
once**; each becomes its own layer that you toggle independently in the viewer.

- `server.py` takes a file or folder: `--annotations /path/to/slide.json` or
  `--annotations /path/to/folder` (every GeoJSON in the folder becomes a layer).
- `server_multiimage.py` takes a folder: `--annotations-dir /path/to/annotations`.

```bash
python server_multiimage.py /path/to/slides --annotations-dir /path/to/annotations
```

For a slide `sub/2111 T2.mrxs`, the multi-image server collects, under the
annotations dir:

- **all** `*.json` / `*.geojson` inside a per-slide subfolder named after the
  slide stem (`<dir>/2111 T2/` or `<dir>/sub/2111 T2/`), and
- flat files whose name is the slide stem, or the stem followed by a separator
  (`_ - . space (`) -- e.g. `2111 T2.json` and `2111 T2_tissue_foreground.json`
  both match, but `2111 T21.json` does not.

Geometry handling in the viewer:

- `Point` / `MultiPoint` are drawn as **circles of a fixed physical diameter**
  (in millimeters) centered on each point, outline only. The diameter is
  adjustable in the Annotations panel; it is converted to pixels using the
  slide's MPP, so points are skipped for slides with no physical scale.
- `Polygon`, `MultiPolygon`, `LineString`, etc. are drawn as outlines with a
  translucent fill.
- Features are colored by their GeoJSON `classification.color` (`[r, g, b]`)
  when present, otherwise by a palette keyed on `classification.name`. Only
  point circles are labeled, since region layers can hold thousands of features.

The Annotations panel in the sidebar has a master show/hide toggle and a picker
to load one or more GeoJSON files directly from your machine. Each layer is
listed by its classification name (falling back to the file name) with its own
controls shown underneath: visibility, color, outline width, opacity, and -- for
layers containing points -- the circle diameter (mm).

## Endpoints

`server.py` (single slide):

| Route                                  | Description                                                                                                                                            |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `GET /`                                | The OpenLayers viewer (`index.html`).                                                                                                                  |
| `GET /info`                            | Slide metadata as JSON: file name/format, `primary_index`, and per-image entries (dimensions, zoom range, tile size, native levels, MPP, resolutions). |
| `GET /annotations`                     | Annotation layers as `{ "annotations": [ { "name", "geojson" }, ... ] }` (empty unless `--annotations` is set).                                         |
| `GET /tiles/{image}/{z}/{x}/{y}.{ext}` | A single tile from image `image`; `ext` is `jpg`, `jpeg`, or `png`.                                                                                    |

`server_multiimage.py` (folder): the slide is selected with a `?slide=<relpath>`
query parameter, where `<relpath>` is the path relative to the served folder.

| Route                                                  | Description                                              |
| ------------------------------------------------------ | -------------------------------------------------------- |
| `GET /`                                                | The file browser (`browser.html`).                       |
| `GET /api/slides`                                      | JSON: served root, supported extensions, and slide list. |
| `GET /viewer?slide=<relpath>`                          | The OpenLayers viewer for one slide.                     |
| `GET /info?slide=<relpath>`                            | Same payload as the single-slide `/info`.                |
| `GET /annotations?slide=<relpath>`                     | All annotation layers for the slide as a named list.     |
| `GET /tiles/{image}/{z}/{x}/{y}.{ext}?slide=<relpath>` | A single tile from a slide in the folder.                |

Slides with multiple navigable images (e.g. an Olympus VSI navigator plus
region images) expose one entry per image under `/info`, and the viewer shows
an image switcher in the sidebar.

## Tile convention

XYZ layout mapped onto the slide's native pyramid levels:

- Each zoom level `z` is one native pyramid level: `z = 0` is the coarsest
  native level and `z = max_zoom` is full resolution. The viewer's `TileGrid`
  uses the slide's `level_downsamples` as resolutions, so tiles are direct
  native reads (no server-side resampling) and OpenLayers scales between levels.
- `x` is the column (increasing rightwards), `y` is the row (increasing
  downwards), with the tile origin at the top-left of the slide.
