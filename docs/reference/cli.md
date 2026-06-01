# Command-Line Reference

mudm-tools ships exactly **one** installed console script — `mudm-serve` — plus a
converter CLI you run as a Python module. This page documents both interfaces in
full: every flag, its default, and what the command actually does on disk and on
the wire.

!!! info "Two entry points, one installed script"
    - **`mudm-serve`** is a real console script declared in `pyproject.toml`
      (`[project.scripts] mudm-serve = "mudm_tools.serve:main"`). After
      `pip install mudm-tools` it lands on your `PATH`.
    - **The converter CLI** has **no** installed script. There is **no**
      `mudm convert` command. Always invoke it as a module:
      `python -m mudm_tools.converters.cli ...`. (The argparse `prog` is `mudm`
      and some in-code docstrings show `mudm convert ...`, but that command is
      not installed — only `mudm-serve` is.)

---

## `mudm-serve`

Serves your tiled output (3D Tiles GLB, 2D MVT/raster, and Neuroglancer
precomputed) over HTTP and mounts a bundled web viewer at `/`. The process binds
to all interfaces (host `""`) and blocks until you press <kbd>Ctrl</kbd>+<kbd>C</kbd>.

=== "Installed script"

    ```bash
    mudm-serve --tiles-base output/
    ```

=== "uv run"

    ```bash
    uv run mudm-serve --tiles-base output/
    ```

=== "Python module"

    ```bash
    uv run python -m mudm_tools.serve --tiles-base output/
    ```

### Synopsis

```text
mudm-serve --tiles-base DIR
           [--port PORT]
           [--tiles2d-base DIR]
           [--viewer {3d,2d}]
           [--viewer-dir DIR]
```

### Flags

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--tiles-base` | str | **required** | Directory containing `pyramids.json` and per-pyramid tile subdirs. Resolved to an absolute path. A `WARNING` is printed (not fatal) if no `pyramids.json` is found. |
| `--port` | int | `8080` | TCP port to bind (`HTTPServer(("", port), ...)`). |
| `--tiles2d-base` | str | `""` | Directory of 2D datasets (each a subdir with `metadata.json`). Resolved to an absolute path only when non-empty; otherwise 2D tile serving is disabled. |
| `--viewer` | str | `3d` | Which bundled viewer to serve at `/`. One of `3d` or `2d`. |
| `--viewer-dir` | str | `None` | A custom viewer directory whose `index.html` is served at `/`. **Overrides `--viewer`.** |

!!! warning "`--tiles-base` is mandatory"
    `mudm-serve` exits with an argparse error if `--tiles-base` is omitted. Point
    it at the directory you passed as the output root when generating tiles —
    the one that holds `pyramids.json`. If that file is missing the server still
    starts, but the 3D viewer's pyramid dropdown will be empty.

### What the server does

When you start `mudm-serve` it:

1. Resolves `--tiles-base` (and `--tiles2d-base`, if set) to absolute paths.
2. Resolves the viewer directory: `--viewer-dir` if given, otherwise the bundled
   viewer named by `--viewer` (via `mudm_tools.serve.get_viewer_dir`).
3. Auto-mounts the **2D Leaflet viewer at `/2d/`** when `--viewer` is `3d`
   (the default) **and** a custom `--viewer-dir` was *not* supplied — best-effort,
   silently skipped if the 2D viewer directory is missing.
4. Prints a short startup banner and blocks on `serve_forever()` until
   <kbd>Ctrl</kbd>+<kbd>C</kbd>.

The startup banner looks like this:

```text
Compression: Brotli
Viewer:      http://localhost:8080 (3d)
2D Viewer:   http://localhost:8080/2d/
Tiles:       /abs/path/to/output
2D Tiles:    /abs/path/to/output2d
Press Ctrl+C to stop
```

The first line reads `gzip (install 'brotli' for better compression)` when the
optional `brotli` package is not installed. The `2D Viewer:` and `2D Tiles:`
lines appear only when the 2D viewer is mounted and `--tiles2d-base` is set,
respectively.

!!! note "Exit code"
    `main()` returns `0` on a clean shutdown and `1` if the selected viewer
    directory has no `index.html` (it prints `ERROR: No index.html in <dir>`
    and exits before binding the port).

### Routes

`mudm-serve` maps request paths to files on disk. Query strings and fragments
are stripped before matching.

| Request path | Serves |
| --- | --- |
| `/` | `<viewer_dir>/index.html` (the primary viewer). |
| `/<asset>` | `<viewer_dir>/<asset>` (viewer JS/CSS/etc.). |
| `/tiles/pyramids.json` | `<tiles_base>/pyramids.json` (the pyramid manifest). |
| `/tiles/<pyramid_id>/<rest>` | First tries `<tiles_base>/<pyramid_id>/<rest>` (e.g. `features.json`, `tileset.json`, `tilejson3d.json`); if that is not a file, falls back to `<tiles_base>/<pyramid_id>/3dtiles/<rest>` (the GLB tile tree). |
| `/neuroglancer/<pyramid_id>/<rest>` | `<tiles_base>/<pyramid_id>/neuroglancer/<rest>` (Neuroglancer precomputed). |
| `/2d` or `/2d/` | `<viewer_2d_dir>/index.html` (only when the 2D viewer is mounted). |
| `/2d/<rel>` | `<viewer_2d_dir>/<rel>` (only when the asset exists). |
| `/tiles2d/datasets.json` | A JSON array synthesized on the fly (see below). |
| `/tiles2d/<rel>` | `<tiles2d_base>/<rel>` if the file exists — raster `{z}/{x}/{y}.png`, vectors `{z}/{x}/{y}.pbf`, `metadata.json`, `gene_list.json`, `gene_colormap.json`. |

`GET /tiles2d/datasets.json` is not read from disk — the server iterates the
sorted subdirectories of `--tiles2d-base` and, for each one that contains a
`metadata.json`, emits `{"id": <dirname>, "name": <metadata.name or dirname>}`.

### GLB compression

Any request whose path ends in `.glb` is compressed on the fly and cached:

- If `Accept-Encoding` contains `br` **and** the optional `brotli` package is
  installed → `Content-Encoding: br` via `brotli.compress(quality=5)`.
- Otherwise, if `Accept-Encoding` contains `gzip` →
  `Content-Encoding: gzip` via `gzip.compress(compresslevel=6)`.
- Otherwise the raw GLB is served by the standard handler.

Compressed bytes are cached in a process-global dict keyed by
`(file_path, "br" | "gzip")`, capped at **512 MB total**
(`_CACHE_MAX_BYTES = 512 * 1024 * 1024`). Compressed GLB responses carry
`Content-Type: model/gltf-binary`, `Vary: Accept-Encoding`,
`Cache-Control: public, max-age=86400`, and `Access-Control-Allow-Origin: *`.

!!! tip "Install `brotli` for smaller transfers"
    `brotli` is an optional dependency. Without it the server falls back to gzip
    and prints a hint at startup. Add it with `uv add brotli` (or
    `pip install brotli`) to enable the higher-ratio path.

!!! note "CORS and caching for every response"
    The handler always adds `Access-Control-Allow-Origin: *`. `.glb` responses
    get `Cache-Control: public, max-age=86400`; `.js`/`.html`/`.json`/`.css`
    responses get `Cache-Control: no-cache`.

### Neuroglancer

The `/neuroglancer/<pyramid_id>/<rest>` route is a **server-only passthrough** to
`<tiles_base>/<pyramid_id>/neuroglancer/`. Point an **external** Neuroglancer
client at it — neither bundled viewer (3D or 2D) contains Neuroglancer client
code, so opening the route in the bundled viewer will not render anything. See
the [Neuroglancer guide](../guides/neuroglancer.md) for exporting precomputed
output and wiring up a client.

### Bundled viewers

Viewer assets are packaged inside the wheel at
`src/mudm_tools/viewers/viewer3d/` (Three.js) and
`src/mudm_tools/viewers/viewer2d/` (Leaflet), and are resolved at runtime by
`mudm_tools.serve.get_viewer_dir("3d" | "2d")`.

| Viewer | Served at | Engine | Feature list |
| --- | --- | --- | --- |
| 3D Tiles | `/` when `--viewer 3d` (default) | Three.js 0.171.0 (Draco + meshopt) | [3D Tiling guide](../guides/3d-tiling.md) |
| 2D MVT + raster | `/` when `--viewer 2d`; also `/2d/` when `--viewer 3d` | Leaflet 1.9.4 + VectorGrid | [2D Tiling guide](../guides/2d-tiling.md) |

!!! warning "Both viewers fetch their JS engine from a CDN"
    The 3D viewer loads three.js, the Draco decoder, and the meshopt decoder
    from CDNs at runtime; the 2D viewer loads Leaflet from unpkg (only
    `Leaflet.VectorGrid` is vendored locally). Both bundled viewers therefore
    require internet access to boot. Plan accordingly for air-gapped deployments.

### Examples

Serve 3D Tiles with the default 3D viewer (and the 2D viewer auto-mounted at
`/2d/`):

```bash
uv run mudm-serve --tiles-base output/
```

Serve the 2D viewer at `/` instead, with a separate 2D dataset root:

```bash
uv run mudm-serve \
    --tiles-base output/ \
    --tiles2d-base output2d/ \
    --viewer 2d
```

Serve 3D Tiles **and** 2D datasets together — 3D at `/`, 2D at `/2d/`:

```bash
uv run mudm-serve \
    --tiles-base output/ \
    --tiles2d-base output2d/ \
    --port 9000
```

Serve your own viewer build (this disables the `/2d/` auto-mount):

```bash
uv run mudm-serve \
    --tiles-base output/ \
    --viewer-dir ./my-viewer
```

### Expected directory layout

```text
# --tiles-base (3D)
<tiles_base>/
├── pyramids.json                     # manifest: { "pyramids": [ {id, label, feature_count, ...}, ... ] }
└── <pyramid_id>/
    ├── tileset.json                  # OGC 3D Tiles tileset
    ├── tilejson3d.json               # optional: { maxzoom, id_fields, encodings }
    ├── features.json                 # MicroJSON FeatureCollection (or legacy features map)
    ├── 3dtiles/                      # GLB tiles, served via the /tiles/<id>/<rest> fallback
    │   └── {z}/{x}/{y}/{d}.glb       # Draco- or meshopt-compressed GLB
    └── neuroglancer/                 # served via /neuroglancer/<id>/<rest>

# --tiles2d-base (2D)
<tiles2d_base>/
└── <dataset_id>/
    ├── metadata.json                 # {name, um_per_px, bounds_um, raster{}, vectors{}}
    ├── raster/{z}/{x}/{y}.png        # DAPI raster pyramid
    ├── vectors/{z}/{x}/{y}.pbf       # multi-layer MVT (path from metadata.vectors.path)
    ├── gene_list.json                # optional: gene-filter dropdown values
    └── gene_colormap.json            # optional: gene category colors
# datasets.json is synthesized on the fly at GET /tiles2d/datasets.json
```

See [Reference → TileJSON](tilejson.md) for the `pyramids.json` and 2D
`metadata.json` schemas, and the [legacy pipeline guide](../guides/legacy-pipeline.md)
for how older output trees map onto these routes.

---

## Converter CLI

The converter CLI tiles source data (GeoJSON, OBJ, or Xenium) into a muDM output
tree. It is **not** an installed script — run it as a module:

=== "uv run"

    ```bash
    uv run python -m mudm_tools.converters.cli convert \
        --format geojson \
        --input data.geojson \
        --output out/
    ```

=== "python"

    ```bash
    python -m mudm_tools.converters.cli convert \
        --format geojson \
        --input data.geojson \
        --output out/
    ```

!!! danger "There is no `mudm convert`"
    The argparse program name is `mudm` and the in-code docstrings show
    `mudm convert ...`, but **no `mudm` console script is installed** — only
    `mudm-serve` is. Always use
    `python -m mudm_tools.converters.cli ...`. Running the module with no
    subcommand just prints help.

### Subcommand: `convert`

```text
python -m mudm_tools.converters.cli convert
    --format {geojson,obj,xenium}
    --input  PATH
    --output PATH
    [--config FILE.json]
    [--temp-dir DIR]
    [--max-zoom N]
```

| Flag | Alias | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `--format` | `-f` | yes | — | Source format: `geojson`, `obj`, or `xenium`. |
| `--input` | `-i` | yes | — | Path to the source data directory or file. |
| `--output` | `-o` | yes | — | Path for the tiled output (created with parents). |
| `--config` | `-c` | no | `None` | Path to a JSON file of converter-specific settings. |
| `--temp-dir` | — | no | `None` | Temp dir for fragments; injected into the config as `temp_dir` when set. |
| `--max-zoom` | — | no | `None` | Max zoom level; injected into the config as `max_zoom` when set. |

!!! note "Flag precedence"
    The `--config` JSON file is loaded first, then `--temp-dir` and `--max-zoom`
    **overwrite** the matching keys. Every other converter option (e.g.
    `bounds`, `layer_name`, `glob`, `min_zoom`, `generate_parquet`) must come
    from the `--config` file — see the [converters guide](../guides/converters.md)
    for the full per-format config key tables.

On success the command prints the converter's result dict as indented JSON:

```text
Result: {
  "total_time": 1.84,
  "feature_count": 412,
  "timings": { "ingest": 0.31, "pbf": 1.20, "parquet": 0.33 }
}
```

!!! info "Result shape depends on the format"
    Only the **xenium** converter returns `layer_counts` and `tile_count`. The
    **obj** and **geojson** converters return `feature_count` instead. All three
    always return `total_time` (seconds) and a `timings` dict.

### Subcommand: `list-formats`

Prints each registered converter format, one per indented line.

```bash
uv run python -m mudm_tools.converters.cli list-formats
```

```text
  geojson
  obj
  xenium
```

### Examples

Tile a directory of OBJ meshes into 3D Tiles plus Parquet, overriding the octree
depth:

```bash
uv run python -m mudm_tools.converters.cli convert \
    --format obj \
    --input meshes/ \
    --output out/ \
    --max-zoom 5
```

Convert a 10x Genomics Xenium bundle, using a config file for the
Xenium-specific options and a dedicated scratch dir:

```bash
uv run python -m mudm_tools.converters.cli convert \
    --format xenium \
    --input xenium_bundle/ \
    --output out/ \
    --config xenium.json \
    --temp-dir /scratch/mudm
```

Where `xenium.json` might contain:

```json
{
  "id_column": "cell_id",
  "point_zoom_offset": 3,
  "skip_raster": false
}
```

!!! warning "GeoJSON: pass `bounds` explicitly"
    The GeoJSON converter's automatic bounds computation is currently a stub
    that falls back to `(0, 0, 1, 1)`. For correct tiling, supply `bounds`
    (a 4-tuple `xmin, ymin, xmax, ymax`) via the `--config` file. OBJ `bounds`
    are a 6-tuple `xmin, ymin, zmin, xmax, ymax, zmax`. See the
    [converters guide](../guides/converters.md) for details.

After converting, serve the result with `mudm-serve` (above) — point
`--tiles-base` at an OBJ/3D output root, or `--tiles2d-base` at a parent of
GeoJSON/Xenium dataset directories.

---

## See also

- [Converters guide](../guides/converters.md) — full per-format config keys, output layouts, and the Python `convert()` / `list_formats()` API.
- [3D Tiling guide](../guides/3d-tiling.md) — the Three.js viewer's full feature set.
- [2D Tiling guide](../guides/2d-tiling.md) — the Leaflet viewer's full feature set.
- [Neuroglancer guide](../guides/neuroglancer.md) — exporting and serving precomputed output.
- [Python API reference](python-api.md) — `mudm_tools.serve` and `mudm_tools.converters` programmatic interfaces.

The serve entry point and viewer resolver are documented below.

::: mudm_tools.serve.main
    options:
      show_root_heading: true

::: mudm_tools.serve.get_viewer_dir
    options:
      show_root_heading: true

::: mudm_tools.converters.cli
    options:
      show_root_heading: true
