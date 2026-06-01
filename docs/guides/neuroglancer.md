# Neuroglancer Export

This guide shows how to export muDM data to the **Neuroglancer precomputed** format and how to serve it to an external Neuroglancer viewer.

There are **two distinct paths**, and choosing the right one matters:

1. **Python writer subsystem** (`mudm_tools.neuroglancer`) — a toolkit of small writers for **skeletons**, **annotations** (Point / LineString), **legacy single-resolution meshes**, and **segment properties**, plus helpers to build a Neuroglancer **viewer state** and a shareable URL. Use this for neuron morphologies (SWC skeletons) and point/line annotations.
2. **Scalable Rust mesh path** (`mudm_tools._rs.StreamingTileGenerator`) — `generate_neuroglancer` (legacy single-res mesh) and `generate_neuroglancer_multilod` (multi-LOD Draco, with optional sharding). Use this for large mesh corpora that need level-of-detail and object-store-scale deployment. See the [3D Tiling guide](3d-tiling.md) for how to feed the generator.

!!! warning "The bundled viewers do NOT render Neuroglancer"
    muDM ships its own 2D and 3D viewers, but they do **not** render Neuroglancer precomputed data. Neuroglancer export targets an **external** Neuroglancer client (the public demo at `https://neuroglancer-demo.appspot.com`, or your own instance). `mudm-serve` only *serves the files* under a `/neuroglancer/` route; the rendering happens in the external Neuroglancer app.

!!! note "Package names"
    The writers live in `mudm_tools.neuroglancer`. The compiled mesh generator is `mudm_tools._rs.StreamingTileGenerator` (never `mudm._rs`). Geometry types (`MuDMFeature`, `MuDMFeatureCollection`) come from the sibling core package `mudm` (`mudm.model`).

---

## Path 1: The Python writer subsystem

### Quick start: SWC neurons to precomputed skeletons

The fastest way to see something is the bundled `swc_to_neuroglancer` example. It converts one or more SWC files into a precomputed **skeleton** source, attaches segment properties, prints a ready-to-open Neuroglancer URL, and serves the files over a CORS-enabled HTTP server.

```bash
# Convert + serve (CORS server on :9000)
uv run python -m mudm_tools.examples.swc_to_neuroglancer neuron.swc

# Several neurons at once
uv run python -m mudm_tools.examples.swc_to_neuroglancer n1.swc n2.swc n3.swc

# Export only, no server
uv run python -m mudm_tools.examples.swc_to_neuroglancer neuron.swc --no-serve

# Custom output dir + port
uv run python -m mudm_tools.examples.swc_to_neuroglancer neuron.swc -o out -p 8080
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--output-dir`, `-o` | `neuroglancer_output` | Root output directory |
| `--port`, `-p` | `9000` | Port for the CORS HTTP server |
| `--no-serve` | off | Export only, do not start the server |

The example writes skeletons to `<output-dir>/skeletons/`, prints the viewer URL, then serves on `http://localhost:<port>`. Copy the printed URL into your browser.

!!! tip "Where to get SWC files"
    NeuroMorpho.Org hosts 200,000+ reconstructions. Search and download `.swc`, then point the example at the file.

### Writing skeletons directly

`write_skeleton` writes **one** precomputed skeleton segment. Call it repeatedly with distinct `segment_id`s into the **same** directory to build a multi-skeleton source — the `info` file is rewritten on each call.

```python
from pathlib import Path

from mudm_tools.swc import _parse_swc
from mudm_tools.neuroglancer import write_skeleton

morphology = _parse_swc("neuron.swc")  # mudm_tools.swc.NeuronMorphology

write_skeleton(Path("out/skeletons"), segment_id=1, morphology=morphology)
```

The signature:

```python
def write_skeleton(
    output_dir: str | Path,
    segment_id: int,
    morphology: NeuronMorphology,
    *,
    transform: Optional[AffineTransform] = None,
    include_radius: bool = True,
    include_type: bool = True,
    segment_properties: Optional[str] = None,
) -> Path
```

| Parameter | Default | Notes |
| --- | --- | --- |
| `output_dir` | — | Created with parents |
| `segment_id` | — | Binary written to a file named `str(segment_id)` |
| `morphology` | — | A `mudm_tools.swc.NeuronMorphology` |
| `transform` | `None` | A `mudm.transforms.AffineTransform`, embedded as a 12-float row-major upper 3×4 |
| `include_radius` | `True` | Emit a per-vertex `radius` attribute |
| `include_type` | `True` | Emit a per-vertex SWC `type` attribute |
| `segment_properties` | `None` | Relative path to a `segment_properties` dir to reference in `info` |

Edges are derived from each sample's `parent` (parent index → child index).

!!! note "Unit scaling"
    `write_skeleton` builds its `info` **without** the micrometre→nanometre transform. If your coordinates are in µm and you want Neuroglancer to treat them as nm, pass an explicit `transform`, or build the `info` yourself with `build_skeleton_info(scale_um_to_nm=True)` and write it after your skeletons (see the SWC example, which rewrites `info` to attach `segment_properties`).

### Segment properties

`write_segment_properties` writes a `neuroglancer_segment_properties` source from each feature's `.properties` dict. Numeric columns become type `number` (`uint32` if all values are ints, else `float32`); everything else becomes a `label`. Values are ordered to match `segment_ids`; missing values become an empty string.

```python
from mudm_tools.neuroglancer.properties_writer import write_segment_properties

write_segment_properties(
    "out/skeletons/seg_props",
    features=features,         # Sequence[MuDMFeature]
    segment_ids=[1, 2, 3],     # one id per feature
)
```

Use `features_to_segment_properties(features, segment_ids)` if you want the info **dict** without writing it.

To wire the property source into the skeleton `info`, rebuild and rewrite `info` with a relative path:

```python
import json
from mudm_tools.neuroglancer.skeleton_writer import build_skeleton_info

info = build_skeleton_info(segment_properties="seg_props")
(skel_dir / "info").write_text(json.dumps(info.to_info_dict(), indent=2))
```

### Annotations: `to_neuroglancer` and `write_annotations`

!!! danger "`to_neuroglancer` is NOT a one-call full exporter"
    `to_neuroglancer` only handles **Point** features (→ `point_annotations/` subdir) and **LineString** features (→ `line_annotations/` subdir). It does **not** export skeletons or meshes — for those, call `write_skeleton` or `write_mesh` directly. Features whose geometry is neither Point nor LineString are silently ignored.

```python
from mudm_tools.neuroglancer import to_neuroglancer

result = to_neuroglancer(collection, "out/mixed")
# result["paths"] -> {"point_annotations": <dir>, "line_annotations": <dir>}
```

Signature:

```python
def to_neuroglancer(
    data: Union[MuDMFeature, MuDMFeatureCollection],
    output_dir: str | Path,
    *,
    base_url: Optional[str] = None,
) -> dict[str, Any]
```

`to_neuroglancer` always returns `{"paths": {layer_name: dir}}`. If you pass `base_url` **and** at least one annotation layer was written, it also returns `"viewer_state"` (a Neuroglancer state dict) and `"viewer_url"` (an encoded URL).

!!! warning "Two different `base_url`s"
    `to_neuroglancer`'s `base_url` is the precomputed **data source** base. Annotation layers are added as `precomputed://{base_url}/point_annotations` and `precomputed://{base_url}/line_annotations`. This is **distinct** from `viewer_state_to_url`'s `base_url`, which is the Neuroglancer **viewer instance** (default `https://neuroglancer-demo.appspot.com`).

For finer control, call `write_annotations` directly:

```python
def write_annotations(
    output_dir: str | Path,
    features: Sequence[MuDMFeature],
    annotation_type: Literal["point", "line"],
) -> Path
```

This produces `{output_dir}/info` (`@type` `neuroglancer_annotations_v1`), an empty `{output_dir}/by_id/` dir, and a single spatial chunk `{output_dir}/spatial0/0_0_0`.

!!! note "Line annotation semantics"
    For `annotation_type="line"`, each consecutive pair of `LineString` coordinates becomes one **LINE** annotation — an N-vertex line yields N−1 segments, all sharing the feature's `enumerate()` index (0-based) as their annotation id. Annotation ids are the feature index, **not** a property. Missing `z` defaults to `0.0`.

A complete in-memory example (adapted from `neuroglancer_export.py`):

```python
from pathlib import Path
from mudm.model import MuDMFeature, MuDMFeatureCollection
from mudm_tools.neuroglancer import to_neuroglancer, write_skeleton

out = Path("out/mixed")

# Skeletons are written directly (NOT via to_neuroglancer)
write_skeleton(out / "skeletons", segment_id=1, morphology=neuron_a)
write_skeleton(out / "skeletons", segment_id=2, morphology=neuron_b)

collection = MuDMFeatureCollection(
    type="FeatureCollection",
    features=[
        MuDMFeature(
            type="Feature",
            geometry={"type": "Point", "coordinates": [500, 500, 200]},
            properties={"label": "pyramidal_soma"},
        ),
        MuDMFeature(
            type="Feature",
            geometry={
                "type": "LineString",
                "coordinates": [[500, 500, 200], [650, 550, 200], [800, 600, 200]],
            },
            properties={"label": "connecting_fiber"},
        ),
    ],
)

result = to_neuroglancer(collection, out)
# -> writes out/point_annotations/ and out/line_annotations/
```

Run all three patterns from the bundled example:

```bash
uv run python -m mudm_tools.examples.neuroglancer_export --output-dir neuroglancer_output
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--output-dir` | `neuroglancer_output` | Root output directory |

### Legacy single-resolution meshes

The Python mesh writer emits the legacy single-LOD mesh format (`@type` `neuroglancer_legacy_mesh`). Write the `info` once with `write_mesh_info`, then one segment at a time with `write_mesh`.

```python
import numpy as np
from mudm_tools.neuroglancer import write_mesh, write_mesh_info

write_mesh_info("out/meshes", segment_properties="segment_properties")
write_mesh("out/meshes", segment_id=1, vertices=verts, indices=faces)
```

```python
def write_mesh_info(output_dir: str | Path, segment_properties: Optional[str] = None) -> Path
def write_mesh(output_dir: str | Path, segment_id: int, vertices: np.ndarray, indices: np.ndarray) -> Path
```

`write_mesh` writes `{output_dir}/{segment_id}` (binary, via `mesh_to_binary`) plus `{output_dir}/{segment_id}:0` (a JSON fragment manifest `{"fragments": ["{segment_id}"]}`). It does **not** write `info` — call `write_mesh_info` separately. `vertices` are Nx3 float32 and `indices` are Mx3 uint32.

!!! tip "Choosing a mesh path"
    `write_mesh` is the **legacy single-resolution** path. For large meshes that need genuine level-of-detail, use the scalable Rust [`generate_neuroglancer_multilod`](#multi-lod-draco-meshes-generate_neuroglancer_multilod) instead. Do not conflate the two — they emit different `@type`s and different on-disk layouts.

Low-level binary codecs are also exported: `mesh_to_binary(vertices, indices) -> bytes` and its inverse `decode_mesh_binary(data) -> (vertices, indices)`. For point/line annotation bytes, see `points_to_annotation_binary` and `lines_to_annotation_binary`.

### Building a viewer state and URL

The viewer-state helpers in `mudm_tools.neuroglancer.state` build a plain JSON state dict and a shareable fragment URL using only `json` + `urllib.parse`. They do **not** depend on the official `neuroglancer` Python package and do **not** run a viewer server.

```python
from mudm_tools.neuroglancer.state import (
    build_skeleton_layer,
    build_annotation_layer,
    build_viewer_state,
    viewer_state_to_url,
)

layer = build_skeleton_layer(
    "neurons",
    "precomputed://http://localhost:9000/skeletons",
)
layer["segments"] = ["1", "2"]  # pre-select segments so they render immediately

state = build_viewer_state([layer], position=[650.0, 550.0, 200.0])
url = viewer_state_to_url(state)
print(url)  # https://neuroglancer-demo.appspot.com/#!<encoded-state>
```

| Function | Signature | Purpose |
| --- | --- | --- |
| `build_skeleton_layer` | `(name, source_url, *, use_radius=True)` | A `segmentation` layer for a skeleton source. With `use_radius=True`, attaches a GLSL `skeletonRendering` shader mapping the `radius` attribute to line width. |
| `build_annotation_layer` | `(name, source_url)` | An `annotation` layer pointing at a `precomputed://` annotation source. |
| `build_viewer_state` | `(layers, position=None, projection_scale=None, layout="3d")` | A full viewer-state dict. `position` sets the camera, `projection_scale` the `zoomFactor`, `layout` defaults to `"3d"` (use `"4panel"` for all views). |
| `viewer_state_to_url` | `(state, base_url="https://neuroglancer-demo.appspot.com")` | Encodes the state as `{base_url}/#!{encoded}`. |

!!! note "Pre-selecting segments"
    `build_skeleton_layer` returns a plain dict; callers commonly set `layer["segments"] = [...]` afterward so the listed segments are visible on load.

### Python writer API reference

::: mudm_tools.to_neuroglancer
    options:
      show_root_heading: true

::: mudm_tools.write_annotations
    options:
      show_root_heading: true

::: mudm_tools.neuroglancer.write_skeleton
    options:
      show_root_heading: true

::: mudm_tools.neuroglancer.build_skeleton_info
    options:
      show_root_heading: true

::: mudm_tools.neuroglancer.neuron_to_skeleton_binary
    options:
      show_root_heading: true

::: mudm_tools.neuroglancer.affine_to_ng_transform
    options:
      show_root_heading: true

::: mudm_tools.neuroglancer.write_segment_properties
    options:
      show_root_heading: true

::: mudm_tools.neuroglancer.features_to_segment_properties
    options:
      show_root_heading: true

::: mudm_tools.neuroglancer.write_mesh
    options:
      show_root_heading: true

::: mudm_tools.neuroglancer.write_mesh_info
    options:
      show_root_heading: true

::: mudm_tools.neuroglancer.mesh_to_binary
    options:
      show_root_heading: true

::: mudm_tools.neuroglancer.decode_mesh_binary
    options:
      show_root_heading: true

::: mudm_tools.neuroglancer.fragments_to_mesh
    options:
      show_root_heading: true

::: mudm_tools.neuroglancer.points_to_annotation_binary
    options:
      show_root_heading: true

::: mudm_tools.neuroglancer.lines_to_annotation_binary
    options:
      show_root_heading: true

::: mudm_tools.neuroglancer.build_skeleton_layer
    options:
      show_root_heading: true

::: mudm_tools.neuroglancer.build_annotation_layer
    options:
      show_root_heading: true

::: mudm_tools.neuroglancer.build_viewer_state
    options:
      show_root_heading: true

::: mudm_tools.neuroglancer.viewer_state_to_url
    options:
      show_root_heading: true

::: mudm_tools.neuroglancer.MeshInfo
    options:
      show_root_heading: true

::: mudm_tools.neuroglancer.models.SkeletonInfo
    options:
      show_root_heading: true

::: mudm_tools.neuroglancer.models.VertexAttributeInfo
    options:
      show_root_heading: true

::: mudm_tools.neuroglancer.models.AnnotationInfo
    options:
      show_root_heading: true

::: mudm_tools.neuroglancer.models.SegmentPropertiesInfo
    options:
      show_root_heading: true

For the rest of the model classes (`AnnotationDimension`, `AnnotationSpatialEntry`, `AnnotationPropertySpec`, `AnnotationRelationship`, `SegmentPropertiesInline`, `SegmentPropertyField`), see the [Python API reference](../reference/python-api.md).

---

## Path 2: The scalable Rust mesh generator

For large mesh corpora, drive `mudm_tools._rs.StreamingTileGenerator`. You ingest geometry (OBJ files, Parquet meshes, or projected feature dicts), the generator accumulates octree-clipped fragments on disk, and a `generate_*` call emits the chosen Neuroglancer format. See the [3D Tiling guide](3d-tiling.md) for ingestion details.

!!! warning "Not autodoc-able"
    `StreamingTileGenerator` is a compiled Rust pyclass (`mudm_tools._rs`), so its methods cannot be introspected by mkdocstrings. The signatures below are transcribed by hand from the source.

### Legacy mesh: `generate_neuroglancer`

```python
def generate_neuroglancer(
    self,
    output_dir: str,
    world_bounds: tuple[float, float, float, float, float, float],  # xmin,ymin,zmin,xmax,ymax,zmax
) -> int
```

Emits a segment-centric `neuroglancer_legacy_mesh` source: one mesh per feature at `max_zoom` (the finest level). It writes `output_dir/info` (`@type` `neuroglancer_legacy_mesh`), a `{segment_id}` binary mesh, a `{segment_id}:0` JSON fragment manifest, and `segment_properties/info` built from tags. Geometry is merged with float32-bit vertex dedup. Returns the number of segments written.

### Multi-LOD Draco meshes: `generate_neuroglancer_multilod`

```python
def generate_neuroglancer_multilod(
    self,
    output_dir: str,
    world_bounds: tuple[float, float, float, float, float, float],
    vertex_quantization_bits: int = 10,
    max_memory_bytes: int = 0,
    sharded: bool = False,
    minishard_bits: int = 6,
    shard_bits: int = 0,
) -> int
```

Emits a `neuroglancer_multilod_draco` source. Unlike the legacy path, this uses **all** zoom levels: LOD 0 = `max_zoom` (finest), LOD N = zoom 0 (coarsest). Fragment positions are sorted in Morton / Z-curve order. The `info` carries `vertex_quantization_bits`, an identity transform, `lod_scale_multiplier` 1.0, and a `segment_properties` reference. Returns the number of segments written.

| Parameter | Default | Notes |
| --- | --- | --- |
| `output_dir` | — | Output directory |
| `world_bounds` | — | `(xmin, ymin, zmin, xmax, ymax, zmax)`; degenerate axes default span to 1.0 |
| `vertex_quantization_bits` | `10` | Per-tile position quantization (`qmax = 2^bits − 1`) |
| `max_memory_bytes` | `0` | Per-path memory ceiling. `0` uses the generator's resolved `self.max_memory_bytes`. The ceiling derives a feature-bucket count `k = min(ceil(est_resident_bytes / budget), 256)`; peak resident ≈ corpus/`k`. A huge ceiling yields `k == 1` (whole-corpus, byte-identical path). |
| `sharded` | `False` | When `True`, emit `neuroglancer_uint64_sharded_v1` `.shard` files (see below) |
| `minishard_bits` | `6` | Sharding spec `minishard_bits` (used only when `sharded=True`) |
| `shard_bits` | `0` | Sharding spec `shard_bits` (used only when `sharded=True`) |

```python
from mudm_tools._rs import StreamingTileGenerator, scan_obj_bounds

paths = ["mesh_a.obj", "mesh_b.obj"]
bounds = scan_obj_bounds(paths)

gen = StreamingTileGenerator(min_zoom=0, max_zoom=4)
gen.add_obj_files(paths, bounds, [{"name": "a"}, {"name": "b"}])

# Loose multi-LOD output (default)
n = gen.generate_neuroglancer_multilod("out/ng_multilod", bounds)
print(f"{n} segments written")
```

#### Loose vs. sharded layout

By default (`sharded=False`), the generator writes **loose** per-segment files: a `{seg_id}.index` binary manifest plus a `{seg_id}` file of concatenated Draco fragments. This read is feature-bucketed and memory-bounded.

With `sharded=True`, per-segment (manifest, fragment bytes) pairs are accumulated and packed into `neuroglancer_uint64_sharded_v1` `.shard` files. The `info` file gains a `sharding` block (hash `murmurhash3_x86_128`, `minishard_index_encoding` `raw`, `data_encoding` `raw`). Loose per-segment files are **not** written in sharded mode.

!!! danger "Sharded mode holds the whole corpus in RAM"
    Sharded packing happens **after** the bucket loop, so `sharded=True` does **not** preserve the per-bucket memory bound — it holds the entire Neuroglancer mesh corpus in memory. Treat it as an opt-in deploy / object-store-scale path, and provision RAM accordingly. The default loose format keeps the memory bound.

=== "Loose (default)"

    ```python
    gen.generate_neuroglancer_multilod("out/ng", world_bounds)
    ```

    ```text
    out/ng/
      info                    # @type neuroglancer_multilod_draco
      {seg_id}.index          # per-segment binary manifest
      {seg_id}                # concatenated Draco fragments
      segment_properties/info
    ```

=== "Sharded"

    ```python
    gen.generate_neuroglancer_multilod(
        "out/ng",
        world_bounds,
        sharded=True,
        minishard_bits=6,
        shard_bits=0,
    )
    ```

    ```text
    out/ng/
      info                    # @type neuroglancer_multilod_draco (+ "sharding" block)
      *.shard                 # neuroglancer_uint64_sharded_v1 files
      segment_properties/info
    ```

---

## Output structure reference

```text
# write_skeleton (repeated per segment_id into one dir)
{output_dir}/
  info                 # JSON, @type neuroglancer_skeletons
  {segment_id}         # u32 nverts, u32 nedges, f32 verts[N*3], u32 edges[E*2], [f32 radii[N]], [f32 types[N]]
  seg_props/info       # optional, from write_segment_properties

# write_annotations / to_neuroglancer
{output_dir}/          # to_neuroglancer makes point_annotations/ and/or line_annotations/ subdirs
  info                 # JSON, @type neuroglancer_annotations_v1
  by_id/               # empty dir (required by Neuroglancer)
  spatial0/0_0_0       # u64 count, f32 coords[count*D] (D=3 point, 6 line), u64 ids[count]

# write_mesh_info + write_mesh (legacy_mesh)
{output_dir}/
  info                 # JSON, @type neuroglancer_legacy_mesh
  {segment_id}         # u32 nverts, f32 verts[N*3], u32 idx[M*3]
  {segment_id}:0       # JSON fragment manifest {"fragments": ["{segment_id}"]}
  segment_properties/info   # optional

# StreamingTileGenerator.generate_neuroglancer_multilod (multilod_draco)
{output_dir}/
  info                 # JSON, @type neuroglancer_multilod_draco (+ "sharding" when sharded=True)
  {seg_id}.index       # loose mode only: binary manifest
  {seg_id}             # loose mode only: concatenated Draco fragments
  *.shard              # sharded=True only (neuroglancer_uint64_sharded_v1)
  segment_properties/info
```

---

## Serving for an external Neuroglancer client

Neuroglancer runs in the browser and fetches data over HTTP, so the server **must** send CORS headers (`Access-Control-Allow-Origin: *`). You have two options.

### Option A: `mudm-serve` (`/neuroglancer/` route)

The installed console script `mudm-serve` (`mudm_tools.serve:main`) serves precomputed files under a `/neuroglancer/` route. A request to `/neuroglancer/{pyramid_id}/...` maps to `{tiles_base}/{pyramid_id}/neuroglancer/...` on disk, where `tiles_base` is the tiles directory you serve.

```bash
uv run mudm-serve
```

This is intended for an **external** Neuroglancer client — point a `precomputed://` source at the served URL. See the [CLI reference](../reference/cli.md) for the full `mudm-serve` options.

!!! warning "External viewer only"
    `mudm-serve` does not render Neuroglancer. Open the served `precomputed://` source in a real Neuroglancer instance.

### Option B: the standalone example servers

Both ship as runnable example modules and use Python's `http.server` with a CORS subclass.

=== "Serve an exported dir"

    ```bash
    # Auto-detect skeleton/annotation layers and print a viewer URL per export
    uv run python -m mudm_tools.examples.neuroglancer_serve neuroglancer_output
    ```

    `neuroglancer_serve` walks each subdirectory, reads its `info` `@type` to classify it (`neuroglancer_skeletons` → `segmentation`, `neuroglancer_annotations_v1` → `annotation`), infers segment ids from the binary files, builds a viewer state, and prints a ready-to-open Neuroglancer URL.

    | Flag | Default | Meaning |
    | --- | --- | --- |
    | `directory` | — | Directory of precomputed data (positional) |
    | `--port` | `9000` | Port to serve on |
    | `--neuroglancer-url` | `https://neuroglancer-demo.appspot.com` | Neuroglancer instance for the printed URLs |

=== "Convert + serve in one step"

    ```bash
    uv run python -m mudm_tools.examples.swc_to_neuroglancer neuron.swc
    ```

    `swc_to_neuroglancer` converts, prints the URL, and serves in a single command (drop `--no-serve` to keep the server running).

---

## See also

- [3D Tiling](3d-tiling.md) — how to ingest geometry into `StreamingTileGenerator` before calling `generate_neuroglancer` / `generate_neuroglancer_multilod`.
- [CLI reference](../reference/cli.md) — `mudm-serve` options and the `/neuroglancer/` route.
- [Python API reference](../reference/python-api.md) — the full `mudm_tools.neuroglancer` writer and model API.
