"""Label-mask (3D segmentation volume) -> muDM 3D-tiles mesh pyramid.

Every integer label becomes one closed surface: marching cubes on that label's sub-volume, with an
optional Gaussian pre-smooth that turns the hard 0/1 voxel indicator into a gradient so the isosurface
stops snapping to voxel faces (removes the blocky "staircase" without changing measured geometry).
By default it writes one OBJ per label and tiles them through the streaming Rust ``obj`` converter
(bounded memory, all cores, meshopt-compressed — the same engine that tiled the 139k-neuron FlyWire
connectome); pass ``streaming=False`` for the legacy in-memory Python ``TileGenerator3D`` path. Either
way it emits an OGC 3D-Tiles pyramid under ``3dtiles/``; ``features.json`` / ``tilejson3d.json`` come
from the separate ``build_feature_index`` step (as for every 3D dataset), so custom per-label
attributes must be relaxed through that index's property filter.

Optional per-label ``properties`` (viewer color-by / filter / hover) and ``color`` are supplied via
config; the converter itself only knows geometry. Needs the ``labelmask`` extra (scikit-image + tifffile).
"""

from __future__ import annotations

import colorsys
from pathlib import Path
from typing import Any

from . import register


def distinct_color(i: int) -> str:
    """A well-spread color per integer id (golden-angle hue) so objects are visually separable in the
    viewer's default 'Original' mode; color-by-feature overrides it dynamically."""
    r, g, b = colorsys.hls_to_rgb((i * 0.6180339887) % 1.0, 0.55, 0.62)
    return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))


def label_mesh(mask, label, slices, spacing, step_size=1, smooth_sigma=0.0):
    """Marching-cubes surface of one label. Returns ``(verts_xyz global, faces)`` in ``spacing`` units.

    ``slices`` = ``scipy.ndimage.find_objects(mask)`` (one pass, reused across labels). ``smooth_sigma``
    > 0 Gaussian-blurs the per-label indicator before marching cubes, so the isosurface follows a smooth
    gradient instead of voxel faces. ``step_size`` > 1 subsamples the MC grid (lighter mesh); a coarse
    stride that misses a small object's surface falls back to full resolution, then to an empty mesh
    (the caller skips it) rather than raising. ``marching_cubes`` yields ``(z, y, x)`` verts; we un-pad,
    add the global voxel offset, reorder columns to ``(x, y, z)`` and return a C-contiguous buffer
    (raw-buffer GLB/Draco/meshopt encoders reject negative-stride views).
    """
    import numpy as np
    from scipy import ndimage as ndi
    from skimage.measure import marching_cubes

    sl = slices[label - 1]
    if sl is None:
        raise ValueError(f"label {label} is absent from the mask (no bounding box)")
    starts = np.array([sl[0].start, sl[1].start, sl[2].start])
    sigma = float(smooth_sigma or 0.0)
    for ss in [int(step_size), 1] if int(step_size) > 1 else [1]:
        pad = max(1, ss, int(np.ceil(sigma)))
        vol = np.pad(mask[sl] == label, pad).astype(np.float32)
        if sigma > 0:
            vol = ndi.gaussian_filter(vol, sigma=sigma)
        try:
            verts, faces, _n, _v = marching_cubes(vol, level=0.5, spacing=spacing, step_size=ss)
        except (RuntimeError, ValueError):
            continue
        verts = verts + (starts - pad) * np.array(spacing)  # (z, y, x) global
        return np.ascontiguousarray(verts[:, ::-1]), faces  # -> (x, y, z)
    return np.zeros((0, 3), float), np.zeros((0, 3), int)


def labelmask_to_features(
    mask,
    *,
    spacing,
    step_size: int = 1,
    smooth_sigma: float = 0.0,
    properties: dict | None = None,
    color: Any = "distinct",
    name_prefix: str = "object",
):
    """Build a MuDM ``FeatureCollection`` of per-label TIN surfaces.

    ``properties``: optional ``{label: {prop: value}}`` merged into each feature (color-by / hover).
    ``color``: ``"distinct"`` (golden-angle per label), a ``{label: "#rrggbb"}`` map, or ``None``.
    Each feature carries ``name`` = ``"<name_prefix>-<label>"`` (required by ``build_feature_index``,
    which only harvests glTF-node extras bearing name/acronym/body_id).
    """
    import numpy as np
    from mudm.model import MuDMFeature, MuDMFeatureCollection
    from scipy import ndimage as ndi

    from ..swc import _mesh_to_tin

    props_map = properties or {}
    slices = ndi.find_objects(mask)
    labels = np.unique(mask)
    labels = labels[labels > 0]
    feats = []
    for raw in labels:
        label = int(raw)
        verts, faces = label_mesh(
            mask, label, slices, spacing, step_size=step_size, smooth_sigma=smooth_sigma
        )
        if len(faces) == 0:
            continue
        props: dict[str, Any] = {"name": f"{name_prefix}-{label}"}
        extra = props_map.get(label, props_map.get(str(label)))
        if extra:
            props.update(extra)
        if color == "distinct":
            props["color"] = distinct_color(label)
        elif isinstance(color, dict):
            chosen = color.get(label, color.get(str(label)))
            if chosen:
                props["color"] = chosen
        feats.append(
            MuDMFeature(type="Feature", geometry=_mesh_to_tin(verts, faces), properties=props)
        )
    return MuDMFeatureCollection(type="FeatureCollection", features=feats)


def _write_obj(path, verts, faces) -> None:
    """Write a triangle mesh to a Wavefront OBJ (1-indexed faces)."""
    import numpy as np

    v = [f"v {x:.6g} {y:.6g} {z:.6g}" for x, y, z in verts]
    f = [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in np.asarray(faces, dtype=np.int64)]
    Path(path).write_text("\n".join(v + f) + "\n")


def labelmask_to_objs(
    mask,
    obj_dir,
    *,
    spacing,
    step_size: int = 1,
    smooth_sigma: float = 0.0,
    properties: dict | None = None,
    color: Any = "distinct",
    name_prefix: str = "object",
):
    """Write one Wavefront OBJ per label (smoothed marching-cubes surface) into ``obj_dir``; return
    ``(tags, bounds)``. ``tags`` maps each file stem -> its feature properties (name/color/attributes);
    ``bounds`` is the ``(xmin, ymin, zmin, xmax, ymax, zmax)`` world box. This feeds the streaming Rust
    ``obj`` converter (bounded memory, all cores, meshopt-compressed) — see ``LabelMaskConverter``.
    """
    import numpy as np
    from scipy import ndimage as ndi

    obj_dir = Path(obj_dir)
    obj_dir.mkdir(parents=True, exist_ok=True)
    props_map = properties or {}
    slices = ndi.find_objects(mask)
    labels = np.unique(mask)
    labels = labels[labels > 0]
    lo = np.array([np.inf, np.inf, np.inf])
    hi = np.array([-np.inf, -np.inf, -np.inf])
    tags: dict[str, dict] = {}
    for raw in labels:
        label = int(raw)
        verts, faces = label_mesh(
            mask, label, slices, spacing, step_size=step_size, smooth_sigma=smooth_sigma
        )
        if len(faces) == 0:
            continue
        name = f"{name_prefix}-{label}"
        _write_obj(obj_dir / f"{name}.obj", verts, faces)
        lo = np.minimum(lo, verts.min(axis=0))
        hi = np.maximum(hi, verts.max(axis=0))
        t: dict[str, Any] = {"name": name}
        extra = props_map.get(label, props_map.get(str(label)))
        if extra:
            t.update(extra)
        if color == "distinct":
            t["color"] = distinct_color(label)
        elif isinstance(color, dict):
            chosen = color.get(label, color.get(str(label)))
            if chosen:
                t["color"] = chosen
        tags[name] = t
    bounds = (float(lo[0]), float(lo[1]), float(lo[2]), float(hi[0]), float(hi[1]), float(hi[2]))
    return tags, bounds


@register("labelmask")
class LabelMaskConverter:
    """Convert a 3D integer label mask into a muDM 3D-tiles mesh pyramid (marching cubes + smoothing)."""

    def convert(self, input_dir: str, output_dir: str, config: dict[str, Any]) -> dict:
        """Tile a label mask into ``<output_dir>/3dtiles/``.

        ``input_dir``: a 3D label-mask image (``.tif`` / ``.tiff`` via tifffile, or ``.npy``).

        Config keys:
            spacing (tuple): ``(z, y, x)`` units per voxel. Default ``(1, 1, 1)``.
            step_size (int): marching-cubes grid stride. Default 1.
            smooth_sigma (float): Gaussian pre-smooth sigma in voxels; 0 disables. Default 1.0.
            max_zoom (int): octree depth. Default 3.
            streaming (bool): DEFAULT True. Write per-label OBJs and tile via the streaming Rust
                ``obj`` converter — bounded memory, all cores, meshopt-compressed (the scalable path,
                same engine as the connectome datasets). False = legacy in-memory Python
                ``TileGenerator3D`` (unbounded RAM; small datasets / tests only).
            compression (str): GLB compression, streaming path. Default "meshopt-q14".
            simplify (bool): per-zoom LOD decimation, streaming path. Default True.
            generate_parquet (bool): also emit features.parquet, streaming path. Default False.
            properties (dict | str): ``{label: {prop: value}}`` or a path to a Parquet keyed by
                ``id_col`` (each row's other columns become that label's properties).
            id_col (str): id column when ``properties`` is a Parquet path. Default "id".
            color (str | dict): "distinct" (default), a ``{label: hex}`` map, or None.
            name_prefix (str): feature name prefix. Default "object".
            workers / max_memory_bytes / output_format: in-memory path only (see TileGenerator3D).

        ``features.json`` / ``tilejson3d.json`` are NOT written here -- run ``build_feature_index``
        afterwards (same as the ``obj`` / connectome pipelines).
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        mask = self._load_mask(Path(input_dir))
        common = dict(
            spacing=tuple(config.get("spacing", (1.0, 1.0, 1.0))),
            step_size=int(config.get("step_size", 1)),
            smooth_sigma=float(config.get("smooth_sigma", 1.0)),
            properties=self._load_properties(config),
            color=config.get("color", "distinct"),
            name_prefix=config.get("name_prefix", "object"),
        )
        if config.get("streaming", True):
            return self._convert_streaming(mask, out, config, common)
        return self._convert_inmemory(mask, out, config, common)

    def _convert_streaming(self, mask, out: Path, config: dict[str, Any], common: dict) -> dict:
        """Write one OBJ per label, then tile through the streaming Rust ``obj`` converter (bounded
        memory, all cores, meshopt-q14) — the scalable path, same engine as the connectome datasets.
        """
        import tempfile

        from mudm_tools.converters import convert as run_convert

        with tempfile.TemporaryDirectory(prefix="labelmask_objs_") as td:
            tags, bounds = labelmask_to_objs(mask, td, **common)
            obj_cfg: dict[str, Any] = {
                "bounds": bounds,
                "tags": tags,
                "glob": "*.obj",
                "max_zoom": int(config.get("max_zoom", 3)),
                "compression": config.get("compression", "meshopt-q14"),
                "simplify": config.get("simplify", True),
                "generate_parquet": config.get("generate_parquet", False),
            }
            if config.get("temp_dir"):
                obj_cfg["temp_dir"] = config["temp_dir"]
            rep = run_convert("obj", td, str(out), obj_cfg)
        rep["features"] = len(tags)
        return rep

    def _convert_inmemory(self, mask, out: Path, config: dict[str, Any], common: dict) -> dict:
        """Legacy in-memory path (Python ``TileGenerator3D``): simpler but holds the whole octree in
        RAM. Kept for small datasets / tests; prefer streaming for large volumes."""
        from ..tiling3d.generator3d import TileGenerator3D
        from ..tiling3d.octree import OctreeConfig

        tiles_dir = out / "3dtiles"
        tiles_dir.mkdir(parents=True, exist_ok=True)
        fc = labelmask_to_features(mask, **common)
        gen = TileGenerator3D(
            OctreeConfig(max_zoom=int(config.get("max_zoom", 3))),
            output_format=config.get("output_format", "3dtiles"),
            workers=config.get("workers"),
            max_memory_bytes=config.get("max_memory_bytes"),
        )
        gen.add_features(fc)
        n_tiles = gen.generate(tiles_dir)
        gen.write_metadata(tiles_dir)
        return {"features": len(fc.features), "tiles": n_tiles}

    def _load_mask(self, path: Path):
        if path.suffix == ".npy":
            import numpy as np

            return np.load(path)
        import tifffile

        return tifffile.imread(str(path))

    def _load_properties(self, config: dict[str, Any]):
        props = config.get("properties")
        if props is None or isinstance(props, dict):
            return props
        import pyarrow.parquet as pq

        id_col = config.get("id_col", "id")
        rows = pq.read_table(str(props)).to_pylist()
        return {int(r[id_col]): {k: v for k, v in r.items() if k != id_col} for r in rows}
