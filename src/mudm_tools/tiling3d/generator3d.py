"""TileGenerator3D — high-level API for 3D vector tile generation.

Orchestrates the full pipeline: convert → index (octree) → transform →
encode → write to disk as ``{z}/{x}/{y}/{d}.pbf3`` (or ``.glb``) files.

Supports two output formats:
- ``"pbf3"`` (default): protobuf-encoded 3D vector tiles + tilejson3d.json
- ``"3dtiles"``: OGC 3D Tiles 1.1 with glTF/GLB content + tileset.json
"""

from __future__ import annotations

import multiprocessing
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Literal, cast

from mudm.model import MuDMFeatureCollection, Vocabulary
from .convert3d import compute_bounds_3d, convert_collection_3d
from .encoder3d import encode_tile_3d
from .octree import Octree, OctreeConfig
from .projector3d import CartesianProjector3D
from .tile3d import transform_tile_3d
from .tilejson3d import TileModel3D

# Below this tile count, process spawn overhead outweighs parallelism.
_MIN_TILES_FOR_MP = 16

# --- Shared state for fork-based multiprocessing -------------------------
# Workers read tile data from this dict (inherited via fork COW) instead of
# receiving it through pickle, avoiding serialization of huge tile dicts.
_SHARED_TILES: dict[tuple, dict] = {}
_SHARED_BOUNDS: tuple[float, ...] | None = None


def _get_mp_context() -> multiprocessing.context.BaseContext:
    """Return the best multiprocessing context for the platform.

    Uses ``fork`` on Unix (tiles shared via COW, no pickle overhead).
    Falls back to ``spawn`` on Windows where fork is unavailable.
    """
    if sys.platform == "win32":
        return multiprocessing.get_context("spawn")
    # fork is safe here — workers only use numpy, struct, protobuf (no macOS
    # GUI frameworks).  Suppress the Python 3.12+ deprecation warning.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*fork.*", DeprecationWarning)
        return multiprocessing.get_context("fork")


# --- Module-level worker functions (fork: read from _SHARED_TILES) --------


def _worker_pbf3(args: tuple) -> int:
    """Process a single pbf3 tile: transform → encode → write."""
    key, extent, extent_z, layer_name, output_dir = args
    tile = _SHARED_TILES[key]
    z, x, y, d = key

    transformed = transform_tile_3d(
        tile,
        extent=extent,
        extent_z=extent_z,
    )
    data = encode_tile_3d(
        transformed,
        layer_name=layer_name,
        extent=extent,
        extent_z=extent_z,
    )

    tile_path = Path(output_dir) / str(z) / str(x) / str(y) / f"{d}.pbf3"
    tile_path.parent.mkdir(parents=True, exist_ok=True)
    tile_path.write_bytes(data)
    return 1


def _simplify_ratio(zoom: int, max_zoom: int) -> float:
    """Compute mesh simplification ratio for a zoom level.

    Leaf tiles (zoom == max_zoom) get full resolution. Each level above
    keeps 1/4 of the faces (area-based reduction for octree).
    """
    if zoom >= max_zoom:
        return 1.0
    levels_above = max_zoom - zoom
    return 1.0 / (4**levels_above)


def _worker_3dtiles(args: tuple) -> int:
    """Process a single 3dtiles tile: encode to GLB → write."""
    key, output_dir, max_zoom = args
    tile = _SHARED_TILES[key]
    assert _SHARED_BOUNDS is not None
    bounds6 = cast("tuple[float, float, float, float, float, float]", _SHARED_BOUNDS)
    z, x, y, d = key

    from .gltf_encoder3d import tile_to_glb

    proj = CartesianProjector3D(bounds6)
    ratio = _simplify_ratio(z, max_zoom)
    data = tile_to_glb(
        tile,
        proj,
        simplify_ratio=ratio,
        world_bounds=bounds6,
        zoom=z,
    )

    tile_path = Path(output_dir) / str(z) / str(x) / str(y) / f"{d}.glb"
    tile_path.parent.mkdir(parents=True, exist_ok=True)
    tile_path.write_bytes(data)
    return 1


def _infer_field_type(value: Any) -> str | None:
    """Map a Python value to a TileJSON field type string."""
    if isinstance(value, bool):
        return "Boolean"
    if isinstance(value, (int, float)):
        return "Number"
    if isinstance(value, str):
        return "String"
    return None


class TileGenerator3D:
    """Generate 3D vector tiles from a MuDM FeatureCollection.

    Usage::

        gen = TileGenerator3D(config, output_format="pbf3")
        gen.add_features(collection)
        gen.generate(Path("output/tiles"))
        gen.write_metadata(Path("output/tiles"))

    Parameters
    ----------
    config : OctreeConfig, optional
        Octree configuration.
    output_format : {"pbf3", "3dtiles"}
        ``"pbf3"`` produces protobuf tiles + tilejson3d.json.
        ``"3dtiles"`` produces glTF/GLB tiles + tileset.json.
    workers : int or None
        Number of parallel worker processes for tile generation.
        ``None`` or ``0`` = auto (``os.cpu_count()``).
        ``1`` = single-threaded (current behavior).
        ``N > 1`` = explicit worker count.
    """

    def __init__(
        self,
        config: OctreeConfig | None = None,
        output_format: Literal["pbf3", "3dtiles"] = "pbf3",
        workers: int | None = None,
    ) -> None:
        self.config = config or OctreeConfig()
        self.output_format = output_format
        self._workers = workers
        self._octree: Octree | None = None
        self._proj: CartesianProjector3D | None = None
        self._bounds: tuple[float, float, float, float, float, float] | None = None
        self._layer_name = "default"
        # Metadata collected during add_features()
        self._field_types: dict[str, str] = {}
        self._field_ranges: dict[str, list[int | float]] = {}
        self._field_enums: dict[str, list[str]] = {}
        self._vocabularies: dict[str, Vocabulary] | None = None

    def _effective_workers(self) -> int:
        """Resolve worker count: None/0 → cpu_count, else use as-is."""
        if self._workers is None or self._workers == 0:
            return os.cpu_count() or 1
        return max(1, self._workers)

    def _scan_properties(self, collection: MuDMFeatureCollection) -> None:
        """Scan all feature properties to compute field types, ranges, enums."""
        numeric_mins: dict[str, float] = {}
        numeric_maxs: dict[str, float] = {}
        string_sets: dict[str, set[str]] = {}

        for feat in collection.features:
            if not feat.properties:
                continue
            for key, value in feat.properties.items():
                if value is None:
                    continue
                ftype = _infer_field_type(value)
                if ftype is None:
                    continue

                # Record type (first non-None wins; Number beats String if mixed)
                if key not in self._field_types:
                    self._field_types[key] = ftype
                elif self._field_types[key] != ftype:
                    # Mixed types — keep Number if either is Number
                    if ftype == "Number" or self._field_types[key] == "Number":
                        self._field_types[key] = "Number"

                # Track numeric ranges
                if ftype == "Number":
                    v = float(value)
                    if key not in numeric_mins:
                        numeric_mins[key] = v
                        numeric_maxs[key] = v
                    else:
                        if v < numeric_mins[key]:
                            numeric_mins[key] = v
                        if v > numeric_maxs[key]:
                            numeric_maxs[key] = v

                # Track string enumerations
                elif ftype == "String":
                    if key not in string_sets:
                        string_sets[key] = set()
                    string_sets[key].add(value)

        # Build ranges for numeric fields
        for key in numeric_mins:
            self._field_ranges[key] = [numeric_mins[key], numeric_maxs[key]]

        # Build enums for string fields (sorted for deterministic output)
        for key, vals in string_sets.items():
            self._field_enums[key] = sorted(vals)

    def add_features(
        self,
        collection: MuDMFeatureCollection,
        layer_name: str = "default",
    ) -> None:
        """Convert and index features into the octree.

        Parameters
        ----------
        collection : MuDMFeatureCollection
            Input features.
        layer_name : str
            Layer name for tile encoding.
        """
        self._layer_name = layer_name

        # Scan properties for metadata (field types, ranges, enums)
        self._scan_properties(collection)

        # Capture vocabularies from collection (dict only, not URI strings)
        if isinstance(collection.vocabularies, dict):
            self._vocabularies = collection.vocabularies

        # Compute world bounds
        bounds = compute_bounds_3d(collection)
        self._bounds = bounds

        # Update config bounds
        self.config.bounds = bounds

        # Project features to normalized [0,1]³
        proj = CartesianProjector3D(bounds)
        self._proj = proj
        features = convert_collection_3d(collection, proj)

        # Build octree
        self._octree = Octree(features, self.config)

    def generate(self, output_dir: Path | str) -> int:
        """Write tiles to disk.

        For ``"pbf3"`` format: ``{z}/{x}/{y}/{d}.pbf3``
        For ``"3dtiles"`` format: ``{z}/{x}/{y}/{d}.glb``

        Returns the number of tiles written.
        """
        if self._octree is None:
            raise RuntimeError("Call add_features() before generate()")

        if self.output_format == "3dtiles":
            return self._generate_3dtiles(Path(output_dir))
        return self._generate_pbf3(Path(output_dir))

    def _generate_pbf3(self, output_dir: Path) -> int:
        """Write protobuf-encoded .pbf3 tiles."""
        assert self._octree is not None

        tiles = [(k, t) for k, t in self._octree.all_tiles.items() if k[0] >= self.config.min_zoom]

        n_workers = self._effective_workers()
        if n_workers <= 1 or len(tiles) < _MIN_TILES_FOR_MP:
            return self._generate_pbf3_serial(tiles, output_dir)
        return self._generate_pbf3_parallel(tiles, output_dir, n_workers)

    def _generate_pbf3_serial(
        self,
        tiles: list,
        output_dir: Path,
    ) -> int:
        """Single-threaded pbf3 tile generation."""
        count = 0
        for (z, x, y, d), tile in tiles:
            transformed = transform_tile_3d(
                tile,
                extent=self.config.extent,
                extent_z=self.config.extent_z,
            )
            data = encode_tile_3d(
                transformed,
                layer_name=self._layer_name,
                extent=self.config.extent,
                extent_z=self.config.extent_z,
            )
            tile_path = output_dir / str(z) / str(x) / str(y) / f"{d}.pbf3"
            tile_path.parent.mkdir(parents=True, exist_ok=True)
            tile_path.write_bytes(data)
            count += 1
        return count

    def _generate_pbf3_parallel(
        self,
        tiles: list,
        output_dir: Path,
        n_workers: int,
    ) -> int:
        """Multiprocessing pbf3 tile generation.

        Uses fork context: tile data is shared via COW memory, only
        lightweight keys are passed through the pool.
        """
        global _SHARED_TILES
        _SHARED_TILES = {key: tile for key, tile in tiles}
        args = [
            (key, self.config.extent, self.config.extent_z, self._layer_name, str(output_dir))
            for key, _ in tiles
        ]
        chunksize = max(1, len(args) // (n_workers * 4))
        ctx = _get_mp_context()
        with ctx.Pool(n_workers) as pool:
            results = pool.map(_worker_pbf3, args, chunksize=chunksize)
        _SHARED_TILES = {}
        return sum(results)

    def _generate_3dtiles(self, output_dir: Path) -> int:
        """Write GLB tiles for OGC 3D Tiles output."""
        assert self._octree is not None
        assert self._proj is not None
        assert self._bounds is not None

        tiles = [(k, t) for k, t in self._octree.all_tiles.items() if k[0] >= self.config.min_zoom]

        n_workers = self._effective_workers()
        if n_workers <= 1 or len(tiles) < _MIN_TILES_FOR_MP:
            return self._generate_3dtiles_serial(tiles, output_dir)
        return self._generate_3dtiles_parallel(tiles, output_dir, n_workers)

    def _generate_3dtiles_serial(
        self,
        tiles: list,
        output_dir: Path,
    ) -> int:
        """Single-threaded 3dtiles generation."""
        from .gltf_encoder3d import tile_to_glb

        assert self._proj is not None
        max_z = self.config.max_zoom
        count = 0
        for (z, x, y, d), tile in tiles:
            ratio = _simplify_ratio(z, max_z)
            data = tile_to_glb(
                tile,
                self._proj,
                simplify_ratio=ratio,
                world_bounds=self._bounds,
                zoom=z,
            )
            tile_path = output_dir / str(z) / str(x) / str(y) / f"{d}.glb"
            tile_path.parent.mkdir(parents=True, exist_ok=True)
            tile_path.write_bytes(data)
            count += 1
        return count

    def _generate_3dtiles_parallel(
        self,
        tiles: list,
        output_dir: Path,
        n_workers: int,
    ) -> int:
        """Multiprocessing 3dtiles generation.

        Uses fork context: tile data + bounds shared via COW memory.
        """
        global _SHARED_TILES, _SHARED_BOUNDS
        _SHARED_TILES = {key: tile for key, tile in tiles}
        _SHARED_BOUNDS = self._bounds
        max_z = self.config.max_zoom
        args = [(key, str(output_dir), max_z) for key, _ in tiles]
        chunksize = max(1, len(args) // (n_workers * 4))
        ctx = _get_mp_context()
        with ctx.Pool(n_workers) as pool:
            results = pool.map(_worker_3dtiles, args, chunksize=chunksize)
        _SHARED_TILES = {}
        _SHARED_BOUNDS = None
        return sum(results)

    def write_metadata(self, output_dir: Path | str) -> None:
        """Write the appropriate metadata file for the output format.

        For ``"pbf3"``: writes ``tilejson3d.json``
        For ``"3dtiles"``: writes ``tileset.json``
        """
        output_dir = Path(output_dir)
        if self.output_format == "3dtiles":
            self.write_tileset_json(output_dir / "tileset.json")
        else:
            self.write_tilejson(output_dir / "tilejson3d.json")

    def write_tilejson(self, path: Path | str) -> None:
        """Write TileJSON 3D metadata file (.pbf3 format)."""
        if self._bounds is None:
            raise RuntimeError("Call add_features() before write_tilejson()")

        path = Path(path)
        bounds = self._bounds

        model = TileModel3D(
            tilejson="3.0.0",
            tiles=["{z}/{x}/{y}/{d}.pbf3"],
            name=self._layer_name,
            minzoom=self.config.min_zoom,
            maxzoom=self.config.max_zoom,
            bounds3d=list(bounds),
            center3d=[
                (bounds[0] + bounds[3]) / 2,
                (bounds[1] + bounds[4]) / 2,
                (bounds[2] + bounds[5]) / 2,
                self.config.min_zoom,
            ],
            depthsize=self.config.extent_z,
            vector_layers=[
                {
                    "id": self._layer_name,
                    "fields": self._field_types or {},
                    "minzoom": self.config.min_zoom,
                    "maxzoom": self.config.max_zoom,
                    **({"fieldranges": self._field_ranges} if self._field_ranges else {}),
                    **({"fieldenums": self._field_enums} if self._field_enums else {}),
                    **(
                        {
                            "vocabularies": {
                                k: v.model_dump(exclude_none=True)
                                for k, v in self._vocabularies.items()
                            }
                        }
                        if self._vocabularies
                        else {}
                    ),
                }
            ],
        )

        path.write_text(model.model_dump_json(indent=2, exclude_none=True))

    def write_tileset_json(self, path: Path | str) -> None:
        """Write OGC 3D Tiles tileset.json metadata file."""
        if self._bounds is None or self._octree is None or self._proj is None:
            raise RuntimeError("Call add_features() before write_tileset_json()")

        from .tileset_json import generate_tileset_json, write_tileset_json

        tileset = generate_tileset_json(
            all_tiles=self._octree.all_tiles,
            world_bounds=self._bounds,
            proj=self._proj,
            min_zoom=self.config.min_zoom,
            max_zoom=self.config.max_zoom,
        )
        write_tileset_json(tileset, path)
