"""TileModel3D — TileJSON with 3D fields and tile encoding types."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, conlist

from mudm.tilemodel import TileModel

# --- Tile encoding types (semi-open enums) ---

KnownTileFormat = Literal["glb", "parquet", "arrow", "neuroglancer-precomputed"]
KnownCompression = Literal["meshopt", "draco", "zstd"]


class TileEncoding(BaseModel):
    """One available encoding for tiled geometry data.

    Attributes:
        format: Data format — known values: glb, parquet, arrow,
            neuroglancer-precomputed. Arbitrary strings accepted for extension.
        compression: Optional compression within the format (meshopt, draco, zstd).
        path: Base path relative to the pyramid root directory.
        extension: File extension including the dot (e.g. ".glb", ".parquet").
    """

    format: Union[KnownTileFormat, str]
    compression: Optional[Union[KnownCompression, str]] = None
    path: str
    extension: str


class TileModel3D(TileModel):
    """Extended TileJSON metadata for 3D tilesets.

    Adds depth axis, 3D bounds/center, per-zoom resolution info,
    and tile encoding declarations.
    """

    # 3D bounds: [xmin, ymin, zmin, xmax, ymax, zmax]
    bounds3d: Optional[conlist(float, min_length=6, max_length=6)] = None  # type: ignore

    # 3D center: [x, y, z, zoom]
    center3d: Optional[conlist(float, min_length=4, max_length=4)] = None  # type: ignore

    # Number of depth slices per zoom level (analogous to tile size)
    depthsize: Optional[int] = 256

    # Map of zoom level -> spatial resolution (units per voxel)
    resolution_per_zoom: Optional[dict[int, float]] = None

    # Available encodings for this pyramid's tiled data
    encodings: Optional[List[TileEncoding]] = None

    # Per-zoom tile counts (string keys for JSON compatibility)
    zoom_counts: Optional[Dict[str, int]] = None

    # Property keys that are identifiers (excluded from filter/color-by in viewer)
    id_fields: Optional[List[str]] = None
