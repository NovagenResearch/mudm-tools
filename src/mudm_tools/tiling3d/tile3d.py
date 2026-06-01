"""3D tile creation and coordinate transform.

Creates tile data structures and transforms normalized [0,1]
coordinates to tile-local integer coordinates.
"""

from __future__ import annotations

from typing import Any


def create_tile_3d(
    features: list[dict],
    z: int,
    tx: int,
    ty: int,
    td: int,
) -> dict:
    """Create a 3D tile dict from intermediate features.

    Parameters
    ----------
    features : list[dict]
        Intermediate features (clipped to this tile's extent).
    z : int
        Zoom level.
    tx, ty, td : int
        Tile coordinates (x, y, depth).

    Returns
    -------
    Tile dict with features, coordinates, and 3D bbox.
    """
    tile: dict[str, Any] = {
        "features": features,
        "z": z,
        "x": tx,
        "y": ty,
        "d": td,
        "num_features": len(features),
        "num_points": sum(len(f["geometry_z"]) for f in features),
    }

    if features:
        tile["minX"] = min(f["minX"] for f in features)
        tile["minY"] = min(f["minY"] for f in features)
        tile["minZ"] = min(f["minZ"] for f in features)
        tile["maxX"] = max(f["maxX"] for f in features)
        tile["maxY"] = max(f["maxY"] for f in features)
        tile["maxZ"] = max(f["maxZ"] for f in features)
    else:
        tile["minX"] = tile["minY"] = tile["minZ"] = 0.0
        tile["maxX"] = tile["maxY"] = tile["maxZ"] = 0.0

    return tile


def transform_tile_3d(
    tile: dict,
    extent: int = 4096,
    extent_z: int = 4096,
) -> dict:
    """Transform tile features from normalized [0,1] sub-range to integer coords.

    The tile covers a sub-range of [0,1] based on its z/x/y/d coordinates.
    This converts to tile-local integers in [0, extent] x [0, extent] x [0, extent_z].

    Parameters
    ----------
    tile : dict
        Tile from create_tile_3d.
    extent : int
        XY extent (default 4096).
    extent_z : int
        Z extent (default 4096).

    Returns
    -------
    New tile dict with integer coordinates.
    """
    z = tile["z"]
    tx = tile["x"]
    ty = tile["y"]
    td = tile["d"]

    # Number of tiles per axis at this zoom
    n = 1 << z  # 2^z

    # Sub-range this tile covers in [0,1]
    x0 = tx / n
    y0 = ty / n
    z0 = td / n
    scale_x = n
    scale_y = n
    scale_z = n

    new_features = []
    for feat in tile["features"]:
        xy = feat["geometry"]
        zz = feat["geometry_z"]
        nv = len(zz)

        new_xy: list[int] = []
        new_z: list[int] = []

        for i in range(nv):
            # Normalize to tile-local [0, 1]
            lx = (xy[i * 2] - x0) * scale_x
            ly = (xy[i * 2 + 1] - y0) * scale_y
            lz = (zz[i] - z0) * scale_z

            # Scale to integer extent
            new_xy.append(round(lx * extent))
            new_xy.append(round(ly * extent))
            new_z.append(round(lz * extent_z))

        new_feat = {
            "geometry": new_xy,
            "geometry_z": new_z,
            "type": feat["type"],
            "tags": feat.get("tags", {}),
        }
        if "ring_lengths" in feat:
            new_feat["ring_lengths"] = feat["ring_lengths"]
        if "radii" in feat:
            new_feat["radii"] = feat["radii"]
        new_features.append(new_feat)

    return {
        "features": new_features,
        "z": z,
        "x": tx,
        "y": ty,
        "d": td,
        "num_features": len(new_features),
        "num_points": sum(len(f["geometry_z"]) for f in new_features),
        "extent": extent,
        "extent_z": extent_z,
    }


# ---------------------------------------------------------------------------
# Rust dispatch: import compiled version from the Rust extension module.
# ---------------------------------------------------------------------------
transform_tile_3d_py = transform_tile_3d

try:
    from mudm_tools._rs import transform_tile_3d  # type: ignore[no-redef]  # noqa: F811
except ImportError:
    pass
