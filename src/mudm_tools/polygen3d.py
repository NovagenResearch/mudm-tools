"""Random 3D geometry generator — TIN-focused.

Generates random TIN meshes (primary), 3D Points, and 3D LineStrings
for testing and demo purposes. Reuses metadata helpers from ``polygen``.
"""

from __future__ import annotations

import random
import math
from typing import List, Tuple, Dict, Any, Optional

from geojson_pydantic import Point, LineString

from mudm.model import TIN, MuDMFeature, MuDMFeatureCollection
from .polygen import assign_meta_types_and_values, generate_meta_values

Bounds6 = Tuple[float, float, float, float, float, float]


def _random_point_in_bounds(bounds: Bounds6) -> Tuple[float, float, float]:
    """Return a random 3D point inside *bounds*."""
    xmin, ymin, zmin, xmax, ymax, zmax = bounds
    return (
        random.uniform(xmin, xmax),
        random.uniform(ymin, ymax),
        random.uniform(zmin, zmax),
    )


def _make_triangle_cluster(
    cx: float,
    cy: float,
    cz: float,
    radius: float,
    n_triangles: int,
) -> List[List[List[List[float]]]]:
    """Build *n_triangles* around a centre point.

    Creates a fan of triangles sharing the centre vertex, connected via
    a ring of randomly-perturbed points on a sphere.
    """
    # Generate ring points on a sphere
    ring: list[tuple[float, float, float]] = []
    for i in range(n_triangles):
        theta = 2 * math.pi * i / n_triangles + random.uniform(-0.15, 0.15)
        phi = math.pi / 2 + random.uniform(-0.4, 0.4)
        r = radius * random.uniform(0.6, 1.0)
        px = cx + r * math.sin(phi) * math.cos(theta)
        py = cy + r * math.sin(phi) * math.sin(theta)
        pz = cz + r * math.cos(phi)
        ring.append((px, py, pz))

    faces: List[List[List[List[float]]]] = []
    for i in range(n_triangles):
        p1 = ring[i]
        p2 = ring[(i + 1) % n_triangles]
        # TIN face: one ring, 4 positions (3 verts + close)
        face = [[
            [cx, cy, cz],
            [p1[0], p1[1], p1[2]],
            [p2[0], p2[1], p2[2]],
            [cx, cy, cz],  # close
        ]]
        faces.append(face)
    return faces


def generate_random_tins(
    n: int,
    bounds: Bounds6,
    triangles_per_tin: int = 6,
) -> List[MuDMFeature]:
    """Generate *n* random TIN features within *bounds*.

    Each TIN is a cluster of triangles around a random centre point.
    """
    xmin, ymin, zmin, xmax, ymax, zmax = bounds
    max_dim = max(xmax - xmin, ymax - ymin, zmax - zmin)
    radius = max_dim * 0.05  # each cluster is ~5% of the span

    features: List[MuDMFeature] = []
    for i in range(n):
        cx, cy, cz = _random_point_in_bounds(bounds)
        faces = _make_triangle_cluster(cx, cy, cz, radius, triangles_per_tin)
        geom = TIN(type="TIN", coordinates=faces)
        features.append(MuDMFeature(
            type="Feature",
            id=i,
            geometry=geom,
            properties={"kind": "tin", "triangles": triangles_per_tin},
        ))
    return features


def generate_random_points_3d(
    n: int,
    bounds: Bounds6,
) -> List[MuDMFeature]:
    """Generate *n* random 3D Point features within *bounds*."""
    features: List[MuDMFeature] = []
    for i in range(n):
        x, y, z = _random_point_in_bounds(bounds)
        geom = Point(type="Point", coordinates=[x, y, z])
        features.append(MuDMFeature(
            type="Feature",
            id=1000 + i,
            geometry=geom,
            properties={"kind": "point"},
        ))
    return features


def generate_random_lines_3d(
    n: int,
    bounds: Bounds6,
    min_verts: int = 3,
    max_verts: int = 8,
) -> List[MuDMFeature]:
    """Generate *n* random 3D LineString features within *bounds*."""
    features: List[MuDMFeature] = []
    for i in range(n):
        nv = random.randint(min_verts, max_verts)
        coords = [list(_random_point_in_bounds(bounds)) for _ in range(nv)]
        geom = LineString(type="LineString", coordinates=coords)
        features.append(MuDMFeature(
            type="Feature",
            id=2000 + i,
            geometry=geom,
            properties={"kind": "line", "vertices": nv},
        ))
    return features


def generate_3d_collection(
    n_tins: int = 10,
    n_points: int = 5,
    n_lines: int = 5,
    bounds: Bounds6 = (0.0, 0.0, 0.0, 100.0, 100.0, 100.0),
    triangles_per_tin: int = 6,
    n_meta_keys: int = 3,
    n_meta_variants: int = 4,
    seed: Optional[int] = None,
) -> MuDMFeatureCollection:
    """Generate a mixed 3D ``MuDMFeatureCollection``.

    Parameters
    ----------
    n_tins : int
        Number of TIN mesh features.
    n_points : int
        Number of 3D Point features.
    n_lines : int
        Number of 3D LineString features.
    bounds : Bounds6
        (xmin, ymin, zmin, xmax, ymax, zmax) for random generation.
    triangles_per_tin : int
        Triangles per TIN cluster.
    n_meta_keys : int
        Number of random metadata keys to attach.
    n_meta_variants : int
        Number of distinct values per metadata key.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    MuDMFeatureCollection
    """
    if seed is not None:
        random.seed(seed)

    tins = generate_random_tins(n_tins, bounds, triangles_per_tin)
    points = generate_random_points_3d(n_points, bounds)
    lines = generate_random_lines_3d(n_lines, bounds)

    all_features = tins + points + lines

    # Attach random metadata via polygen helpers
    if n_meta_keys > 0:
        _, meta_values_options = assign_meta_types_and_values(
            n_meta_keys, n_meta_variants,
        )
        for feat in all_features:
            extra = generate_meta_values(meta_values_options)
            if feat.properties is None:
                feat.properties = {}
            feat.properties.update(extra)

    return MuDMFeatureCollection(
        type="FeatureCollection",
        features=all_features,
        properties={"generator": "polygen3d"},
    )
