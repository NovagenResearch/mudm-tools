"""Parity tests for the Python 2D polygon simplification against the Rust
implementation in rust/src/simplify2d.rs.

The core assertions mirror the Rust unit tests one-to-one (same inputs, same
expected outputs) so the pure-Python tiler and the Rust streaming tiler agree
on simplified geometry. Integration tests exercise the MuDMVt wiring
(per-zoom tolerance array + min-edge floor + reduce-to-floor)."""

import math
import random

from mudm_tools.mudm2vt.convert import convert
from mudm_tools.mudm2vt.mudm2vt import build_zoom_geometries, get_default_options
from mudm_tools.mudm2vt.simplify import (
    compute_tolerance,
    douglas_peucker_floor,
    simplify,
    simplify_polygon_ring,
    simplify_polygon_rings,
)

# --------------------------------------------------------------------------
# Core algorithm parity (mirrors rust/src/simplify2d.rs #[cfg(test)] module)
# --------------------------------------------------------------------------


def test_dp_straight_line():
    # Rust test_dp_straight_line: collinear points collapse to the 2 endpoints.
    pts = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    out = douglas_peucker_floor(pts, 0.01, 2)
    assert len(out) == 2


def test_dp_zigzag():
    # Rust test_dp_zigzag: every vertex of a zigzag is kept.
    pts = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0), (3.0, 1.0), (4.0, 0.0)]
    out = douglas_peucker_floor(pts, 0.01, 2)
    assert len(out) == 5


def test_dp_preserves_endpoints():
    # Rust test_dp_preserves_endpoints: middle point within epsilon is dropped.
    pts = [(0.0, 0.0), (0.5, 0.001), (1.0, 0.0)]
    out = douglas_peucker_floor(pts, 0.01, 2)
    assert out == [[0.0, 0.0], [1.0, 0.0]]


def test_simplify_polygon_ring_collinear():
    # Rust test_simplify_polygon_rings: square with a collinear midpoint (5
    # verts) drops the midpoint to 4 with floor 4.
    ring = [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    out = simplify_polygon_ring(ring, 0.01, 4)
    assert len(out) == 4


def test_floor_reduces_not_reverts():
    # Rust test_floor_reduces_not_reverts: a 20-vertex ring with a collapsing
    # epsilon must reduce TO the floor, never revert to the full ring.
    nv = 20
    ring = [(math.cos(i / nv * math.tau), math.sin(i / nv * math.tau)) for i in range(nv)]
    out = simplify_polygon_ring(ring, 1e9, 6)
    assert len(out) == 6, "must reduce to the floor, not revert to full"


def test_dp_floor_keeps_min():
    # Rust test_dp_floor_keeps_min: huge epsilon would collapse to 2 endpoints;
    # the floor keeps min_verts.
    pts = [(0.0, 0.0), (1.0, 0.001), (2.0, 0.0), (3.0, 0.001), (4.0, 0.0)]
    out = douglas_peucker_floor(pts, 1e9, 4)
    assert len(out) == 4


def test_compute_tolerance():
    # Rust test_compute_tolerance.
    assert compute_tolerance(4, 4) == 0.0
    t = compute_tolerance(3, 4)
    assert t > 0.0
    t2 = compute_tolerance(2, 4)
    assert abs(t2 - t * 2.0) < 1e-15


def test_dp_floor_at_or_below_floor_is_untouched():
    # A ring already at/below the floor is returned unchanged.
    pts = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    out = douglas_peucker_floor(pts, 1e9, 6)
    assert out == [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]


def test_simplify_polygon_rings_multi():
    # Multi-ring wrapper applies the floor per ring.
    outer = [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    inner = [(0.2, 0.2), (0.3, 0.2), (0.4, 0.2), (0.4, 0.4), (0.2, 0.4)]
    out = simplify_polygon_rings([outer, inner], 0.01, 4)
    assert len(out) == 2
    assert len(out[0]) == 4 and len(out[1]) == 4


def test_legacy_simplify_never_reverts():
    # The historical simplify() no longer reverts to the full ring: a 20-vertex
    # ring under a huge squared tolerance reduces to the floor (min_vertices).
    nv = 20
    ring = [(math.cos(i / nv * math.tau), math.sin(i / nv * math.tau)) for i in range(nv)]
    out = simplify(ring, 1e18, min_vertices=5)
    assert len(out) == 5


# --------------------------------------------------------------------------
# MuDMVt integration (per-zoom array + min-edge floor + reduce-to-floor)
# --------------------------------------------------------------------------


def _jagged_polygon(cx, cy, r, nv, seed):
    """A closed ring with a jagged radius so it carries many real vertices."""
    rng = random.Random(seed)
    ring = []
    for i in range(nv):
        a = 2 * math.pi * i / nv
        rr = r * (0.7 + 0.6 * rng.random())
        ring.append([cx + rr * math.cos(a), cy + rr * math.sin(a)])
    ring.append(ring[0])  # close the ring
    return ring


def _feature_collection(nv=40, npoly=8):
    feats = []
    for k in range(npoly):
        cx = (k % 4) * 100.0 + 50.0
        cy = (k // 4) * 100.0 + 50.0
        feats.append(
            {
                "type": "Feature",
                "properties": {"id": k},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [_jagged_polygon(cx, cy, 35.0, nv, k)],
                },
            }
        )
    return {"type": "FeatureCollection", "features": feats}


def _total_vertices(features, z):
    total = 0
    for f in features:
        for ring in f[f"geometry_z{z}"]:
            total += len(ring) // 3
    return total


def _total_vertices_source(features):
    total = 0
    for f in features:
        for ring in f["geometry"]:
            total += len(ring) // 3
    return total


def _opts(**overrides):
    opts = get_default_options()
    opts.update({"maxZoom": 6, "extent": 4096, "bounds": [0.0, 0.0, 400.0, 400.0]})
    opts.update(overrides)
    return opts


def test_mudmvt_no_lod_without_config():
    # Polygon LOD is opt-in: with no per-zoom tolerance array, every zoom keeps
    # full detail, so the tiler's default output is unchanged.
    data = _feature_collection(nv=40)
    opts = _opts()
    feats = convert(data, opts)
    build_zoom_geometries(feats, opts)

    source = _total_vertices_source(feats)
    for z in range(opts["maxZoom"] + 1):
        assert _total_vertices(feats, z) == source


def test_mudmvt_array_enables_lod():
    # Providing a per-zoom tolerance array opts in: the coarsest level holds far
    # fewer vertices than the finest (which is never simplified).
    data = _feature_collection(nv=40)
    opts = _opts(poly_simplify_tolerances=[0.2, 0.1, 0.05, 0.02, 0.01, 0.005])
    feats = convert(data, opts)
    build_zoom_geometries(feats, opts)

    coarse = _total_vertices(feats, 0)
    fine = _total_vertices(feats, opts["maxZoom"])
    assert fine > coarse
    # Coarsest level is aggressively reduced (large epsilon => floor-ish).
    assert coarse < fine / 4


def test_mudmvt_finest_level_is_full_detail():
    # Even with LOD enabled, the finest zoom keeps every input vertex.
    data = _feature_collection(nv=40)
    opts = _opts(poly_simplify_tolerances=[0.2, 0.1, 0.05, 0.02, 0.01, 0.005])
    feats = convert(data, opts)
    build_zoom_geometries(feats, opts)

    for f in feats:
        for src_ring, fine_ring in zip(f["geometry"], f[f"geometry_z{opts['maxZoom']}"]):
            assert len(fine_ring) == len(src_ring)


def test_mudmvt_min_edges_floor_respected():
    # A per-zoom tolerance array with an aggressive coarse epsilon reduces each
    # ring to the min-edge floor, never below it.
    data = _feature_collection(nv=40)
    min_edges = 6
    opts = _opts(
        poly_simplify_tolerances=[0.4, 0.1, 0.05, 0.02, 0.01, 0.005],
        poly_min_edges=min_edges,
    )
    feats = convert(data, opts)
    build_zoom_geometries(feats, opts)

    for f in feats:
        for ring in f["geometry_z0"]:
            assert len(ring) // 3 == min_edges


def test_mudmvt_min_edges_clamped_to_four():
    # poly_min_edges below 4 is clamped to 4 (a valid closed ring floor).
    data = _feature_collection(nv=40)
    opts = _opts(poly_simplify_tolerances=[0.4, 0.1, 0.05, 0.02, 0.01, 0.005], poly_min_edges=2)
    feats = convert(data, opts)
    build_zoom_geometries(feats, opts)

    for f in feats:
        for ring in f["geometry_z0"]:
            assert len(ring) // 3 == 4


def test_mudmvt_polygon_only():
    # LineString geometry is copied unchanged at every zoom (simplification is
    # polygon-only).
    line = [[float(i), float(i * i % 7)] for i in range(30)]
    data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"id": 0},
                "geometry": {"type": "LineString", "coordinates": line},
            },
        ],
    }
    opts = _opts(poly_simplify_tolerances=[0.4, 0.1, 0.05, 0.02, 0.01, 0.005], poly_min_edges=6)
    feats = convert(data, opts)
    build_zoom_geometries(feats, opts)

    n_full = len(feats[0][f"geometry_z{opts['maxZoom']}"])
    for z in range(opts["maxZoom"] + 1):
        assert len(feats[0][f"geometry_z{z}"]) == n_full
