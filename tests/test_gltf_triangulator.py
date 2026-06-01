"""Tests for polygon triangulation."""

import numpy as np
from shapely.geometry import MultiPolygon, Polygon

from mudm_tools.gltf.triangulator import multipolygon_to_mesh, polygon_to_mesh


class TestPolygonToMesh:
    def test_simple_square(self):
        square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        verts, indices = polygon_to_mesh(square)

        assert verts.shape[1] == 3
        assert indices.shape[1] == 3
        # A square triangulates into 2 triangles
        assert indices.shape[0] == 2
        # All Z coords should be 0 (default)
        np.testing.assert_array_equal(verts[:, 2], 0.0)

    def test_z_coordinate_passthrough(self):
        square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        verts, _ = polygon_to_mesh(square, z=5.0)
        np.testing.assert_array_equal(verts[:, 2], 5.0)

    def test_concave_polygon(self):
        # L-shape
        L = Polygon([(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)])
        verts, indices = polygon_to_mesh(L)

        assert verts.shape[0] >= 4
        assert indices.shape[0] >= 3  # At least 3 triangles for L-shape
        # All index values must be valid vertex indices
        assert indices.max() < verts.shape[0]

    def test_polygon_with_hole(self):
        outer = [(0, 0), (10, 0), (10, 10), (0, 10)]
        hole = [(3, 3), (7, 3), (7, 7), (3, 7)]
        poly = Polygon(outer, [hole])
        verts, indices = polygon_to_mesh(poly)

        assert verts.shape[0] > 0
        assert indices.shape[0] > 0
        # Reconstruct triangles and ensure no centroid falls inside hole
        hole_poly = Polygon(hole)
        for face in indices:
            tri_coords = verts[face]
            cx = tri_coords[:, 0].mean()
            cy = tri_coords[:, 1].mean()
            from shapely.geometry import Point

            assert not hole_poly.contains(Point(cx, cy)), "Triangle inside hole"

    def test_empty_polygon(self):
        empty = Polygon()
        verts, indices = polygon_to_mesh(empty)
        assert verts.shape == (0, 3)
        assert indices.shape == (0, 3)

    def test_triangle(self):
        tri = Polygon([(0, 0), (1, 0), (0.5, 1)])
        verts, indices = polygon_to_mesh(tri)
        assert indices.shape[0] == 1  # single triangle
        assert verts.shape[0] == 3


class TestMultipolygonToMesh:
    def test_two_squares(self):
        p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        p2 = Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])
        mp = MultiPolygon([p1, p2])
        verts, indices = multipolygon_to_mesh(mp)

        # 2 squares = 4 triangles total
        assert indices.shape[0] == 4
        # All indices valid
        assert indices.max() < verts.shape[0]

    def test_empty_multipolygon(self):
        mp = MultiPolygon()
        verts, indices = multipolygon_to_mesh(mp)
        assert verts.shape == (0, 3)
        assert indices.shape == (0, 3)

    def test_z_passthrough(self):
        p = Polygon([(0, 0), (1, 0), (0.5, 1)])
        mp = MultiPolygon([p])
        verts, _ = multipolygon_to_mesh(mp, z=3.0)
        np.testing.assert_array_equal(verts[:, 2], 3.0)
