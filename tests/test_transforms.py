"""Tests for 3D coordinate transforms."""

import pytest
from mudm.transforms import (
    AffineTransform,
    VoxelCoordinateSystem,
    apply_transform,
    translate_geometry,
    voxel_to_physical,
    physical_to_voxel,
)
from mudm.model import (
    TIN,
    PolyhedralSurface,
)
from geojson_pydantic import Point, LineString, Polygon, MultiPolygon


class TestAffineTransform:
    def test_identity(self):
        t = AffineTransform(
            type="affine",
            matrix=[
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
        )
        assert t.matrix[0][0] == 1.0

    def test_translation(self):
        t = AffineTransform(
            type="affine",
            matrix=[
                [1, 0, 0, 10],
                [0, 1, 0, 20],
                [0, 0, 1, 30],
                [0, 0, 0, 1],
            ],
        )
        assert t.matrix[0][3] == 10.0

    def test_must_be_4x4(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            AffineTransform(
                type="affine",
                matrix=[[1, 0], [0, 1]],
            )

    def test_roundtrip(self):
        t = AffineTransform(
            type="affine",
            matrix=[
                [1, 0, 0, 5],
                [0, 1, 0, 10],
                [0, 0, 1, 15],
                [0, 0, 0, 1],
            ],
        )
        data = t.model_dump()
        t2 = AffineTransform.model_validate(data)
        assert t2.matrix[2][3] == 15.0


class TestVoxelCoordinateSystem:
    def test_create(self):
        vcs = VoxelCoordinateSystem(
            axes=["x", "y", "z"],
            units=["micrometer", "micrometer", "micrometer"],
            resolution=[0.4, 0.4, 0.05],
            origin=[0.0, 0.0, 0.0],
        )
        assert vcs.resolution == [0.4, 0.4, 0.05]

    def test_default_origin(self):
        vcs = VoxelCoordinateSystem(
            axes=["x", "y", "z"],
            units=["micrometer", "micrometer", "micrometer"],
            resolution=[1.0, 1.0, 1.0],
        )
        assert vcs.origin == [0.0, 0.0, 0.0]

    def test_roundtrip(self):
        vcs = VoxelCoordinateSystem(
            axes=["x", "y", "z"],
            units=["micrometer", "micrometer", "micrometer"],
            resolution=[0.4, 0.4, 0.05],
        )
        data = vcs.model_dump()
        vcs2 = VoxelCoordinateSystem.model_validate(data)
        assert vcs2.resolution == [0.4, 0.4, 0.05]


class TestApplyTransform:
    def test_translate_point(self):
        p = Point(type="Point", coordinates=(10.0, 20.0, 30.0))
        t = AffineTransform(
            type="affine",
            matrix=[
                [1, 0, 0, 5],
                [0, 1, 0, 10],
                [0, 0, 1, 15],
                [0, 0, 0, 1],
            ],
        )
        result = apply_transform(p, t)
        assert result.coordinates[0] == pytest.approx(15.0)
        assert result.coordinates[1] == pytest.approx(30.0)
        assert result.coordinates[2] == pytest.approx(45.0)

    def test_scale_point(self):
        p = Point(type="Point", coordinates=(10.0, 20.0, 30.0))
        t = AffineTransform(
            type="affine",
            matrix=[
                [2, 0, 0, 0],
                [0, 3, 0, 0],
                [0, 0, 4, 0],
                [0, 0, 0, 1],
            ],
        )
        result = apply_transform(p, t)
        assert result.coordinates[0] == pytest.approx(20.0)
        assert result.coordinates[1] == pytest.approx(60.0)
        assert result.coordinates[2] == pytest.approx(120.0)

    def test_transform_linestring(self):
        ls = LineString(
            type="LineString",
            coordinates=[(0, 0, 0), (10, 10, 10)],
        )
        t = AffineTransform(
            type="affine",
            matrix=[
                [1, 0, 0, 100],
                [0, 1, 0, 200],
                [0, 0, 1, 300],
                [0, 0, 0, 1],
            ],
        )
        result = apply_transform(ls, t)
        assert result.coordinates[0][0] == pytest.approx(100.0)
        assert result.coordinates[1][2] == pytest.approx(310.0)


class TestApplyTransform3D:
    """Tests for 3D MuDM type transforms."""

    def _translation(self, dx=10, dy=20, dz=30):
        return AffineTransform(
            type="affine",
            matrix=[
                [1, 0, 0, dx],
                [0, 1, 0, dy],
                [0, 0, 1, dz],
                [0, 0, 0, 1],
            ],
        )

    def test_translate_tin(self):
        tin = TIN(
            type="TIN",
            coordinates=[
                [[(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 0)]],
            ],
        )
        result = apply_transform(tin, self._translation(10, 20, 30))
        assert isinstance(result, TIN)
        # First vertex of first triangle
        assert result.coordinates[0][0][0][0] == pytest.approx(10.0)
        assert result.coordinates[0][0][0][1] == pytest.approx(20.0)
        assert result.coordinates[0][0][0][2] == pytest.approx(30.0)

    def test_translate_polyhedral_surface(self):
        ps = PolyhedralSurface(
            type="PolyhedralSurface",
            coordinates=[
                [[(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 0)]],
            ],
        )
        result = apply_transform(ps, self._translation(5, 10, 15))
        assert isinstance(result, PolyhedralSurface)
        assert result.coordinates[0][0][0][0] == pytest.approx(5.0)
        assert result.coordinates[0][0][0][1] == pytest.approx(10.0)
        assert result.coordinates[0][0][0][2] == pytest.approx(15.0)

class TestTranslateGeometry:
    """Tests for translate_geometry() convenience function."""

    def test_translate_point(self):
        p = Point(type="Point", coordinates=(1.0, 2.0, 3.0))
        result = translate_geometry(p, 10, 20, 30)
        assert result.coordinates[0] == pytest.approx(11.0)
        assert result.coordinates[1] == pytest.approx(22.0)
        assert result.coordinates[2] == pytest.approx(33.0)

    def test_translate_tin(self):
        tin = TIN(
            type="TIN",
            coordinates=[
                [[(0, 0, 0), (1, 0, 0), (0.5, 1, 0), (0, 0, 0)]],
            ],
        )
        result = translate_geometry(tin, 100, 200, 300)
        assert result.coordinates[0][0][0][0] == pytest.approx(100.0)
        assert result.coordinates[0][0][0][1] == pytest.approx(200.0)
        assert result.coordinates[0][0][0][2] == pytest.approx(300.0)

    def test_translate_zero_is_identity(self):
        p = Point(type="Point", coordinates=(5.0, 6.0, 7.0))
        result = translate_geometry(p, 0, 0, 0)
        assert result.coordinates[0] == pytest.approx(5.0)
        assert result.coordinates[1] == pytest.approx(6.0)
        assert result.coordinates[2] == pytest.approx(7.0)


class TestVoxelPhysicalConversion:
    def _vcs(self):
        return VoxelCoordinateSystem(
            axes=["x", "y", "z"],
            units=["micrometer", "micrometer", "micrometer"],
            resolution=[0.4, 0.4, 0.05],
            origin=[10.0, 20.0, 30.0],
        )

    def test_voxel_to_physical(self):
        vcs = self._vcs()
        result = voxel_to_physical((100.0, 200.0, 300.0), vcs)
        # physical = voxel * resolution + origin
        assert result[0] == pytest.approx(100.0 * 0.4 + 10.0)
        assert result[1] == pytest.approx(200.0 * 0.4 + 20.0)
        assert result[2] == pytest.approx(300.0 * 0.05 + 30.0)

    def test_physical_to_voxel(self):
        vcs = self._vcs()
        result = physical_to_voxel((50.0, 100.0, 45.0), vcs)
        # voxel = (physical - origin) / resolution
        assert result[0] == pytest.approx((50.0 - 10.0) / 0.4)
        assert result[1] == pytest.approx((100.0 - 20.0) / 0.4)
        assert result[2] == pytest.approx((45.0 - 30.0) / 0.05)

    def test_roundtrip(self):
        vcs = self._vcs()
        original = (100.0, 200.0, 300.0)
        physical = voxel_to_physical(original, vcs)
        back = physical_to_voxel(physical, vcs)
        assert back[0] == pytest.approx(original[0])
        assert back[1] == pytest.approx(original[1])
        assert back[2] == pytest.approx(original[2])
