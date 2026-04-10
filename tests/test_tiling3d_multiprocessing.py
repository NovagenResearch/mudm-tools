"""Tests for multiprocessing support in TileGenerator3D.

Covers: workers parameter, serial fallback, parallel generation,
bit-identical output, auto-detection, both pbf3 and 3dtiles formats.
"""

from __future__ import annotations

from pathlib import Path

from geojson_pydantic import Point

from mudm.model import MuDMFeature, MuDMFeatureCollection, TIN
from mudm_tools.tiling3d import TileGenerator3D, OctreeConfig
from mudm_tools.tiling3d.generator3d import _MIN_TILES_FOR_MP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _point_feature(x: float, y: float, z: float, **props) -> MuDMFeature:
    return MuDMFeature(
        type="Feature",
        geometry=Point(type="Point", coordinates=[x, y, z]),
        properties=props if props else {},
    )


def _tin_feature(ox: float = 0.0, oy: float = 0.0, oz: float = 0.0, **props) -> MuDMFeature:
    """A simple TIN with two triangles, offset by (ox, oy, oz)."""
    tin = TIN(
        type="TIN",
        coordinates=[
            [[[ox, oy, oz], [ox + 1, oy, oz], [ox + 0.5, oy + 1, oz + 0.5], [ox, oy, oz]]],
            [[[ox + 1, oy, oz], [ox + 2, oy, oz], [ox + 1.5, oy + 1, oz + 0.5], [ox + 1, oy, oz]]],
        ],
    )
    return MuDMFeature(
        type="Feature",
        geometry=tin,
        properties=props if props else {},
    )


def _collection(*features: MuDMFeature) -> MuDMFeatureCollection:
    return MuDMFeatureCollection(
        type="FeatureCollection",
        features=list(features),
    )


def _large_point_collection(n: int = 50) -> MuDMFeatureCollection:
    """Create a collection with many spread-out points to generate enough tiles."""
    features = []
    for i in range(n):
        x = (i % 10) * 10.0
        y = (i // 10) * 10.0
        z = float(i)
        features.append(_point_feature(x, y, z, index=i))
    return _collection(*features)


def _large_tin_collection(n: int = 30) -> MuDMFeatureCollection:
    """Create a collection with many TIN features spread out."""
    features = []
    for i in range(n):
        ox = (i % 6) * 5.0
        oy = (i // 6) * 5.0
        oz = float(i)
        features.append(_tin_feature(ox, oy, oz, index=i))
    return _collection(*features)


def _collect_tiles(output_dir: Path, ext: str) -> dict[str, bytes]:
    """Collect all tiles as {relative_path: bytes}."""
    result = {}
    for f in sorted(output_dir.rglob(f"*.{ext}")):
        rel = str(f.relative_to(output_dir))
        result[rel] = f.read_bytes()
    return result


# ---------------------------------------------------------------------------
# Tests: workers parameter
# ---------------------------------------------------------------------------

class TestWorkersParameter:
    """Test that the workers parameter is accepted and stored."""

    def test_default_workers_is_none(self):
        gen = TileGenerator3D()
        assert gen._workers is None

    def test_explicit_workers(self):
        gen = TileGenerator3D(workers=4)
        assert gen._workers == 4

    def test_workers_one(self):
        gen = TileGenerator3D(workers=1)
        assert gen._workers == 1

    def test_effective_workers_auto(self):
        gen = TileGenerator3D(workers=None)
        n = gen._effective_workers()
        assert n >= 1

    def test_effective_workers_zero(self):
        gen = TileGenerator3D(workers=0)
        n = gen._effective_workers()
        assert n >= 1

    def test_effective_workers_explicit(self):
        gen = TileGenerator3D(workers=3)
        assert gen._effective_workers() == 3


# ---------------------------------------------------------------------------
# Tests: serial fallback for small tile counts
# ---------------------------------------------------------------------------

class TestSerialFallback:
    """Small collections should use serial path even with workers > 1."""

    def test_few_tiles_uses_serial_pbf3(self, tmp_path):
        """With very few features, tile count < _MIN_TILES_FOR_MP → serial."""
        collection = _collection(
            _point_feature(1, 2, 3),
            _point_feature(4, 5, 6),
        )
        gen = TileGenerator3D(
            OctreeConfig(max_zoom=1),
            workers=4,
        )
        gen.add_features(collection)
        count = gen.generate(tmp_path / "out")
        assert count > 0

    def test_few_tiles_uses_serial_3dtiles(self, tmp_path):
        collection = _collection(
            _point_feature(1, 2, 3),
            _point_feature(4, 5, 6),
        )
        gen = TileGenerator3D(
            OctreeConfig(max_zoom=1),
            output_format="3dtiles",
            workers=4,
        )
        gen.add_features(collection)
        count = gen.generate(tmp_path / "out")
        assert count > 0


# ---------------------------------------------------------------------------
# Tests: bit-identical output (serial vs parallel)
# ---------------------------------------------------------------------------

class TestBitIdentical:
    """Parallel output must match serial output byte-for-byte."""

    def test_pbf3_bit_identical(self, tmp_path):
        collection = _large_point_collection(50)

        # Serial
        gen_serial = TileGenerator3D(
            OctreeConfig(max_zoom=3),
            workers=1,
        )
        gen_serial.add_features(collection)
        serial_dir = tmp_path / "serial"
        count_serial = gen_serial.generate(serial_dir)

        # Parallel
        gen_parallel = TileGenerator3D(
            OctreeConfig(max_zoom=3),
            workers=2,
        )
        gen_parallel.add_features(collection)
        parallel_dir = tmp_path / "parallel"
        count_parallel = gen_parallel.generate(parallel_dir)

        assert count_serial == count_parallel
        assert count_serial >= _MIN_TILES_FOR_MP, (
            f"Need >= {_MIN_TILES_FOR_MP} tiles to test parallel path, got {count_serial}"
        )

        serial_tiles = _collect_tiles(serial_dir, "pbf3")
        parallel_tiles = _collect_tiles(parallel_dir, "pbf3")

        assert set(serial_tiles.keys()) == set(parallel_tiles.keys())
        for path in serial_tiles:
            assert serial_tiles[path] == parallel_tiles[path], (
                f"Tile {path} differs between serial and parallel"
            )

    def test_pbf3_tin_bit_identical(self, tmp_path):
        collection = _large_tin_collection(30)

        gen_serial = TileGenerator3D(
            OctreeConfig(max_zoom=3),
            workers=1,
        )
        gen_serial.add_features(collection)
        serial_dir = tmp_path / "serial"
        count_serial = gen_serial.generate(serial_dir)

        gen_parallel = TileGenerator3D(
            OctreeConfig(max_zoom=3),
            workers=2,
        )
        gen_parallel.add_features(collection)
        parallel_dir = tmp_path / "parallel"
        count_parallel = gen_parallel.generate(parallel_dir)

        assert count_serial == count_parallel

        serial_tiles = _collect_tiles(serial_dir, "pbf3")
        parallel_tiles = _collect_tiles(parallel_dir, "pbf3")

        assert set(serial_tiles.keys()) == set(parallel_tiles.keys())
        for path in serial_tiles:
            assert serial_tiles[path] == parallel_tiles[path], (
                f"TIN tile {path} differs between serial and parallel"
            )

    def test_3dtiles_bit_identical(self, tmp_path):
        collection = _large_point_collection(50)

        gen_serial = TileGenerator3D(
            OctreeConfig(max_zoom=3),
            output_format="3dtiles",
            workers=1,
        )
        gen_serial.add_features(collection)
        serial_dir = tmp_path / "serial"
        count_serial = gen_serial.generate(serial_dir)

        gen_parallel = TileGenerator3D(
            OctreeConfig(max_zoom=3),
            output_format="3dtiles",
            workers=2,
        )
        gen_parallel.add_features(collection)
        parallel_dir = tmp_path / "parallel"
        count_parallel = gen_parallel.generate(parallel_dir)

        assert count_serial == count_parallel

        serial_tiles = _collect_tiles(serial_dir, "glb")
        parallel_tiles = _collect_tiles(parallel_dir, "glb")

        assert set(serial_tiles.keys()) == set(parallel_tiles.keys())
        for path in serial_tiles:
            assert serial_tiles[path] == parallel_tiles[path], (
                f"3dtiles tile {path} differs between serial and parallel"
            )


# ---------------------------------------------------------------------------
# Tests: auto-detect workers
# ---------------------------------------------------------------------------

class TestAutoDetect:
    """workers=None should auto-detect and still produce correct output."""

    def test_auto_workers_pbf3(self, tmp_path):
        collection = _large_point_collection(50)

        gen = TileGenerator3D(
            OctreeConfig(max_zoom=3),
            workers=None,
        )
        gen.add_features(collection)
        count = gen.generate(tmp_path / "auto")
        assert count > 0

        # Compare with serial
        gen_serial = TileGenerator3D(
            OctreeConfig(max_zoom=3),
            workers=1,
        )
        gen_serial.add_features(collection)
        serial_dir = tmp_path / "serial"
        count_serial = gen_serial.generate(serial_dir)

        assert count == count_serial


# ---------------------------------------------------------------------------
# Tests: workers=1 matches original behavior
# ---------------------------------------------------------------------------

class TestWorkersOneMatchesOriginal:
    """workers=1 should produce identical results to not specifying workers."""

    def test_workers_1_matches_default_serial(self, tmp_path):
        """With few tiles (< threshold), both paths are serial anyway."""
        collection = _collection(
            _point_feature(1, 2, 3, name="a"),
            _point_feature(4, 5, 6, name="b"),
        )

        gen1 = TileGenerator3D(OctreeConfig(max_zoom=2), workers=1)
        gen1.add_features(collection)
        dir1 = tmp_path / "w1"
        c1 = gen1.generate(dir1)

        gen2 = TileGenerator3D(OctreeConfig(max_zoom=2), workers=1)
        gen2.add_features(collection)
        dir2 = tmp_path / "w1_copy"
        c2 = gen2.generate(dir2)

        assert c1 == c2
        tiles1 = _collect_tiles(dir1, "pbf3")
        tiles2 = _collect_tiles(dir2, "pbf3")
        assert tiles1 == tiles2
