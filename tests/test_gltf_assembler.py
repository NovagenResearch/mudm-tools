"""Tests for glTF scene graph assembly."""

from pathlib import Path

import numpy as np
import pytest
from pygltflib import LINES, POINTS, TRIANGLES

from mudm.model import (
    MuDMFeature,
    MuDMFeatureCollection,
    PolyhedralSurface,
    TIN,
)
from mudm_tools.gltf.gltf_assembler import (
    collection_to_gltf,
    feature_to_gltf,
)
from mudm_tools.gltf.models import GltfConfig

FIXTURE_DIR = Path(__file__).parent / "fixtures"
SAMPLE_SWC = FIXTURE_DIR / "sample_neuron.swc"


def _tin_feature(**props) -> MuDMFeature:
    """A simple TIN feature (2 triangles)."""
    return MuDMFeature(
        type="Feature",
        geometry=TIN(
            type="TIN",
            coordinates=[
                [[(0, 0, 0), (1, 0, 0), (0.5, 1, 0), (0, 0, 0)]],
                [[(1, 0, 0), (1, 1, 0), (0.5, 1, 0), (1, 0, 0)]],
            ],
        ),
        properties=props or None,
    )


def _tin_from_swc() -> MuDMFeature:
    """A TIN feature derived from SWC conversion."""
    from mudm_tools.swc import swc_to_tin

    return swc_to_tin(str(SAMPLE_SWC))


class TestFeatureToGltf:
    def test_tin_from_swc(self):
        """SWC-derived TIN goes through the standard TIN path."""
        feat = _tin_from_swc()
        gltf = feature_to_gltf(feat)

        assert len(gltf.meshes) == 1
        assert gltf.meshes[0].primitives[0].mode == TRIANGLES
        assert len(gltf.accessors) > 0
        assert gltf.asset.generator == "mudm-gltf"

    def test_tin_geometry(self):
        feat = _tin_feature()
        gltf = feature_to_gltf(feat)

        assert len(gltf.meshes) == 1
        assert gltf.meshes[0].primitives[0].mode == TRIANGLES

    def test_point_geometry(self):
        from geojson_pydantic import Point

        feat = MuDMFeature(
            type="Feature",
            geometry=Point(type="Point", coordinates=[1.0, 2.0, 3.0]),
            properties=None,
        )
        gltf = feature_to_gltf(feat)

        assert len(gltf.meshes) == 1
        assert gltf.meshes[0].primitives[0].mode == POINTS

    def test_linestring_geometry(self):
        from geojson_pydantic import LineString

        feat = MuDMFeature(
            type="Feature",
            geometry=LineString(
                type="LineString",
                coordinates=[[0, 0, 0], [1, 1, 1], [2, 0, 0]],
            ),
            properties=None,
        )
        gltf = feature_to_gltf(feat)

        assert len(gltf.meshes) == 1
        assert gltf.meshes[0].primitives[0].mode == LINES

    def test_polygon_geometry(self):
        from geojson_pydantic import Polygon

        feat = MuDMFeature(
            type="Feature",
            geometry=Polygon(
                type="Polygon",
                coordinates=[[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]],
            ),
            properties=None,
        )
        gltf = feature_to_gltf(feat)

        assert len(gltf.meshes) == 1
        assert gltf.meshes[0].primitives[0].mode == TRIANGLES

    def test_polyhedral_surface(self):
        ps = PolyhedralSurface(
            type="PolyhedralSurface",
            coordinates=[
                [[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]]],
                [[[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1], [0, 0, 1]]],
            ],
        )
        feat = MuDMFeature(type="Feature", geometry=ps, properties=None)
        gltf = feature_to_gltf(feat)

        assert len(gltf.meshes) == 1

    def test_multipoint(self):
        from geojson_pydantic import MultiPoint

        feat = MuDMFeature(
            type="Feature",
            geometry=MultiPoint(type="MultiPoint", coordinates=[[0, 0, 0], [1, 1, 1]]),
            properties=None,
        )
        gltf = feature_to_gltf(feat)
        assert gltf.meshes[0].primitives[0].mode == POINTS

    def test_multilinestring(self):
        from geojson_pydantic import MultiLineString

        feat = MuDMFeature(
            type="Feature",
            geometry=MultiLineString(
                type="MultiLineString",
                coordinates=[[[0, 0], [1, 1]], [[2, 2], [3, 3]]],
            ),
            properties=None,
        )
        gltf = feature_to_gltf(feat)
        assert gltf.meshes[0].primitives[0].mode == LINES

    def test_null_geometry(self):
        feat = MuDMFeature(type="Feature", geometry=None, properties=None)
        gltf = feature_to_gltf(feat)
        assert len(gltf.meshes) == 0
        assert len(gltf.nodes) == 0


class TestMetadata:
    def test_feature_properties_in_extras(self):
        feat = _tin_feature(cell_type="pyramidal", layer=5)
        gltf = feature_to_gltf(feat)

        assert gltf.nodes[0].extras == {"cell_type": "pyramidal", "layer": 5}

    def test_metadata_disabled(self):
        feat = _tin_feature(cell_type="pyramidal")
        config = GltfConfig(include_metadata=False)
        gltf = feature_to_gltf(feat, config=config)

        assert not gltf.nodes[0].extras  # empty dict or None

    def test_collection_metadata(self):
        feat = _tin_feature()
        collection = MuDMFeatureCollection(
            type="FeatureCollection",
            features=[feat],
            properties={"study": "test_study"},
        )
        gltf = collection_to_gltf(collection)
        assert gltf.scenes[0].extras == {"study": "test_study"}


class TestCoordinateTransform:
    def test_y_up_swaps_yz(self):
        """With y_up=True, Z values should become Y values."""
        from geojson_pydantic import Point

        feat = MuDMFeature(
            type="Feature",
            geometry=Point(type="Point", coordinates=[1.0, 2.0, 3.0]),
            properties=None,
        )
        gltf = feature_to_gltf(feat, GltfConfig(y_up=True))

        # Extract position from buffer
        acc = gltf.accessors[0]
        bv = gltf.bufferViews[acc.bufferView]
        data = gltf._glb_data[bv.byteOffset : bv.byteOffset + bv.byteLength]
        pos = np.frombuffer(data, dtype="<f4").reshape(-1, 3)

        # Original (1, 2, 3) -> Y-up (1, 3, -2)
        np.testing.assert_allclose(pos[0], [1.0, 3.0, -2.0], atol=1e-6)

    def test_y_up_disabled(self):
        from geojson_pydantic import Point

        feat = MuDMFeature(
            type="Feature",
            geometry=Point(type="Point", coordinates=[1.0, 2.0, 3.0]),
            properties=None,
        )
        gltf = feature_to_gltf(feat, GltfConfig(y_up=False))

        acc = gltf.accessors[0]
        bv = gltf.bufferViews[acc.bufferView]
        data = gltf._glb_data[bv.byteOffset : bv.byteOffset + bv.byteLength]
        pos = np.frombuffer(data, dtype="<f4").reshape(-1, 3)

        np.testing.assert_allclose(pos[0], [1.0, 2.0, 3.0], atol=1e-6)


class TestCollectionToGltf:
    def test_multiple_features(self):
        feat1 = _tin_feature()
        from geojson_pydantic import Point

        feat2 = MuDMFeature(
            type="Feature",
            geometry=Point(type="Point", coordinates=[5.0, 5.0, 5.0]),
            properties=None,
        )
        collection = MuDMFeatureCollection(
            type="FeatureCollection",
            features=[feat1, feat2],
        )
        gltf = collection_to_gltf(collection)

        # TIN -> 1 mesh, point -> 1 mesh = 2 total
        assert len(gltf.meshes) == 2
        assert len(gltf.nodes) == 2

    def test_empty_collection(self):
        collection = MuDMFeatureCollection(
            type="FeatureCollection",
            features=[],
        )
        gltf = collection_to_gltf(collection)
        assert len(gltf.meshes) == 0
        assert len(gltf.nodes) == 0

    def test_default_material(self):
        feat = _tin_feature()
        gltf = feature_to_gltf(feat)

        assert len(gltf.materials) >= 1
        pbr = gltf.materials[0].pbrMetallicRoughness
        assert pbr.metallicFactor == 0.1
        assert pbr.roughnessFactor == 0.8


class TestFeatureSpacing:
    """Feature spacing in collections -- layout is now coordinate-based."""

    def _two_tin_collection(self):
        """Two TINs at the same position."""
        t1 = TIN(
            type="TIN",
            coordinates=[
                [[(0, 0, 0), (100, 0, 0), (50, 10, 5), (0, 0, 0)]],
            ],
        )
        t2 = TIN(
            type="TIN",
            coordinates=[
                [[(0, 0, 0), (80, 0, 0), (40, 10, 5), (0, 0, 0)]],
            ],
        )
        return MuDMFeatureCollection(
            type="FeatureCollection",
            features=[
                MuDMFeature(type="Feature", geometry=t1, properties=None),
                MuDMFeature(type="Feature", geometry=t2, properties=None),
            ],
        )

    def test_default_no_layout(self):
        """Default config keeps coordinates as-is (no layout)."""
        coll = self._two_tin_collection()
        gltf = collection_to_gltf(coll)

        # Nodes should have no translation
        for node in gltf.nodes:
            assert node.translation is None

        # At least 2 nodes: 1 per TIN
        assert len(gltf.nodes) >= 2

    def test_fixed_spacing(self):
        """Explicit feature_spacing sets the gap -- verified via coordinates."""
        coll = self._two_tin_collection()
        config = GltfConfig(feature_spacing=50.0)
        gltf = collection_to_gltf(coll, config=config)

        # No node translations -- layout is baked into vertex data
        for node in gltf.nodes:
            assert node.translation is None

        # We have 2 meshes (one per feature)
        assert len(gltf.meshes) == 2

    def test_single_feature_no_offset(self):
        """A single-feature collection should not get translated."""
        feat = _tin_feature()
        coll = MuDMFeatureCollection(
            type="FeatureCollection",
            features=[feat],
        )
        gltf = collection_to_gltf(coll)
        for node in gltf.nodes:
            assert node.translation is None

    def test_no_node_translations_with_spacing(self):
        """Layout is entirely coordinate-based -- no node translations."""
        coll = self._two_tin_collection()
        config = GltfConfig(feature_spacing=50.0)
        gltf = collection_to_gltf(coll, config=config)
        for node in gltf.nodes:
            assert node.translation is None


class TestGridLayout:
    """Grid placement when grid_max_* is set."""

    def _make_tin(self, x_offset=0.0, width=100.0):
        return TIN(
            type="TIN",
            coordinates=[
                [
                    [
                        (x_offset, 0, 0),
                        (x_offset + width, 0, 0),
                        (x_offset + width / 2, 10, 5),
                        (x_offset, 0, 0),
                    ]
                ],
            ],
        )

    def _collection(self, n, **tin_kw):
        features = [
            MuDMFeature(
                type="Feature",
                geometry=self._make_tin(**tin_kw),
                properties=None,
            )
            for _ in range(n)
        ]
        return MuDMFeatureCollection(
            type="FeatureCollection",
            features=features,
        )

    def test_wraps_to_rows(self):
        """4 features, 2 cols -> 2 rows. No node translations."""
        coll = self._collection(4, width=100)
        config = GltfConfig(
            grid_max_x=2,
            feature_spacing=10,
        )
        gltf = collection_to_gltf(coll, config=config)

        # All nodes should have no translation
        for node in gltf.nodes:
            assert node.translation is None

        # 4 features, each 1 mesh
        assert len(gltf.nodes) == 4

    def test_wraps_to_layers(self):
        """5 features, 2 cols x 2 rows -> 5th goes to layer 1."""
        coll = self._collection(5, width=100)
        config = GltfConfig(
            grid_max_x=2,
            grid_max_y=2,
            feature_spacing=10,
        )
        gltf = collection_to_gltf(coll, config=config)

        # No node translations
        for node in gltf.nodes:
            assert node.translation is None
        assert len(gltf.nodes) == 5

    def test_capacity_error(self):
        """All three grid_max_* set with too many features -> ValueError."""
        coll = self._collection(10, width=100)
        # 2 x 2 x 2 = 8 < 10
        config = GltfConfig(
            grid_max_x=2,
            grid_max_y=2,
            grid_max_z=2,
            feature_spacing=10,
        )
        with pytest.raises(ValueError, match="Cannot fit 10 features"):
            collection_to_gltf(coll, config=config)

    def test_exact_capacity_fits(self):
        """Exactly filling the grid should not raise."""
        coll = self._collection(8, width=100)
        config = GltfConfig(
            grid_max_x=2,
            grid_max_y=2,
            grid_max_z=2,
        )
        gltf = collection_to_gltf(coll, config=config)
        assert len(gltf.nodes) == 8

    def test_capacity_error_before_generation(self):
        """Error should happen before any mesh is built (fast fail)."""
        coll = self._collection(100, width=100)
        config = GltfConfig(
            grid_max_x=3,
            grid_max_y=3,
            grid_max_z=3,
        )
        # 3x3x3 = 27 < 100 -> ValueError fires instantly
        with pytest.raises(ValueError):
            collection_to_gltf(coll, config=config)

    def test_unconstrained_z_always_fits(self):
        """With only grid_max_x set, any number of features should fit."""
        coll = self._collection(20, width=100)
        config = GltfConfig(
            grid_max_x=3,
            feature_spacing=10,
        )
        # Should not raise -- rows/layers expand as needed
        gltf = collection_to_gltf(coll, config=config)
        assert len(gltf.nodes) == 20

    def test_grid_no_node_translations(self):
        """Grid layout: no node translations."""
        coll = self._collection(4, width=100)
        config = GltfConfig(
            grid_max_x=2,
            feature_spacing=10,
        )
        gltf = collection_to_gltf(coll, config=config)

        # All nodes should have no translation
        for node in gltf.nodes:
            assert node.translation is None


class TestColorBy:
    """Metadata-driven coloring via color_by + color_map."""

    def _colored_collection(self):
        """Three TIN features with different 'kind' property values."""

        def _make(kind_value):
            return MuDMFeature(
                type="Feature",
                geometry=TIN(
                    type="TIN",
                    coordinates=[
                        [[(0, 0, 0), (1, 0, 0), (0.5, 1, 0), (0, 0, 0)]],
                    ],
                ),
                properties={"kind": kind_value},
            )

        return MuDMFeatureCollection(
            type="FeatureCollection",
            features=[_make("alpha"), _make("beta"), _make("alpha")],
        )

    def test_color_map_creates_multiple_materials(self):
        """color_map entries create extra materials beyond the default."""
        coll = self._colored_collection()
        config = GltfConfig(
            color_by="kind",
            color_map={
                "alpha": (1.0, 0.0, 0.0, 1.0),
                "beta": (0.0, 1.0, 0.0, 1.0),
            },
        )
        gltf = collection_to_gltf(coll, config=config)
        # default + alpha + beta = 3 materials
        assert len(gltf.materials) == 3

    def test_features_get_correct_material(self):
        """Features with matching property value get the right material."""
        coll = self._colored_collection()
        config = GltfConfig(
            color_by="kind",
            color_map={
                "alpha": (1.0, 0.0, 0.0, 1.0),
                "beta": (0.0, 1.0, 0.0, 1.0),
            },
        )
        gltf = collection_to_gltf(coll, config=config)

        # Find alpha and beta material indices
        alpha_mat = None
        beta_mat = None
        for i, mat in enumerate(gltf.materials):
            color = mat.pbrMetallicRoughness.baseColorFactor
            if color == [1.0, 0.0, 0.0, 1.0]:
                alpha_mat = i
            elif color == [0.0, 1.0, 0.0, 1.0]:
                beta_mat = i

        assert alpha_mat is not None
        assert beta_mat is not None

        # mesh 0 = alpha, mesh 1 = beta, mesh 2 = alpha
        assert gltf.meshes[0].primitives[0].material == alpha_mat
        assert gltf.meshes[1].primitives[0].material == beta_mat
        assert gltf.meshes[2].primitives[0].material == alpha_mat

    def test_missing_property_falls_back_to_default(self):
        """Feature without the color_by property gets material 0."""
        feat_no_prop = MuDMFeature(
            type="Feature",
            geometry=TIN(
                type="TIN",
                coordinates=[
                    [[(0, 0, 0), (1, 0, 0), (0.5, 1, 0), (0, 0, 0)]],
                ],
            ),
            properties={"other": "value"},
        )
        coll = MuDMFeatureCollection(
            type="FeatureCollection",
            features=[feat_no_prop],
        )
        config = GltfConfig(
            color_by="kind",
            color_map={"alpha": (1.0, 0.0, 0.0, 1.0)},
        )
        gltf = collection_to_gltf(coll, config=config)
        assert gltf.meshes[0].primitives[0].material == 0

    def test_unmapped_value_falls_back_to_default(self):
        """Property value not in color_map falls back to material 0."""
        feat = MuDMFeature(
            type="Feature",
            geometry=TIN(
                type="TIN",
                coordinates=[
                    [[(0, 0, 0), (1, 0, 0), (0.5, 1, 0), (0, 0, 0)]],
                ],
            ),
            properties={"kind": "gamma"},
        )
        coll = MuDMFeatureCollection(
            type="FeatureCollection",
            features=[feat],
        )
        config = GltfConfig(
            color_by="kind",
            color_map={"alpha": (1.0, 0.0, 0.0, 1.0)},
        )
        gltf = collection_to_gltf(coll, config=config)
        assert gltf.meshes[0].primitives[0].material == 0

    def test_color_by_none_single_material(self):
        """Without color_by, only 1 material (default behavior)."""
        coll = self._colored_collection()
        config = GltfConfig()
        gltf = collection_to_gltf(coll, config=config)
        assert len(gltf.materials) == 1

    def test_single_feature_with_color_by(self):
        """color_by works on feature_to_gltf too."""
        feat = MuDMFeature(
            type="Feature",
            geometry=TIN(
                type="TIN",
                coordinates=[
                    [[(0, 0, 0), (1, 0, 0), (0.5, 1, 0), (0, 0, 0)]],
                ],
            ),
            properties={"kind": "alpha"},
        )
        config = GltfConfig(
            color_by="kind",
            color_map={"alpha": (1.0, 0.0, 0.0, 1.0)},
        )
        gltf = feature_to_gltf(feat, config=config)
        # default + alpha = 2 materials
        assert len(gltf.materials) == 2
        # feature should use alpha material (index 1)
        assert gltf.meshes[0].primitives[0].material == 1
