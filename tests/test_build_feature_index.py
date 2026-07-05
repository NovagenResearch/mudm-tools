"""build_feature_index.feature_properties keeps all scalar extras (no hardcoded allowlist)."""

import build_feature_index as bfi


def test_feature_properties_keeps_all_scalars_drops_structural():
    extras = {
        "name": "nucleus-1",
        "color": "#abc123",
        "acronym": "CP",
        "ccf_id": 672,
        "volume_um3": 206.06,
        "emt_lean": 0.016,
        "sphericity": 0.82,
        "flag": True,
        "tile_ids": ["2/0/0/0"],
        "vocab": {"a": 1},
        "tags": [1, 2, 3],
    }
    props = bfi.feature_properties(extras)
    # custom scalars survive (the regression this fixes) alongside the connectome scalars
    assert props == {
        "acronym": "CP",
        "ccf_id": 672,
        "volume_um3": 206.06,
        "emt_lean": 0.016,
        "sphericity": 0.82,
        "flag": True,
    }
    # structural / id / non-scalar fields are excluded
    for key in ("name", "color", "tile_ids", "vocab", "tags"):
        assert key not in props
