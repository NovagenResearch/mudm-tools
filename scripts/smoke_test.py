#!/usr/bin/env python3
"""Smoke test for mudm-tools package installation.

Usage:
    python scripts/smoke_test.py                # test pure-Python only
    python scripts/smoke_test.py --require-rust  # also test Rust acceleration
"""

import json
import sys

from mudm.model import MuDM

# Validate a simple feature (via core mudm package)
data = {
    "type": "FeatureCollection",
    "features": [{
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [10, 20]},
        "properties": {"label": "test"},
    }],
}
obj = MuDM.model_validate(data)
print(f"mudm core OK: {len(obj.root.features)} feature(s)")

import mudm_tools

print(f"Rust available: {mudm_tools.RUST_AVAILABLE}")

if "--require-rust" in sys.argv:
    if not mudm_tools.RUST_AVAILABLE:
        print("FAIL: --require-rust specified but Rust extension not found")
        sys.exit(1)
    from mudm_tools._rs import StreamingTileGenerator, StreamingTileGenerator2D

    gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=2, buffer=0.0)
    bounds = (0.0, 0.0, 100.0, 100.0)
    gen.add_geojson(json.dumps(data), bounds)
    print(f"2D generator OK: {gen.feature_count_val()} feature(s)")

    gen3d = StreamingTileGenerator(min_zoom=0, max_zoom=2)
    print("3D generator OK")

print("All checks passed.")
