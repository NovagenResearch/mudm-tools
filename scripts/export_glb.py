#!/usr/bin/env python3
"""Export SWC file(s) to a single GLB for visualization.

Usage:
    .venv/bin/python scripts/export_glb.py [SWC_FILES...] [OPTIONS]

Options:
    -o OUTPUT       Output file path (default: neuron.glb)
    --grid X,Y,Z    Grid layout, e.g. --grid 5,4 or --grid 5,4,2
    --spacing N     Gap between features in source units (default: 0 = auto 20%)
    --draco         Enable Draco mesh compression (requires DracoPy)
    --color         Color by SWC compartment type (soma, axon, dendrites)

Examples:
    # All SWC files in swcs/ (single row)
    .venv/bin/python scripts/export_glb.py

    # 5 columns, unlimited rows
    .venv/bin/python scripts/export_glb.py --grid 5

    # 5×4 grid
    .venv/bin/python scripts/export_glb.py --grid 5,4

    # 5×4×2 grid (max 40 cells)
    .venv/bin/python scripts/export_glb.py --grid 5,4,2

Then drag the .glb file into https://gltf-viewer.donmccurdy.com/
"""

import sys
from pathlib import Path

from mudm.model import MuDMFeature, MuDMFeatureCollection
from mudm_tools.swc import swc_to_microjson, swc_to_feature_collection
from mudm_tools.gltf import to_glb, GltfConfig
from mudm.layout import compute_collection_offsets
from mudm.transforms import translate_geometry

DEFAULT_SWC_DIR = "swcs"
DEFAULT_OUT = "neuron.glb"


def _pop_flag(args, flag):
    """Remove ``flag VALUE`` from *args*, return VALUE or None."""
    if flag in args:
        idx = args.index(flag)
        val = args[idx + 1]
        del args[idx:idx + 2]
        return val
    return None


def main():
    args = sys.argv[1:]

    out_path = _pop_flag(args, "-o") or DEFAULT_OUT
    out_path = _pop_flag(args, "--output") or out_path
    grid_str = _pop_flag(args, "--grid")
    spacing_str = _pop_flag(args, "--spacing")
    feature_spacing = float(spacing_str) if spacing_str else 0.0
    draco = "--draco" in args
    if draco:
        args.remove("--draco")
    color = "--color" in args
    if color:
        args.remove("--color")

    # Parse grid spec
    grid_x = grid_y = grid_z = None
    if grid_str:
        parts = [int(x) for x in grid_str.split(",")]
        grid_x = parts[0]
        if len(parts) > 1:
            grid_y = parts[1]
        if len(parts) > 2:
            grid_z = parts[2]

    # Collect SWC paths
    if args:
        swc_paths = [Path(a) for a in args]
    else:
        swc_dir = Path(DEFAULT_SWC_DIR)
        swc_paths = sorted(swc_dir.glob("*.swc"))
        if not swc_paths:
            print(f"No SWC files found in {swc_dir}/")
            sys.exit(1)

    for p in swc_paths:
        if not p.exists():
            print(f"Error: SWC file not found: {p}")
            sys.exit(1)

    # SWC compartment color map
    swc_color_map = {
        "soma": (1.0, 0.0, 0.0, 1.0),
        "axon": (0.5, 0.5, 0.5, 1.0),
        "basal_dendrite": (0.0, 1.0, 0.0, 1.0),
        "apical_dendrite": (1.0, 0.0, 1.0, 1.0),
    }

    if color:
        # --- Color mode: lay out whole neurons, then split into compartments ---
        # 1. Build one monochrome feature per SWC (for layout calculation)
        mono_features = []
        for swc_path in swc_paths:
            mono_features.append(
                swc_to_microjson(str(swc_path), smooth_subdivisions=10, mesh_quality=0.3)
            )

        # 2. Compute per-neuron layout offsets
        offsets = compute_collection_offsets(
            mono_features,
            spacing=feature_spacing,
            grid_max_x=grid_x,
            grid_max_y=grid_y,
            grid_max_z=grid_z,
        )

        # 3. Generate colored compartments and translate by neuron offset
        all_features: list[MuDMFeature] = []
        for swc_path, (dx, dy, dz) in zip(swc_paths, offsets):
            coll = swc_to_feature_collection(
                str(swc_path), smooth_subdivisions=10, mesh_quality=0.3,
            )
            for feat in coll.features:
                if abs(dx) > 1e-12 or abs(dy) > 1e-12 or abs(dz) > 1e-12:
                    new_geom = translate_geometry(feat.geometry, dx, dy, dz)
                    feat = feat.model_copy(update={"geometry": new_geom})
                if feat.properties is None:
                    feat.properties = {}
                feat.properties["source"] = swc_path.name
                all_features.append(feat)

        # 4. Export without layout (default feature_spacing=None = no layout)
        config = GltfConfig(
            draco=draco,
            color_by="compartment",
            color_map=swc_color_map,
        )
    else:
        # --- Standard mode: one feature per SWC, normal layout ---
        config = GltfConfig(
            grid_max_x=grid_x,
            grid_max_y=grid_y,
            grid_max_z=grid_z,
            feature_spacing=feature_spacing,
            draco=draco,
        )
        all_features = []
        for swc_path in swc_paths:
            feat = swc_to_microjson(
                str(swc_path), smooth_subdivisions=10, mesh_quality=0.3,
            )
            if feat.properties is None:
                feat.properties = {}
            feat.properties["source"] = swc_path.name
            all_features.append(feat)

    print(f"Loaded {len(swc_paths)} SWC files ({len(all_features)} features)")

    if len(all_features) == 1:
        data = to_glb(
            all_features[0], config=config, output_path=out_path,
        )
    else:
        coll = MuDMFeatureCollection(
            type="FeatureCollection",
            features=all_features,
            properties={"neuron_count": len(swc_paths)},
        )
        data = to_glb(coll, config=config, output_path=out_path)

    mb = len(data) / 1024 / 1024
    grid_info = ""
    if grid_x:
        dims = [str(grid_x)]
        if grid_y:
            dims.append(str(grid_y))
        if grid_z:
            dims.append(str(grid_z))
        grid_info = f"  grid={'x'.join(dims)}"

    color_info = "  color=compartment" if color else ""
    print(
        f"Wrote {out_path} ({mb:.1f} MB)"
        f" — {len(swc_paths)} neuron(s), {len(all_features)} feature(s){grid_info}{color_info}"
    )
    print(
        "Open https://gltf-viewer.donmccurdy.com/"
        f" and drag in {out_path}"
    )


if __name__ == "__main__":
    main()
