#!/usr/bin/env python3
"""Export SWC file(s) to MuDM JSON.

Each neuron becomes a FeatureCollection with one Feature per compartment
(soma, axon, basal_dendrite, apical_dendrite), each carrying a
``compartment`` property.

Usage:
    .venv/bin/python scripts/export_json.py [SWC_FILES...] [OPTIONS]

Options:
    -o OUTPUT       Output file path (default: stdout)

Examples:
    # Single neuron to stdout
    .venv/bin/python scripts/export_json.py swcs/cnic_041.CNG.swc

    # Save to file
    .venv/bin/python scripts/export_json.py swcs/cnic_041.CNG.swc -o neuron.json

    # Multiple neurons
    .venv/bin/python scripts/export_json.py swcs/*.swc -o neurons.json
"""

import sys
from pathlib import Path

from mudm_tools.swc import swc_to_feature_collection
from mudm.model import MuDMFeatureCollection


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

    out_path = _pop_flag(args, "-o") or _pop_flag(args, "--output")

    if not args:
        swc_dir = Path("swcs")
        swc_paths = sorted(swc_dir.glob("*.swc"))
        if not swc_paths:
            print(f"No SWC files found in {swc_dir}/", file=sys.stderr)
            sys.exit(1)
    else:
        swc_paths = [Path(a) for a in args]

    for p in swc_paths:
        if not p.exists():
            print(f"Error: SWC file not found: {p}", file=sys.stderr)
            sys.exit(1)

    collections = []
    for swc_path in swc_paths:
        coll = swc_to_feature_collection(str(swc_path))
        collections.append(coll)

    if len(collections) == 1:
        result = collections[0]
    else:
        # Flatten into one collection, tagging each feature with its neuron
        all_features = []
        for coll in collections:
            neuron_name = (coll.properties or {}).get("name", "unknown")
            for feat in coll.features:
                if feat.properties is None:
                    feat.properties = {}
                feat.properties["neuron"] = neuron_name
                all_features.append(feat)
        result = MuDMFeatureCollection(
            type="FeatureCollection",
            features=all_features,
            properties={"neuron_count": len(collections)},
        )

    json_str = result.model_dump_json(indent=2, exclude_none=True)

    if out_path:
        Path(out_path).write_text(json_str)
        n_feat = len(result.features) if hasattr(result, "features") else 1
        size_kb = len(json_str) / 1024
        print(
            f"Wrote {out_path} ({size_kb:.0f} KB)"
            f" — {len(swc_paths)} neuron(s), {n_feat} feature(s)",
            file=sys.stderr,
        )
    else:
        print(json_str)


if __name__ == "__main__":
    main()
