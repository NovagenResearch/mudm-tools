"""Example: Export MuDM neuron morphologies to Neuroglancer precomputed format.

Demonstrates three usage patterns:
  1. Single skeleton export (write_skeleton)
  2. Mixed collection export (to_neuroglancer orchestrator)
  3. Viewer URL generation

Run:
    python -m mudm.examples.neuroglancer_export [--output-dir OUTPUT_DIR]

This creates a Neuroglancer-compatible directory structure that can be
served via HTTP and loaded with `precomputed://http://localhost:8080/...`
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mudm.model import (
    MuDMFeature,
    MuDMFeatureCollection,
)
from mudm_tools.swc import NeuronMorphology, SWCSample
from mudm_tools.neuroglancer import (
    to_neuroglancer,
    write_skeleton,
)
from mudm_tools.neuroglancer.state import (
    build_annotation_layer,
    build_skeleton_layer,
    build_viewer_state,
    viewer_state_to_url,
)

# ---------------------------------------------------------------------------
# Sample data — coordinates in micrometers (realistic neuron scale)
# ---------------------------------------------------------------------------


def make_pyramidal_neuron() -> NeuronMorphology:
    """Create a sample pyramidal neuron (soma + dendrites + axon).

    Coordinates are in micrometers — a typical neuron spans ~200-500 um.
    """
    return NeuronMorphology(
        type="NeuronMorphology",
        tree=[
            # Soma (center of the neuron)
            SWCSample(id=1, type=1, x=500.0, y=500.0, z=200.0, r=15.0, parent=-1),
            # Basal dendrite branch 1
            SWCSample(id=2, type=3, x=550.0, y=520.0, z=210.0, r=4.0, parent=1),
            SWCSample(id=3, type=3, x=620.0, y=560.0, z=230.0, r=3.0, parent=2),
            SWCSample(id=4, type=3, x=700.0, y=580.0, z=250.0, r=2.0, parent=3),
            # Basal dendrite branch 2 (fork from node 3)
            SWCSample(id=5, type=3, x=650.0, y=620.0, z=240.0, r=2.5, parent=3),
            # Axon (extends opposite direction)
            SWCSample(id=6, type=2, x=460.0, y=470.0, z=190.0, r=2.0, parent=1),
            SWCSample(id=7, type=2, x=380.0, y=420.0, z=170.0, r=1.5, parent=6),
            SWCSample(id=8, type=2, x=280.0, y=360.0, z=150.0, r=1.0, parent=7),
        ],
    )


def make_interneuron() -> NeuronMorphology:
    """Create a simpler interneuron example. Coordinates in micrometers."""
    return NeuronMorphology(
        type="NeuronMorphology",
        tree=[
            SWCSample(id=1, type=1, x=800.0, y=600.0, z=200.0, r=10.0, parent=-1),
            SWCSample(id=2, type=3, x=860.0, y=640.0, z=220.0, r=3.0, parent=1),
            SWCSample(id=3, type=3, x=930.0, y=680.0, z=250.0, r=2.0, parent=2),
            SWCSample(id=4, type=2, x=740.0, y=560.0, z=180.0, r=2.0, parent=1),
            SWCSample(id=5, type=2, x=670.0, y=520.0, z=160.0, r=1.5, parent=4),
        ],
    )


# ---------------------------------------------------------------------------
# Example 1: Single skeleton export
# ---------------------------------------------------------------------------


def example_single_skeleton(output_dir: Path) -> None:
    """Export a single neuron as a Neuroglancer skeleton."""
    print("=== Example 1: Single skeleton export ===")

    skel_dir = output_dir / "single_skeleton"
    neuron = make_pyramidal_neuron()

    write_skeleton(skel_dir, segment_id=1, morphology=neuron)

    print(f"  Written to: {skel_dir}/")
    print("  Files: info (JSON), 1 (binary skeleton)")
    print()


# ---------------------------------------------------------------------------
# Example 2: Mixed collection export (orchestrator)
# ---------------------------------------------------------------------------


def example_mixed_collection(output_dir: Path) -> None:
    """Export a collection with neurons + point markers + line traces."""
    print("=== Example 2: Mixed collection export ===")

    mixed_dir = output_dir / "mixed_collection"

    # Write neuron skeletons directly (NeuronMorphology → precomputed skeleton)
    skel_dir = mixed_dir / "skeletons"
    write_skeleton(skel_dir, segment_id=1, morphology=make_pyramidal_neuron())
    write_skeleton(skel_dir, segment_id=2, morphology=make_interneuron())

    # Build a collection with Point/LineString annotations
    collection = MuDMFeatureCollection(
        type="FeatureCollection",
        features=[
            # Soma markers (Point annotations)
            MuDMFeature(
                type="Feature",
                geometry={"type": "Point", "coordinates": [500, 500, 200]},
                properties={"label": "pyramidal_soma"},
            ),
            MuDMFeature(
                type="Feature",
                geometry={"type": "Point", "coordinates": [800, 600, 200]},
                properties={"label": "interneuron_soma"},
            ),
            # A traced fiber (LineString annotation)
            MuDMFeature(
                type="Feature",
                geometry={
                    "type": "LineString",
                    "coordinates": [
                        [500, 500, 200],
                        [650, 550, 200],
                        [800, 600, 200],
                    ],
                },
                properties={"label": "connecting_fiber"},
            ),
        ],
    )

    # Export Point/LineString annotations via orchestrator
    result = to_neuroglancer(collection, mixed_dir)

    print("  Output paths:")
    print(f"    skeletons: {skel_dir}/")
    for key, path in result["paths"].items():
        print(f"    {key}: {path}/")
    print()


# ---------------------------------------------------------------------------
# Example 3: Viewer URL generation
# ---------------------------------------------------------------------------


def example_viewer_url(output_dir: Path) -> None:
    """Generate a Neuroglancer viewer state URL."""
    print("=== Example 3: Viewer state URL ===")

    # Build URLs manually for full control over the viewer state.
    # The source URLs use precomputed:// scheme pointing to the local server.
    layers = [
        build_skeleton_layer(
            "neurons",
            "precomputed://http://localhost:9000/mixed_collection/skeletons",
        ),
        build_annotation_layer(
            "point_markers",
            "precomputed://http://localhost:9000/mixed_collection/point_annotations",
        ),
        build_annotation_layer(
            "fibers",
            "precomputed://http://localhost:9000/mixed_collection/line_annotations",
        ),
    ]

    # Center the view on the midpoint of the data
    state = build_viewer_state(layers, position=[650.0, 550.0, 200.0])

    # Pre-select skeleton segments so they're visible immediately
    state["layers"][0]["segments"] = ["1", "2"]

    url = viewer_state_to_url(state)

    print(f"  Viewer URL: {url}")
    print()
    print("  Copy-paste this URL into your browser after starting the server.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MuDM to Neuroglancer precomputed format")
    parser.add_argument(
        "--output-dir",
        default="neuroglancer_output",
        help="Output directory (default: neuroglancer_output)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    example_single_skeleton(output_dir)
    example_mixed_collection(output_dir)
    example_viewer_url(output_dir)

    print(f"All examples written to: {output_dir}/")
    print()
    print("To view in Neuroglancer:")
    print(f"  1. python -m mudm.examples.neuroglancer_serve {output_dir}")
    print("  2. Copy the URL printed by Example 3 above into your browser")


if __name__ == "__main__":
    main()
