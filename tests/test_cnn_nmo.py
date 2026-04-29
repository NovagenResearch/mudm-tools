"""Tests for cnn-nmo batch builder helpers."""
from __future__ import annotations

from mudm_tools.cnn_nmo import classify_source, build_feature_collection


def test_classify_source_hust():
    assert classify_source(
        "This reconstruction was obtained from the HUST-Suzhou Institute for "
        "Brainsmatics, and it is registered to the Common Coordinate "
        "Framework (CCF)."
    ) == "HUST"


def test_classify_source_allen():
    assert classify_source(
        "This reconstruction was obtained from the Allen Institute for Brain "
        "Science, and it is registered to the Common Coordinate Framework "
        "(CCF)."
    ) == "Allen"


def test_classify_source_seu_allen():
    assert classify_source(
        "This reconstruction was obtained from SEU-ALLEN, ..."
    ) == "SEU-Allen"


def test_classify_source_unknown():
    assert classify_source("") == "unknown"
    assert classify_source(None) == "unknown"


def test_build_feature_collection_tags_and_parentid(tmp_path):
    """Every compartment feature must carry parentId=neuron_id and identity tags."""
    swc = tmp_path / "demo.swc"
    swc.write_text(
        "1 1 0.0 0.0 0.0 8.0 -1\n"
        "2 3 10.0 5.0 2.0 2.0 1\n"
        "3 3 20.0 10.0 5.0 1.5 2\n"
        "4 2 -5.0 -3.0 1.0 1.0 1\n"
    )
    coll = build_feature_collection(
        swc_path=swc,
        neuron_id=42,
        neuron_name="demo",
        archive="BICCN-MOp-miniatlas-anatomy",
        source="HUST",
    )
    assert coll.properties["name"] == "demo"
    assert len(coll.features) >= 2
    for f in coll.features:
        assert f.parentId == 42
        t = f.properties
        assert t["neuron_id"] == 42
        assert t["neuron_name"] == "demo"
        assert t["source"] == "HUST"
        assert t["archive"] == "BICCN-MOp-miniatlas-anatomy"
        assert t["compartment"] in {"soma", "axon", "basal_dendrite",
                                    "apical_dendrite"}


def test_build_smoke(tmp_path, monkeypatch):
    """Build HUST-CCF scene from a 3-neuron in-memory manifest + stub cortex."""
    from pathlib import Path
    import json

    from mudm_tools.cnn_nmo import run_build

    # Stage three tiny SWCs + a metadata.json like the real pipeline expects
    data = tmp_path / "data" / "cnn-nmo"
    data.mkdir(parents=True)
    for i in range(1, 4):
        (data / f"n{i}.swc").write_text(
            "1 1 0.0 0.0 0.0 8.0 -1\n"
            f"2 3 {i*10.0} 5.0 2.0 2.0 1\n"
            f"3 3 {i*20.0} 10.0 5.0 1.5 2\n"
        )

    note_hust = (
        "This reconstruction was obtained from the HUST-Suzhou Institute for "
        "Brainsmatics, and it is registered to the Common Coordinate "
        "Framework (CCF)."
    )
    meta = {
        f"n{i}.swc": {
            "neuron": {
                "neuron_id": i, "neuron_name": f"n{i}",
                "archive": "BICCN-MOp-miniatlas-anatomy", "note": note_hust,
                "species": "mouse",
                "brain_region": ["neocortex"], "cell_type": ["pyramidal"],
            },
            "morphometry": {"surface": 1.0, "volume": 1.0, "length": 10.0},
        }
        for i in range(1, 4)
    }
    (data / "metadata.json").write_text(json.dumps(meta))

    ccf_stub = Path(__file__).parent / "fixtures" / "ccf_stub.obj"

    run_build(
        data_dir=data,
        tiles_dir=data / "tiles",
        cortex_obj=ccf_stub,
        scene_id="hust-ccf",
    )

    assert (data / "tiles" / "neurons_meta.parquet").exists()
    assert (data / "tiles" / "pyramids.json").exists()
    assert (data / "tiles" / "hust-ccf" / "3dtiles" / "tileset.json").exists()
    assert (data / "tiles" / "hust-ccf" / "3dtiles" / "tilejson3d.json").exists()
    assert (data / "tiles" / "hust-ccf" / "3dtiles" / "features.json").exists()
    assert (data / "tiles" / "hust-ccf" / "neuroglancer" / "skeletons" / "info").exists()
    assert (data / "tiles" / "hust-ccf" / "geom.parquet").exists()

    import json
    import pyarrow.parquet as pq

    # pyramids.json has one pyramid entry with the expected id
    pyramids = json.loads((data / "tiles" / "pyramids.json").read_text())
    assert len(pyramids["pyramids"]) == 1
    py = pyramids["pyramids"][0]
    assert py["id"] == "hust-ccf"
    assert py["max_zoom"] == 4
    assert py["tiles"] > 0
    assert py["features"] >= 4   # 3 neurons × ≥1 compartment + cortex

    # features.json has per-compartment entries keyed by "<name>/<compartment>"
    feats_doc = json.loads(
        (data / "tiles" / "hust-ccf" / "3dtiles" / "features.json").read_text()
    )
    assert "features" in feats_doc
    names = set(feats_doc["features"].keys())
    # Cortex feature key uses its neuron_name "CCF Isocortex"
    assert any(n.endswith("/cortex") for n in names), f"no cortex feature; got {sorted(names)}"
    assert any(n.startswith("n1/") for n in names), f"no n1/* feature; got {sorted(names)}"
    for feat in feats_doc["features"].values():
        assert "tiles" in feat  # per-zoom tile list

    # tilejson3d has canonical minzoom/maxzoom/bounds3d
    tj = json.loads(
        (data / "tiles" / "hust-ccf" / "3dtiles" / "tilejson3d.json").read_text()
    )
    assert tj["tilejson"] == "3.0.0"
    assert tj["minzoom"] == 0 and tj["maxzoom"] == 4
    assert len(tj["bounds3d"]) == 6

    # sidecar has all 3 neurons, all classified HUST
    tbl = pq.read_table(data / "tiles" / "neurons_meta.parquet")
    assert tbl.num_rows == 3
    assert set(tbl["source"].to_pylist()) == {"HUST"}
    assert set(tbl["in_ccf_frame"].to_pylist()) == {True}

    # tileset.json is a JSON document with a "root" key
    ts = json.loads(
        (data / "tiles" / "hust-ccf" / "3dtiles" / "tileset.json").read_text()
    )
    assert "root" in ts

    # geom.parquet has at least one row
    geom = pq.read_table(data / "tiles" / "hust-ccf" / "geom.parquet")
    assert geom.num_rows > 0


def test_build_rejects_duplicate_ids(tmp_path):
    """Two SWCs resolving to the same neuron_id (or 0) must be captured in build_errors."""
    from mudm_tools.cnn_nmo import run_build
    import json

    data = tmp_path / "data"
    data.mkdir()
    # Two SWCs, both with missing neuron_id (null -> 0 after coercion)
    for name in ("a.swc", "b.swc"):
        (data / name).write_text(
            "1 1 0.0 0.0 0.0 8.0 -1\n"
            "2 3 10.0 5.0 2.0 2.0 1\n"
        )
    note_hust = (
        "This reconstruction was obtained from the HUST-Suzhou Institute for "
        "Brainsmatics, and it is registered to the Common Coordinate "
        "Framework (CCF)."
    )
    (data / "metadata.json").write_text(json.dumps({
        "a.swc": {"neuron": {"neuron_id": None, "neuron_name": "a",
                             "note": note_hust, "archive": "x"},
                  "morphometry": {}},
        "b.swc": {"neuron": {"neuron_id": None, "neuron_name": "b",
                             "note": note_hust, "archive": "x"},
                  "morphometry": {}},
    }))

    run_build(data_dir=data, tiles_dir=data / "tiles",
              cortex_obj=None, scene_id="hust-ccf")

    errors_path = data / "tiles" / "build_errors.json"
    assert errors_path.exists(), "expected build_errors.json for missing ids"
    errs = json.loads(errors_path.read_text())
    assert len(errs) == 2
    assert {e["file"] for e in errs} == {"a.swc", "b.swc"}
