"""Tests for the per-neuron sidecar writer/reader."""
from __future__ import annotations

import pyarrow.parquet as pq
import pytest

from mudm_tools.neuron_meta import NeuronRecord, write_sidecar, load_sidecar


def _sample_records():
    return [
        NeuronRecord(
            neuron_id=1,
            neuron_name="test_a",
            archive="BICCN-MOp-miniatlas-anatomy",
            note="This reconstruction was obtained from the HUST-Suzhou Institute"
                 " for Brainsmatics, and it is registered to the Common"
                 " Coordinate Framework (CCF).",
            species="mouse",
            brain_region=["neocortex", "frontal", "primary motor"],
            cell_type=["pyramidal"],
            surface_m=1234.5,
            volume_m=678.9,
            source="HUST",
            in_ccf_frame=True,
            n_nodes_swc=5000,
            n_compartments_swc=4,
            swc_bbox_min_x=10.0, swc_bbox_min_y=20.0, swc_bbox_min_z=30.0,
            swc_bbox_max_x=100.0, swc_bbox_max_y=200.0, swc_bbox_max_z=300.0,
            soma_ccf_x=50.0, soma_ccf_y=60.0, soma_ccf_z=70.0,
            brain_region_flat="neocortex/frontal/primary motor",
            cell_type_flat="pyramidal",
            ingest_date="2026-04-21T00:00:00Z",
            ingest_git_sha="test-sha",
        ),
        NeuronRecord(
            neuron_id=2,
            neuron_name="test_b",
            archive="BICCN-MOp-miniatlas-anatomy",
            note="obtained from the Allen Institute for Brain Science, and it"
                 " is registered to the Common Coordinate Framework (CCF).",
            species="mouse",
            brain_region=["neocortex"],
            cell_type=[],
            source="Allen",
            in_ccf_frame=False,
            n_nodes_swc=2000,
            n_compartments_swc=3,
            swc_bbox_min_x=1.0, swc_bbox_min_y=2.0, swc_bbox_min_z=3.0,
            swc_bbox_max_x=10.0, swc_bbox_max_y=20.0, swc_bbox_max_z=30.0,
            soma_ccf_x=None, soma_ccf_y=None, soma_ccf_z=None,
            brain_region_flat="neocortex",
            cell_type_flat="",
            ingest_date="2026-04-21T00:00:00Z",
            ingest_git_sha="test-sha",
        ),
    ]


def test_schema_columns_present(tmp_path):
    out = tmp_path / "neurons_meta.parquet"
    write_sidecar(_sample_records(), out)
    table = pq.read_table(out)
    cols = set(table.column_names)
    # Required columns
    for c in [
        "neuron_id", "neuron_name", "archive", "note", "species",
        "brain_region", "cell_type",
        "surface_m", "volume_m",
        "source", "in_ccf_frame",
        "n_nodes_swc", "n_compartments_swc",
        "swc_bbox_min_x", "swc_bbox_max_x",
        "soma_ccf_x", "soma_ccf_y", "soma_ccf_z",
        "brain_region_flat", "cell_type_flat",
        "ingest_date", "ingest_git_sha",
    ]:
        assert c in cols, f"missing column: {c}"


def test_roundtrip(tmp_path):
    out = tmp_path / "neurons_meta.parquet"
    write_sidecar(_sample_records(), out)
    records = load_sidecar(out)
    assert set(records.keys()) == {1, 2}
    a = records[1]
    assert a.neuron_name == "test_a"
    assert a.source == "HUST"
    assert a.in_ccf_frame is True
    assert a.brain_region == ["neocortex", "frontal", "primary motor"]
    assert a.soma_ccf_x == pytest.approx(50.0)

    b = records[2]
    assert b.source == "Allen"
    assert b.in_ccf_frame is False
    assert b.soma_ccf_x is None
