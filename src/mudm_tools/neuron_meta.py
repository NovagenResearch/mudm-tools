"""Per-neuron metadata sidecar for the cnn-nmo pipeline.

Writes a flat Parquet table keyed by ``neuron_id`` with all 43 NeuroMorpho
neuron fields, all 23 morphometry fields, and derived columns (source
institution, bbox, soma CCF coord, flat renderings of list columns,
build provenance).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, Field


class NeuronRecord(BaseModel):
    """One row of neurons_meta.parquet."""

    # Primary key
    neuron_id: int

    # --- 43 NeuroMorpho neuron fields ---
    neuron_name: str
    archive: Optional[str] = None
    age_scale: Optional[str] = None
    gender: Optional[str] = None
    age_classification: Optional[str] = None
    species: Optional[str] = None
    strain: Optional[str] = None
    scientific_name: Optional[str] = None
    stain: Optional[str] = None
    protocol: Optional[str] = None
    slicing_direction: Optional[str] = None
    reconstruction_software: Optional[str] = None
    objective_type: Optional[str] = None
    original_format: Optional[str] = None
    domain: Optional[str] = None
    attributes: Optional[str] = None
    magnification: Optional[str] = None
    upload_date: Optional[str] = None
    deposition_date: Optional[str] = None
    shrinkage_reported: Optional[str] = None
    shrinkage_corrected: Optional[str] = None
    slicing_thickness: Optional[str] = None
    min_age: Optional[str] = None
    max_age: Optional[str] = None
    min_weight: Optional[str] = None
    max_weight: Optional[str] = None
    png_url: Optional[str] = None
    physical_integrity: Optional[str] = None
    note: Optional[str] = None

    brain_region: list[str] = Field(default_factory=list)
    cell_type: list[str] = Field(default_factory=list)
    experiment_condition: list[str] = Field(default_factory=list)
    reference_pmid: list[str] = Field(default_factory=list)
    reference_doi: list[str] = Field(default_factory=list)

    # Kept as str because NeuroMorpho returns these as strings.
    reported_value: Optional[str] = None
    reported_xy: Optional[str] = None
    reported_z: Optional[str] = None
    corrected_value: Optional[str] = None
    corrected_xy: Optional[str] = None
    corrected_z: Optional[str] = None
    soma_surface: Optional[str] = None
    surface: Optional[str] = None
    volume: Optional[str] = None

    # --- 23 morphometry fields (floats) ---
    surface_m: Optional[float] = None
    volume_m: Optional[float] = None
    length: Optional[float] = None
    n_stems: Optional[float] = None
    n_bifs: Optional[float] = None
    n_branch: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None
    depth: Optional[float] = None
    diameter: Optional[float] = None
    eucDistance: Optional[float] = None
    pathDistance: Optional[float] = None
    branch_Order: Optional[float] = None
    contraction: Optional[float] = None
    fragmentation: Optional[float] = None
    partition_asymmetry: Optional[float] = None
    pk_classic: Optional[float] = None
    bif_ampl_local: Optional[float] = None
    bif_ampl_remote: Optional[float] = None
    fractal_Dim: Optional[float] = None
    n_nodes_m: Optional[float] = None
    soma_Surface_m: Optional[float] = None
    neuron_name_m: Optional[str] = None

    # --- Derived columns ---
    source: str                       # HUST / Allen / SEU-Allen / unknown
    in_ccf_frame: bool
    n_nodes_swc: int
    n_compartments_swc: int
    swc_bbox_min_x: float
    swc_bbox_min_y: float
    swc_bbox_min_z: float
    swc_bbox_max_x: float
    swc_bbox_max_y: float
    swc_bbox_max_z: float
    soma_ccf_x: Optional[float] = None
    soma_ccf_y: Optional[float] = None
    soma_ccf_z: Optional[float] = None
    brain_region_flat: str
    cell_type_flat: str
    ingest_date: str
    ingest_git_sha: str


_STRING_FIELDS = [
    "neuron_name", "archive", "age_scale", "gender",
    "age_classification", "species", "strain", "scientific_name",
    "stain", "protocol", "slicing_direction", "reconstruction_software",
    "objective_type", "original_format", "domain", "attributes",
    "magnification", "upload_date", "deposition_date", "shrinkage_reported",
    "shrinkage_corrected", "slicing_thickness", "min_age", "max_age",
    "min_weight", "max_weight", "png_url", "physical_integrity", "note",
    "reported_value", "reported_xy", "reported_z",
    "corrected_value", "corrected_xy", "corrected_z",
    "soma_surface", "surface", "volume",
    "neuron_name_m",
    "source", "brain_region_flat", "cell_type_flat",
    "ingest_date", "ingest_git_sha",
]

_LIST_FIELDS = [
    "brain_region", "cell_type", "experiment_condition",
    "reference_pmid", "reference_doi",
]

_FLOAT_FIELDS = [
    "surface_m", "volume_m", "length", "n_stems", "n_bifs", "n_branch",
    "width", "height", "depth", "diameter", "eucDistance", "pathDistance",
    "branch_Order", "contraction", "fragmentation",
    "partition_asymmetry", "pk_classic",
    "bif_ampl_local", "bif_ampl_remote", "fractal_Dim",
    "n_nodes_m", "soma_Surface_m",
    "swc_bbox_min_x", "swc_bbox_min_y", "swc_bbox_min_z",
    "swc_bbox_max_x", "swc_bbox_max_y", "swc_bbox_max_z",
    "soma_ccf_x", "soma_ccf_y", "soma_ccf_z",
]


def _schema() -> pa.Schema:
    fields = [
        pa.field("neuron_id", pa.uint32(), nullable=False),
        pa.field("in_ccf_frame", pa.bool_(), nullable=False),
        pa.field("n_nodes_swc", pa.uint32(), nullable=False),
        pa.field("n_compartments_swc", pa.uint8(), nullable=False),
    ]
    for name in _STRING_FIELDS:
        fields.append(pa.field(name, pa.string()))
    for name in _LIST_FIELDS:
        fields.append(pa.field(name, pa.list_(pa.string())))
    for name in _FLOAT_FIELDS:
        fields.append(pa.field(name, pa.float32()))
    return pa.schema(fields)


def write_sidecar(records: list[NeuronRecord], out_path: str | Path) -> Path:
    """Write NeuronRecords to a Parquet file. Overwrites if exists."""
    schema = _schema()
    cols: dict[str, list] = {f.name: [] for f in schema}
    for r in records:
        d = r.model_dump()
        for name in cols:
            cols[name].append(d.get(name))
    table = pa.Table.from_pydict(cols, schema=schema)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        table, out_path,
        compression="zstd",
        compression_level=3,
    )
    return out_path


def load_sidecar(path: str | Path) -> dict[int, NeuronRecord]:
    """Read a sidecar Parquet and return ``{neuron_id: NeuronRecord}``."""
    table = pq.read_table(str(path))
    out: dict[int, NeuronRecord] = {}
    for row in table.to_pylist():
        r = NeuronRecord.model_validate(row)
        out[r.neuron_id] = r
    return out
