"""2D vector tile generation and reading for MuDM.

Provides quadtree-based spatial indexing and a full pipeline from
GeoJSON features to tiled Parquet output for ML training.
"""

from .parquet_prime import deprime_parquet, prime_parquet, repartition_parquet
from .parquet_reader import read_parquet
from .parquet_writer import generate_parquet
from .pbf_reader import read_pbf
from .pbf_writer import generate_pbf

try:
    from mudm_tools._rs import CartesianProjector2D, StreamingTileGenerator2D  # noqa: F401

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

__all__ = [
    "RUST_AVAILABLE",
    "generate_parquet",
    "read_parquet",
    "generate_pbf",
    "read_pbf",
    "prime_parquet",
    "deprime_parquet",
    "repartition_parquet",
]
