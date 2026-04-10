"""3D vector tile generation and reading for MuDM.

Provides octree-based spatial indexing, 3D protobuf encoding,
and a full pipeline from MuDM features to 3D vector tiles.
"""

from .generator3d import TileGenerator3D
from .octree import OctreeConfig
from .parquet_prime import deprime_parquet, prime_parquet, repartition_parquet
from .parquet_reader import read_parquet
from .parquet_writer import generate_parquet
from .reader3d import TileReader3D
from .tilejson3d import TileModel3D

# Rust acceleration is always available (built via maturin)
RUST_AVAILABLE = True

try:
    from mudm_tools._rs import parse_obj  # noqa: F401
except ImportError:
    RUST_AVAILABLE = False

__all__ = [
    "TileGenerator3D",
    "OctreeConfig",
    "TileReader3D",
    "TileModel3D",
    "RUST_AVAILABLE",
    "generate_parquet",
    "read_parquet",
    "prime_parquet",
    "deprime_parquet",
    "repartition_parquet",
]
