"""mudm-tools — processing pipelines, tiling, and format converters for muDM."""

from .mudm2vt.mudm2vt import mudm2vt  # noqa: F401
from .neuroglancer import (  # noqa: F401
    to_neuroglancer,
    write_annotations,
)
from .gltf import to_gltf, to_glb, GltfConfig  # noqa: F401
from .arrow import to_arrow_table, to_geoparquet, ArrowConfig  # noqa: F401
from .arrow import from_arrow_table, from_geoparquet  # noqa: F401
from .tiling3d import (  # noqa: F401
    TileGenerator3D,
    OctreeConfig,
    TileReader3D,
    TileModel3D,
)
from .tiling3d.tilejson3d import (  # noqa: F401
    TileEncoding,
    KnownTileFormat,
    KnownCompression,
)

try:
    from ._rs import StreamingTileGenerator, StreamingTileGenerator2D  # noqa: F401

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

__version__ = "0.6.1"
