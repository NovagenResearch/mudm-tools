"""Pluggable featurizers for the AL harness. Each featurizer exposes
``load_features(feature_ids) -> dict[feature_id, np.ndarray]``.

Two implementations exist for Xenium:
  - XeniumParquetFeaturizer : reads muDM-tiled Parquet (the path the paper advocates)
  - XeniumReextractFeaturizer : re-reads raw 10x cell_feature_matrix on every call
                                (the baseline the paper compares against)
"""
from .xenium_parquet import XeniumParquetFeaturizer
from .xenium_reextract import XeniumReextractFeaturizer

__all__ = ["XeniumParquetFeaturizer", "XeniumReextractFeaturizer"]
