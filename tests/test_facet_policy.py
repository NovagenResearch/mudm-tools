import numpy as np
from mudm_tools.facets import FacetPolicy, select_facet_keys

def test_selection_heuristic_and_overrides():
    pol = FacetPolicy.from_config({"keep_inline": ["cell_type"],
                                   "select": {"numeric_high_card": True, "card_threshold": 64,
                                              "include": ["m_*"], "exclude": ["area"]}})
    cols = {"cell_id": np.dtype("O"), "cell_type": np.dtype("O"),
            "m_CD8": np.dtype("float32"), "area": np.dtype("float32"),
            "n_genes": np.dtype("int32")}
    card = {"m_CD8": 5000, "area": 5000, "n_genes": 5000}
    facet, inline = select_facet_keys(pol, cols, card)
    assert "m_CD8" in facet and "n_genes" in facet      # high-card numeric / included
    assert "cell_id" in inline and "cell_type" in inline  # key + keep_inline
    assert "area" in inline                              # excluded override
