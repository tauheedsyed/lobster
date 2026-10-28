"""Layer names of an AnnData object, without the ``.X`` alias.

anndata>=0.13 lists ``.X`` inside ``.layers`` under the key ``None``. That entry
is the main matrix, not a layer: it breaks ``', '.join`` over the keys and makes
every matrix look like it has one extra layer.
"""

from typing import Any, List


def layer_names(adata: Any) -> List[str]:
    """Return the names of the real layers of ``adata`` as strings.

    Skips the ``None`` key (the ``.X`` alias). Returns ``[]`` when ``adata`` has
    no ``layers`` mapping. Works for AnnData and MuData modalities.
    """
    layers = getattr(adata, "layers", None)
    if layers is None:
        return []
    return [str(key) for key in layers.keys() if key is not None]
