"""Managed figure lifecycle: immutable revisions, trusted save, local store.

Contract: the managed figure contract (see module docs).

Public surface (stable for this release line):

- :mod:`lobster.core.figures.ids` - opaque identifiers and FigureRef parsing
- :mod:`lobster.core.figures.payload` - byte-exact figure/config handling and fingerprints
- :mod:`lobster.core.figures.records` - candidate/execution/commit record types and the attempt digest
- :mod:`lobster.core.figures.paths` - managed-path classifier and local layout
- :mod:`lobster.core.figures.store` - ``FigureStore`` protocol and ``LocalFigureStore``
- :mod:`lobster.core.figures.saver` - ``TrustedSaver`` (fence + execution-before-revision ordering)

Importing this package must stay cheap: no plotly/numpy/pandas at module import time.
"""

from lobster.core.figures.errors import (
    AttemptConflictError,
    FigureError,
    FigureInvalidError,
    FigureNotFoundError,
    ManagedPathError,
)
from lobster.core.figures.ids import FigureRef, is_valid_id, new_id
from lobster.core.figures.paths import is_managed_figure_path

__all__ = [
    "AttemptConflictError",
    "FigureError",
    "FigureInvalidError",
    "FigureNotFoundError",
    "FigureRef",
    "ManagedPathError",
    "is_managed_figure_path",
    "is_valid_id",
    "new_id",
]
