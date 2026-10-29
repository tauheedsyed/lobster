"""Refuse tool exports into the managed figure namespace (figure contract §1).

Native export tools (DE tables, annotation mappings, ML matrices) and modality saves
write to a path built from model-supplied input. Figures are saved only through the
trusted save step, so every such write is checked with the same writer rule the
filesystem tools use (:func:`lobster.core.figures.paths.assert_not_managed`) before
anything is created on disk. The check fails closed: an unexpected error while
classifying the target refuses the write.

Every refusal raised here carries the same fixed reason and no caller path, so a caller
that prints the error shows that text only. It carries no details (no path, no code),
and it is raised without a displayed chain.
"""

from pathlib import Path
from typing import Union

from lobster.core.figures.errors import ManagedPathError

#: The fixed reason carried by every export refusal (no path is echoed).
MANAGED_EXPORT_REASON = (
    "this export path is inside the managed figure folder (plots/<figure>/...). "
    "Exports cannot write there; choose a file name under exports/ instead."
)

#: The fixed user-facing refusal returned by export tools.
MANAGED_EXPORT_MESSAGE = "Error: " + MANAGED_EXPORT_REASON


def assert_export_allowed(
    workspace: Union[str, Path], target: Union[str, Path]
) -> None:
    """Raise :class:`ManagedPathError` unless a tool export may write ``target``.

    Pass ``target`` as the export spelled it (joined to its base directory, NOT
    resolved), so the lexical half of the classification can see a symlink or ``..``.
    Any failure other than the refusal itself (a classifier error, an unreadable
    ancestor) also raises :class:`ManagedPathError`: the write is refused, not allowed.
    Both carry :data:`MANAGED_EXPORT_REASON` and empty ``details``.
    """
    from lobster.core.figures.paths import assert_not_managed

    try:
        assert_not_managed(Path(workspace), Path(target))
    except Exception:  # nosec B110 # fail closed: every error is refused below
        pass
    else:
        return
    raise ManagedPathError(MANAGED_EXPORT_REASON) from None


def export_refused(workspace: Union[str, Path], *targets: Union[str, Path]) -> bool:
    """True when a tool export must not write ANY of ``targets`` (fail closed)."""
    for target in targets:
        try:
            assert_export_allowed(workspace, target)
        except ManagedPathError:
            return True
    return False
