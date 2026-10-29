"""Refuse directory creation that would pass through the managed figure folder.

``Path.mkdir(parents=True)`` and ``os.makedirs`` create every lexical prefix of the
path they are given, while :func:`~lobster.core.figures.export_guard.assert_export_allowed`
classifies one (normalized) target. A spelling such as
``exports/../plots/<figure>/<new>/../../x`` is classified as ``x``, yet creating it
recursively creates ``plots/<figure>/<new>`` on the way. A native writer that creates a
directory recursively calls :func:`assert_dirs_allowed` on that exact directory first.

A directory is checked as the place files will be written, not as a file: the guard's
legacy allowance for a flat file directly under ``plots/`` does not let a writer create
a folder there (a new ``plots/<name>/`` is figure-shaped).
"""

import os
from pathlib import Path
from typing import Union

from lobster.core.figures.errors import ManagedPathError
from lobster.core.figures.export_guard import (
    MANAGED_EXPORT_REASON,
    assert_export_allowed,
)

# A name only classified (never created) to ask "may a file be written in here?".
_CHILD = ".export-dir-check"


def _assert_container(workspace: Union[str, Path], directory: Path) -> None:
    probe = directory / _CHILD
    try:
        os.lstat(probe)
    except (FileNotFoundError, NotADirectoryError):
        pass  # absent: its classification resolves the folder itself
    except Exception:  # cannot tell whether it exists: refuse (fail closed)
        raise ManagedPathError(MANAGED_EXPORT_REASON) from None
    else:
        # An existing entry under this name (a link above all) would redirect the
        # resolved half of the check away from the folder: refuse.
        raise ManagedPathError(MANAGED_EXPORT_REASON) from None
    assert_export_allowed(workspace, probe)


def assert_dirs_allowed(
    workspace: Union[str, Path], directory: Union[str, Path]
) -> None:
    """Raise :class:`ManagedPathError` unless ``directory`` may be created recursively.

    Checks, with :func:`assert_export_allowed` (the same fixed refusal):

    * the final ``directory``, existing or not, as the folder files are written in
      (a name inside it that is only classified; if that name exists, the folder is
      refused, since a link there could point the check elsewhere);
    * every lexical prefix of its UNnormalized absolute spelling that does not exist
      yet (``..`` is not collapsed before the walk), both as a new entry and as a
      folder files are written in.

    Prefixes that already exist are not created; one that is a link is covered by the
    resolved half of the final-folder check, which resolves from the original spelling.
    """
    spelled = Path(directory)
    absolute = spelled if spelled.is_absolute() else Path.cwd() / spelled
    _assert_container(workspace, absolute)
    parts = absolute.parts
    for end in range(1, len(parts) + 1):
        prefix = Path(*parts[:end])
        if not os.path.exists(prefix):
            assert_export_allowed(workspace, prefix)
            _assert_container(workspace, prefix)
