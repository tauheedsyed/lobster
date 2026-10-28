"""Bounded reads of files that another process left behind.

The code-execution child writes its result, its completion marker and its figure
submissions as files, and the parent reads them after the child has exited. The child
controls what is at those paths, so every such read here:

- refuses a final symlink, FIFO, socket or device instead of reading it, and never waits
  on one: the entry is checked with ``lstat`` first, then opened with ``O_NOFOLLOW`` and
  ``O_NONBLOCK`` and checked again with ``fstat`` (same file, still regular);
- never reads more than ``limit`` bytes (one byte more than the limit is read to detect an
  over-size file before it is used).

This bounds the parent's own reads after the child's timeout has ended. It is not a promise
about I/O on a network or device filesystem that stalls. Custom code runs as your own user
and is not sandboxed: another process of that user can still disrupt the session.
"""

from __future__ import annotations

import os
import stat
from typing import Optional, Tuple

NOT_REGULAR = "not_regular"
TOO_LARGE = "too_large"
UNREADABLE = "unreadable"

_CHUNK = 1 << 20


class RefusedRead(Exception):
    """A present entry that is not read. ``reason`` is one stable code (never a path)."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _open_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )


def regular_identity(path: "str | os.PathLike") -> Optional[Tuple[int, int]]:
    """``(st_dev, st_ino)`` of the regular file at ``path`` (``lstat``; nothing is opened).

    None if absent. A symlink (dangling or not), FIFO, socket, device or directory raises
    :class:`RefusedRead`.
    """
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return None
    except OSError:
        raise RefusedRead(UNREADABLE) from None
    if not stat.S_ISREG(st.st_mode):
        raise RefusedRead(NOT_REGULAR)
    return st.st_dev, st.st_ino


def read_regular(path: "str | os.PathLike", limit: int) -> Optional[bytes]:
    """The bytes of the regular file at ``path``, at most ``limit`` of them.

    None if absent. Anything else that is not a regular file of at most ``limit`` bytes
    raises :class:`RefusedRead`; the descriptor is closed on every path.
    """
    identity = regular_identity(path)
    if identity is None:
        return None
    try:
        fd = os.open(path, _open_flags())
    except FileNotFoundError:
        return None
    except OSError:  # ELOOP: it became a symlink after the check
        raise RefusedRead(UNREADABLE) from None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or (st.st_dev, st.st_ino) != identity:
            raise RefusedRead(NOT_REGULAR)
        if st.st_size > limit:
            raise RefusedRead(TOO_LARGE)
        chunks = []
        total = 0
        while total <= limit:
            chunk = os.read(fd, min(_CHUNK, limit + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        if total > limit:  # it grew after the size check
            raise RefusedRead(TOO_LARGE)
        return b"".join(chunks)
    except OSError:
        raise RefusedRead(UNREADABLE) from None
    finally:
        os.close(fd)
