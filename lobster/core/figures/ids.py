"""Opaque identifiers and FigureRef handling (contract §1).

IDs are opaque to every consumer: never parsed for meaning and never treated as ordinals.
The local authority draws ``<prefix>_`` + 10 lowercase base32 characters from ``secrets``.
The Cloud catalog writer may use any ID matching :data:`ID_PATTERN`.
"""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass

#: Shared opaque-ID grammar (engine, Cloud, FE).
ID_PATTERN = r"[a-z0-9_-]{1,64}"
_ID_RE = re.compile(rf"^{ID_PATTERN}$")

#: FigureRef literal scan with token boundaries (contract §6.4). A bounded convenience,
#: not a parser or a security control.
REF_SCAN_RE = re.compile(rf"(?<![a-z0-9_-])({ID_PATTERN})@({ID_PATTERN})(?![a-z0-9_-])")

_BASE32 = "abcdefghijklmnopqrstuvwxyz234567"

#: Local ID prefixes. Presentation-neutral; consumers must not depend on them.
PREFIX_FIGURE = "f"
PREFIX_REVISION = "r"
PREFIX_RENDITION = "p"
PREFIX_EXECUTION = "x"


def new_id(prefix: str) -> str:
    """Return a fresh opaque local ID, e.g. ``f_k3m9q2xa7d``."""
    if prefix not in {
        PREFIX_FIGURE,
        PREFIX_REVISION,
        PREFIX_RENDITION,
        PREFIX_EXECUTION,
    }:
        raise ValueError(f"unknown id prefix: {prefix!r}")
    return prefix + "_" + "".join(secrets.choice(_BASE32) for _ in range(10))


def is_valid_id(value: object) -> bool:
    # fullmatch: with .match, "$" also matched before a final "\n"
    return isinstance(value, str) and bool(_ID_RE.fullmatch(value))


@dataclass(frozen=True)
class FigureRef:
    """``<figure_id>@<revision_id>``; always resolved inside an authenticated scope."""

    figure_id: str
    revision_id: str

    def __post_init__(self) -> None:
        if not is_valid_id(self.figure_id) or not is_valid_id(self.revision_id):
            raise ValueError(
                f"invalid FigureRef parts: {self.figure_id!r}@{self.revision_id!r}"
            )

    def __str__(self) -> str:
        return f"{self.figure_id}@{self.revision_id}"

    @classmethod
    def parse(cls, value: "str | FigureRef") -> "FigureRef":
        if isinstance(value, FigureRef):
            return value
        if not isinstance(value, str) or value.count("@") != 1:
            raise ValueError(f"not a FigureRef: {value!r}")
        figure_id, revision_id = value.split("@")
        return cls(figure_id, revision_id)


def scan_refs(text: str) -> list[FigureRef]:
    """Return distinct FigureRef-shaped tokens in ``text`` (in order of first appearance)."""
    seen: dict[str, FigureRef] = {}
    for match in REF_SCAN_RE.finditer(text or ""):
        ref = FigureRef(match.group(1), match.group(2))
        seen.setdefault(str(ref), ref)
    return list(seen.values())
