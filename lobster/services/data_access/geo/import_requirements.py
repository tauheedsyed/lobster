"""Opt-in GEO import requirements and numeric/identity candidate evidence."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Optional, Sequence

from lobster.services.data_access.geo.constants import GEOServiceError


@dataclass(frozen=True)
class ImportRequirement:
    representation: Literal["any", "counts"] = "any"
    level: Literal["any", "gene"] = "any"

    @classmethod
    def parse(cls, value: Optional[str]) -> ImportRequirement:
        """An empty requirement retains legacy imports; other values are explicit."""
        if value is None or value == "":
            return cls()
        if value == "gene_counts":
            return cls(representation="counts", level="gene")
        raise ValueError(
            f"Unknown data_requirement {value!r}. Use '' or 'gene_counts'."
        )

    @property
    def active(self) -> bool:
        return self.representation != "any"


@dataclass
class CandidateEvidence:
    file_name: str
    level: Literal["gene", "transcript", "unknown"]
    complete_parse: Optional[bool] = None
    sample_rule: Optional[str] = None
    whole_numbers: Optional[bool] = None
    equal_sample_totals: Optional[bool] = None
    verdict: Literal["accepted", "rejected", "not_checked"] = "not_checked"
    reason: str = ""


class SampleIdentityError(ValueError):
    """The table cannot be safely oriented by one exact sample identity rule."""


def orient_by_sample_identity(df, samples):
    """Return a samples × features copy and gsm_id or title_exact evidence.

    Every sample must match the full metadata sample set, in exactly one axis.
    title_exact compares the parsed label to metadata['title'] literally, just
    like BioVault's title_exact comparison to original_label: no surrounding
    whitespace stripping, case folding, string coercion or other normalization.
    Exact GSM accessions have authority; a title cannot reassign or mix them.
    No numeric cast is requested. Refuse if pandas' transpose changes values.
    """
    import pandas as pd

    if not isinstance(df, pd.DataFrame) or df.empty:
        raise SampleIdentityError("sample identity requires a nonempty matrix")
    if not isinstance(samples, Mapping) or not samples:
        raise SampleIdentityError("GSM sample metadata is missing")
    ids = set(samples)
    if any(
        not isinstance(g, str) or re.fullmatch(r"GSM[0-9]+", g) is None for g in ids
    ):
        raise SampleIdentityError("metadata sample IDs must be exact GSM accessions")

    titles = {}
    for gsm, metadata in samples.items():
        title = metadata.get("title") if isinstance(metadata, Mapping) else None
        if not isinstance(title, str) or not title.strip() or title in titles:
            titles = None
            break
        titles[title] = gsm

    choices = []
    for axis, transpose in ((df.columns, True), (df.index, False)):
        if len(axis) != len(ids):
            continue
        labels = list(axis)
        if (
            not all(isinstance(label, str) and label.strip() for label in labels)
            or not axis.is_unique
        ):
            continue
        if set(labels) == ids:
            mapping, rule = {label: label for label in labels}, "gsm_id"
        elif (
            titles is not None
            and not any(re.fullmatch(r"GSM[0-9]+", label) for label in labels)
            and set(labels) == set(titles)
        ):
            mapping, rule = {label: titles[label] for label in labels}, "title_exact"
        else:
            continue
        choices.append((transpose, mapping, rule))

    if len(choices) > 1:
        raise SampleIdentityError("ambiguous sample orientation: both axes match")
    if not choices:
        raise SampleIdentityError("no exact full-set one-to-one sample identity")
    transpose, mapping, rule = choices[0]
    table = (df.T if transpose else df).copy()
    if transpose and len(set(df.dtypes)) > 1:
        # Box only bounded comparison blocks, never the returned numeric table.
        # Python scalar equality catches integer precision loss from mixed types.
        step = max(1, (2**20) // df.shape[1])
        for start in range(0, len(df), step):
            expected = df.iloc[start : start + step].astype(object)
            actual = table.iloc[:, start : start + step].T.astype(object)
            if not expected.equals(actual):
                raise SampleIdentityError(
                    "sample orientation would change source values"
                )
    table.attrs["source_sample_id"] = mapping
    return table, rule


class CountValueFidelityError(ValueError):
    """Storage or adapter preprocessing would change accepted count values."""


def check_required_count_fidelity(table, staged=None):
    """Enforce the conservative float32 ceiling, then bounded exact equality."""
    import numpy as np
    from scipy.sparse import issparse

    from lobster.core.utils.count_matrix import is_whole_number_matrix

    values = table.to_numpy()
    if not is_whole_number_matrix(values):
        raise CountValueFidelityError("source is not a whole-number count matrix")
    largest = values.max().item()
    if largest > 2**24:
        raise CountValueFidelityError(
            f"largest value {largest} exceeds the exact float32 count ceiling {2**24}"
        )
    if staged is None:
        return
    if (
        staged.shape != table.shape
        or list(staged.obs_names) != list(table.index)
        or list(staged.var_names) != [str(label) for label in table.columns]
    ):
        raise CountValueFidelityError(
            "adapter preprocessing changed matrix shape or axis labels"
        )
    width = min(table.shape[1], 2**20)
    step = max(1, (2**20) // width)
    for start in range(0, len(table), step):
        for column in range(0, table.shape[1], width):
            actual = staged.X[start : start + step, column : column + width]
            if issparse(actual):
                actual = actual.toarray()
            if not np.array_equal(
                actual, values[start : start + step, column : column + width]
            ):
                raise CountValueFidelityError(
                    "adapter preprocessing changed accepted source values"
                )


class ImportRequirementNotMet(GEOServiceError):
    """Retain per-file evidence and let callers propagate the refusal unchanged."""

    def __init__(
        self,
        geo_id: str,
        requirement: ImportRequirement,
        candidates: Sequence[CandidateEvidence],
        *,
        problem: Literal[
            "no_candidate",
            "ambiguous",
            "identity",
            "too_many_candidates",
            "value_fidelity",
        ],
    ):
        self.geo_id = geo_id
        self.requirement = requirement
        self.candidates = list(candidates)
        self.problem = problem
        super().__init__(self.to_message())

    def to_message(self) -> str:
        lines = [
            f"Requirement not met for {self.geo_id} ({self.problem}): "
            f"needs {self.requirement.level}-level {self.requirement.representation}."
        ]
        lines.extend(
            f"- {candidate.file_name}: {candidate.verdict}, {candidate.reason}"
            for candidate in self.candidates
        )
        lines.append(
            "Next: choose another dataset, or report that this series has no "
            "file that meets the requirement. Do not retry this entry unchanged."
        )
        return "\n".join(lines)
