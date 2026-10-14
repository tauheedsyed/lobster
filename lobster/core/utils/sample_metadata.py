"""Exact sample identity and dtype-preserving obs metadata alignment."""

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api import types as pdt


def read_sample_metadata(path, obs_names, sample_id_column="auto"):
    """Read typed annotations while keeping identifier text before NA/type inference.

    No case folding, trimming, or numeric ID normalization is performed. Empty
    IDs and ambiguous auto-detection fail rather than inventing sample identities.
    """
    path = Path(path)
    reader = pd.read_excel if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv
    kwargs = {"sep": "\t"} if path.suffix.lower() == ".tsv" else {}
    text = reader(path, dtype=str, keep_default_na=False, na_filter=False, **kwargs)
    if sample_id_column == "auto":
        overlaps = {
            column: text[column].isin(obs_names).sum() for column in text.columns
        }
        best = max(overlaps.values(), default=0)
        candidates = [column for column, count in overlaps.items() if count == best]
        if not best:
            raise ValueError(
                "Could not auto-detect sample ID column by exact matching; specify sample_id_column"
            )
        if len(candidates) != 1:
            raise ValueError(
                "Ambiguous sample ID columns; specify sample_id_column explicitly"
            )
        sample_id_column = candidates[0]
    if sample_id_column not in text:
        raise ValueError(f"Sample ID column '{sample_id_column}' not found")
    ids = text[sample_id_column]
    if (
        ids.isna().any()
        or ids.map(lambda value: not isinstance(value, str) or not value.strip()).any()
    ):
        raise ValueError("Sample IDs must be nonempty text")
    if not ids.is_unique or not obs_names.is_unique:
        raise ValueError("Sample IDs must be unique in both metadata and modality")
    metadata = reader(path, **kwargs)
    metadata[sample_id_column] = (
        ids  # Restore exact text, including literal NA-like IDs.
    )
    return metadata.set_index(sample_id_column), sample_id_column


def _boolean_values(values):
    nonnull = values.dropna()
    return len(nonnull) > 0 and all(
        isinstance(value, (bool, np.bool_)) for value in nonnull
    )


def _aligned_column(existing, incoming):
    """Assign compatible values without moving bool/numeric columns into object."""
    values = existing.copy()
    if pdt.is_bool_dtype(values.dtype) or _boolean_values(values):
        if len(incoming.dropna()) and not _boolean_values(incoming):
            raise ValueError(
                "Boolean annotations require boolean values, not strings or numbers"
            )
        if isinstance(values.dtype, pd.CategoricalDtype) or not pdt.is_bool_dtype(
            values.dtype
        ):
            values = values.astype(
                "boolean" if values.isna().any() or incoming.isna().any() else bool
            )
        elif incoming.isna().any():
            values = values.astype("boolean")
    elif isinstance(values.dtype, pd.CategoricalDtype):
        additions = incoming.dropna().unique()
        additions = [value for value in additions if value not in values.cat.categories]
        if additions:
            values = values.cat.add_categories(additions)
    elif pdt.is_numeric_dtype(values.dtype):
        if len(incoming.dropna()) and not pdt.is_numeric_dtype(incoming.dtype):
            incoming = incoming.infer_objects()
            if not pdt.is_numeric_dtype(incoming.dtype):
                raise ValueError("Numeric annotations require numeric values")
        common = pd.concat([values, incoming]).dtype
        if not pdt.is_numeric_dtype(common):
            raise ValueError("Incompatible numeric annotation dtype")
        if pdt.is_float_dtype(common):
            for integers in (values, incoming):
                if pdt.is_integer_dtype(integers.dtype):
                    widened = integers.astype(common)
                    for original, converted in zip(integers.tolist(), widened.tolist()):
                        if not pd.isna(original) and int(original) != float(converted):
                            raise ValueError(
                                "Numeric updates would lose integer precision"
                            )
        if common != values.dtype:
            values = values.astype(common)
    values.loc[incoming.index] = incoming
    return values


def merge_obs_metadata(
    obs, metadata, *, overwrite_existing=False, preserve_missing=False
):
    """Return updated obs and written columns, preserving unmatched rows and types.

    ``preserve_missing`` additionally excludes missing updates (standardization
    does not replace values absent from successfully validated schema rows).
    New boolean annotations use pandas nullable boolean when unmatched rows exist.
    """
    if not obs.index.is_unique or not metadata.index.is_unique:
        raise ValueError("Sample IDs must be unique in both metadata and modality")
    result = obs.copy()
    matched = obs.index.intersection(metadata.index, sort=False)
    written = []
    for column in metadata.columns:
        if column in obs and not overwrite_existing:
            continue
        incoming = metadata.loc[matched, column]
        if preserve_missing:
            incoming = incoming.dropna()
        if incoming.empty:
            continue
        if column in obs:
            result[column] = _aligned_column(obs[column], incoming)
        else:
            values = incoming.reindex(obs.index)
            if _boolean_values(values):
                values = values.astype("boolean" if values.isna().any() else bool)
            else:
                values = values.infer_objects()
            result[column] = values
        written.append(column)
    return result, written
