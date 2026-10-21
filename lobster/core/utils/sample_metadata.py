"""Exact sample identity and lossless obs metadata alignment."""

import re
from decimal import Decimal, InvalidOperation
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api import types as pdt

_WARNINGS = "metadata_writeback_warnings"


def _missing(value):
    return bool(pd.isna(value))


def _python(value):
    return value.item() if isinstance(value, np.generic) else value


def _boolean_values(values):
    nonnull = values.dropna()
    return len(nonnull) > 0 and all(
        isinstance(value, (bool, np.bool_)) for value in nonnull
    )


def _text_category(values):
    """String categories bypass H5AD's object-to-numeric coercion."""
    text = [np.nan if _missing(value) else str(_python(value)) for value in values]
    return pd.Series(pd.Categorical(text), index=values.index, name=values.name)


def _same_number(original, converted):
    if _missing(original):
        return _missing(converted)
    if _missing(converted):
        return False
    original, converted = _python(original), _python(converted)
    if isinstance(original, Integral):
        # Python's int/float equality does not round the integer first.
        return int(original) == converted
    return original == converted


def _infer_losslessly(values):
    """Infer only when every scalar survives; otherwise retain exact text."""
    values = values.astype(object)
    nonnull = values.dropna()
    if _boolean_values(values):
        return values.astype("boolean" if values.isna().any() else bool), False
    if len(nonnull) and all(isinstance(_python(value), Real) for value in nonnull):
        candidate = values.infer_objects()
        if pdt.is_numeric_dtype(candidate.dtype) and all(
            _same_number(original, converted)
            for original, converted in zip(values, candidate)
        ):
            return candidate, False
        return _text_category(values), True
    if len(nonnull) and all(isinstance(value, str) for value in nonnull):
        return _text_category(values), False
    return _text_category(values), bool(len(nonnull))


def _numeric_text(values):
    if values.empty:
        return False
    try:
        return all(Decimal(value).is_finite() for value in values)
    except InvalidOperation:
        return False


def _mixed_numeric_text(values):
    numeric = 0
    for value in values:
        try:
            Decimal(value)
            numeric += 1
        except InvalidOperation:
            continue  # A legitimate text label, not a numeric token.
    return 0 < numeric < len(values)


def _parse_numeric_tokens(text, typed):
    """Parse raw tokens, not pandas-rounded scalars; return None on lossy input.

    Annotation NA tokens use pandas' normal missing-value policy. Nonfinite
    numbers are retained as text with warnings. Primary IDs bypass this policy.
    """
    values = []
    for token, inferred in zip(text, typed):
        if _missing(inferred):
            values.append(np.nan)
            continue
        try:
            original = Decimal(token)
            if not original.is_finite():
                return None
            if re.fullmatch(r"[+-]?\d+", token.strip()):
                digits = token.strip().lstrip("+-")
                if len(digits) > 1 and digits.startswith("0"):
                    return None  # Preserve identifier-like integer spelling.
                if original.is_zero() and original.is_signed():
                    value = -0.0
                else:
                    value = int(original)
            else:
                value = float(token)
                if not np.isfinite(value) or original != Decimal(repr(value)):
                    return None
            values.append(value)
        except (InvalidOperation, ValueError, TypeError, OverflowError):
            return None
    # Explicit object construction prevents mixed/missing inputs from rounding
    # integers before the exact-value inference check can see them.
    parsed, retained = _infer_losslessly(
        pd.Series(values, index=text.index, dtype=object)
    )
    return None if retained else parsed


def read_sample_metadata(path, obs_names, sample_id_column="auto"):
    """Preserve ID text and validate annotation precision before parser inference.

    No case folding, trimming, or numeric ID normalization is performed. Numeric
    annotations that cannot retain their exact token values are stored as string
    categories, with warnings, rather than accepting rounded inferred floats.
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
    metadata[sample_id_column] = ids
    warnings = []
    for column in metadata.columns:
        if column == sample_id_column or pdt.is_bool_dtype(metadata[column].dtype):
            continue
        numeric = pdt.is_numeric_dtype(metadata[column].dtype) or _numeric_text(
            text[column][metadata[column].notna()]
        )
        nonmissing_text = text[column][metadata[column].notna()]
        if not numeric and _mixed_numeric_text(nonmissing_text):
            metadata[column] = _text_category(
                text[column].where(metadata[column].notna(), np.nan)
            )
            warnings.append(
                f"Column '{column}': mixed numeric/non-numeric annotation tokens retained as categorical text"
            )
        if numeric:
            parsed = _parse_numeric_tokens(text[column], metadata[column])
            if parsed is not None:
                metadata[column] = parsed
            else:
                raw = text[column].where(metadata[column].notna(), np.nan)
                metadata[column] = _text_category(raw)
                warnings.append(
                    f"Column '{column}': retained original numeric tokens as text to prevent precision/identity loss"
                )
    metadata = metadata.set_index(sample_id_column)
    metadata.attrs[_WARNINGS] = warnings
    return metadata, sample_id_column


def _aligned_column(existing, incoming):
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
        incoming = incoming.astype("boolean" if incoming.isna().any() else bool)
        values.loc[incoming.index] = incoming
        return values, False
    if isinstance(values.dtype, pd.CategoricalDtype):
        # Never let category union coerce integer categories to floats. Numeric
        # categories already require text in H5AD, so encode each scalar exactly
        # before expansion, preserving unmatched/invalid values and their spelling.
        numeric_categories = pdt.is_numeric_dtype(values.cat.categories.dtype) or all(
            isinstance(_python(value), Real) for value in values.cat.categories
        )
        if numeric_categories:
            objects = values.astype(object)
            objects.loc[incoming.index] = incoming.astype(object)
            return _text_category(objects), True
        additions = [
            value
            for value in incoming.dropna().unique()
            if value not in values.cat.categories
        ]
        if additions:
            values = values.cat.add_categories(additions)
        # Incoming categoricals may have a different vocabulary; assign scalars
        # after expanding the destination rather than categorical codes.
        values.loc[incoming.index] = incoming.astype(object)
        return values, False
    objects = values.astype(object)
    objects.loc[incoming.index] = incoming.astype(object)
    return _infer_losslessly(objects)


def merge_obs_metadata(
    obs, metadata, *, overwrite_existing=False, preserve_missing=False
):
    """Align on exact obs IDs; report any exact-text numeric retention in attrs.

    Boolean types/missingness are preserved. Object alignment precedes inference
    so reindexing cannot silently turn large integers into rounded floats.
    """
    if not obs.index.is_unique or not metadata.index.is_unique:
        raise ValueError("Sample IDs must be unique in both metadata and modality")
    result = obs.copy()
    warnings = list(metadata.attrs.get(_WARNINGS, []))
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
            values, retained = _aligned_column(obs[column], incoming)
        elif isinstance(incoming.dtype, pd.CategoricalDtype):
            values = incoming.reindex(obs.index)
            retained = False
        else:
            values, retained = _infer_losslessly(
                incoming.astype(object).reindex(obs.index)
            )
        result[column] = values
        if retained:
            warnings.append(
                f"Column '{column}': stored exact values as text to avoid lossy numeric/category coercion"
            )
        written.append(column)
    result.attrs[_WARNINGS] = list(dict.fromkeys(warnings))
    return result, written
