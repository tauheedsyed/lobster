"""Compare a replay's outputs with the reference, under a ComparisonPolicy.

The comparator walks two trees of values (the reference captured from the live
session, and what the replayed notebook produced) and returns one
:class:`~lobster.testing.replay_contract.SlotResult` per required path.

Supported values: ``AnnData`` (slots ``X``, ``layers``, ``obs``, ``var``,
``obsm``, ``obsp``, ``varm``, ``varp``, ``uns``, ``raw`` with ``X``/``var``/
``varm``), pandas ``DataFrame``/``Series``, NumPy arrays and scalars, SciPy
sparse matrices, mappings, lists and tuples, Python scalars, and output files
(:func:`output_files`; CSV/TSV, JSON and H5AD are parsed, anything else is
compared byte for byte). Anything else is reported as NOT_COMPARED and never
converted to text to make it comparable.

Rules that keep a wrong result from passing:

* inventories are exact: keys, columns and files present on one side only are
  a mismatch, and axis order matters (nothing is sorted before comparing);
* shapes, dtypes, integers, labels, categories and missing values compare
  exactly; floats use the policy's predeclared tolerance (exact by default),
  NaN must sit at the same positions and infinities must be identical;
* a required path missing from the reference is NOT_COMPARED, and missing from
  the output is a MISMATCH;
* an output file that is a symbolic link is not followed.

The inputs must already be in comparable form. If the reference comes from
memory and the output from disk, load both through the same codec first; the
comparator treats a list and an array, or a float32 and a float64 array, as
different.

Scientific packages are imported lazily, so importing this module is cheap.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from lobster.testing.replay_contract import (
    ComparisonPolicy,
    SlotResult,
    SlotStatus,
    SparsePolicy,
    split_path,
)

MAX_FINDINGS = 20

_MISSING = object()


@dataclass(frozen=True)
class OutputFile:
    """A file a replay wrote (or the reference copy of it)."""

    path: Path


def output_files(directory: Path) -> Dict[str, Any]:
    """Nested mapping of every file below ``directory`` to an OutputFile.

    Directories become mappings, so ``files/split/train.csv`` addresses
    ``<directory>/split/train.csv``. Symbolic links are kept as entries (and
    later reported as not compared) rather than followed.
    """
    directory = Path(directory)
    tree: Dict[str, Any] = {}
    for entry in sorted(directory.iterdir()):
        if entry.is_dir() and not entry.is_symlink():
            tree[entry.name] = output_files(entry)
        else:
            tree[entry.name] = OutputFile(entry)
    return tree


def compare(
    expected: Any, actual: Any, policy: ComparisonPolicy
) -> Tuple[SlotResult, ...]:
    """Compare ``actual`` with ``expected`` for every path the policy requires."""
    if not isinstance(policy, ComparisonPolicy):
        raise TypeError("policy must be a ComparisonPolicy")
    results = []
    for path in policy.required:
        rtol, atol = policy.tolerance_for(path)
        reference = _resolve(expected, path)
        output = _resolve(actual, path)
        if reference is _MISSING:
            results.append(
                SlotResult(
                    path,
                    SlotStatus.NOT_COMPARED,
                    rtol=rtol,
                    atol=atol,
                    not_compared=(f"{path}: the reference has no value here",),
                )
            )
            continue
        if output is _MISSING:
            results.append(
                SlotResult(
                    path,
                    SlotStatus.MISMATCH,
                    rtol=rtol,
                    atol=atol,
                    diffs=(f"{path}: missing from the output",),
                )
            )
            continue
        walk = _Walk(policy)
        walk.value(reference, output, path)
        results.append(walk.result(path, rtol, atol))
    return tuple(results)


# --------------------------------------------------------------------------
# Tree navigation
# --------------------------------------------------------------------------


def _module_type(module: str, name: str):
    loaded = sys.modules.get(module)
    return getattr(loaded, name, None) if loaded is not None else None


def _is(value: Any, module: str, name: str) -> bool:
    cls = _module_type(module, name)
    return cls is not None and isinstance(value, cls)


def _issparse(value: Any) -> bool:
    sparse = sys.modules.get("scipy.sparse")
    return sparse is not None and sparse.issparse(value)


def _anndata_tree(adata: Any) -> Dict[str, Any]:
    raw = adata.raw
    return {
        "X": adata.X,
        # Recent anndata lists a ``None`` layer that aliases X; X is compared
        # once, under "X", and an alias is not separate evidence.
        "layers": {k: adata.layers[k] for k in adata.layers.keys() if k is not None},
        "obs": adata.obs,
        "var": adata.var,
        "obsm": dict(adata.obsm),
        "obsp": dict(adata.obsp),
        "varm": dict(adata.varm),
        "varp": dict(adata.varp),
        "uns": dict(adata.uns),
        "raw": (
            None
            if raw is None
            else {"X": raw.X, "var": raw.var, "varm": dict(raw.varm)}
        ),
    }


def _child(node: Any, key: str) -> Any:
    if _is(node, "anndata", "AnnData"):
        return _anndata_tree(node).get(key, _MISSING)
    if isinstance(node, Mapping):
        return node.get(key, _MISSING) if key in node else _MISSING
    if _is(node, "pandas", "DataFrame"):
        columns = list(node.columns)
        if columns.count(key) == 1:
            return node[key]
    return _MISSING


def _resolve(tree: Any, path: str) -> Any:
    node = tree
    for part in split_path(path):
        node = _child(node, part)
        if node is _MISSING:
            return _MISSING
    return node


def _type_name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _kind(value: Any) -> str:
    np = sys.modules.get("numpy")
    if value is None:
        return "none"
    if isinstance(value, OutputFile):
        return "file"
    if _is(value, "anndata", "AnnData"):
        return "anndata"
    if _is(value, "pandas", "DataFrame"):
        return "dataframe"
    if _is(value, "pandas", "Series"):
        return "series"
    if _issparse(value):
        return "sparse"
    if np is not None and isinstance(value, np.ndarray):
        return "array"
    if isinstance(value, bool) or (np is not None and isinstance(value, np.bool_)):
        return "bool"
    if isinstance(value, int) or (np is not None and isinstance(value, np.integer)):
        return "int"
    if isinstance(value, float) or (np is not None and isinstance(value, np.floating)):
        return "float"
    if isinstance(value, complex) or (
        np is not None and isinstance(value, np.complexfloating)
    ):
        return "complex"
    if isinstance(value, str):
        return "str"
    if isinstance(value, bytes):
        return "bytes"
    if isinstance(value, Mapping):
        return "mapping"
    if type(value) in (list, tuple):
        return "sequence"
    pandas = sys.modules.get("pandas")
    if pandas is not None and (value is pandas.NA or value is pandas.NaT):
        return "missing"
    if isinstance(value, (_dt.datetime, _dt.date, _dt.timedelta)):
        return "datetime"
    if np is not None and isinstance(value, (np.datetime64, np.timedelta64)):
        return "datetime"
    return "unknown"


# --------------------------------------------------------------------------
# Comparison walk
# --------------------------------------------------------------------------


class _Walk:
    def __init__(self, policy: ComparisonPolicy) -> None:
        self.policy = policy
        self.compared = 0
        self.diffs: List[str] = []
        self.diff_count = 0
        self.not_compared: List[str] = []

    # -- findings ---------------------------------------------------------

    def diff(self, path: str, message: str) -> None:
        self.diff_count += 1
        if len(self.diffs) < MAX_FINDINGS:
            self.diffs.append(f"{path}: {message}")

    def skip(self, path: str, message: str) -> None:
        if len(self.not_compared) < MAX_FINDINGS:
            self.not_compared.append(f"{path}: {message}")

    def result(self, path: str, rtol: float, atol: float) -> SlotResult:
        diffs = list(self.diffs)
        if self.diff_count > len(diffs):
            diffs.append(f"... and {self.diff_count - len(diffs)} more differences")
        if diffs:
            status = SlotStatus.MISMATCH
        elif self.not_compared:
            status = SlotStatus.NOT_COMPARED
        else:
            status = SlotStatus.MATCH
        return SlotResult(
            path,
            status,
            compared=self.compared,
            rtol=rtol,
            atol=atol,
            diffs=tuple(diffs),
            not_compared=tuple(self.not_compared),
        )

    def excluded(self, path: str) -> bool:
        return self.policy.exclusion_for(path) is not None

    # -- dispatch ---------------------------------------------------------

    def value(self, expected: Any, actual: Any, path: str) -> None:
        kind = _kind(expected)
        if kind == "unknown":
            self.skip(path, f"unsupported type {_type_name(expected)}")
            return
        other = _kind(actual)
        if other != kind:
            self.diff(
                path,
                f"type differs: expected {_type_name(expected)}, "
                f"got {_type_name(actual)}",
            )
            return
        getattr(self, f"_{kind}")(expected, actual, path)

    # -- scalars ----------------------------------------------------------

    def _none(self, expected, actual, path):
        self.compared += 1

    def _missing(self, expected, actual, path):
        self.compared += 1
        if expected is not actual:
            self.diff(path, f"expected {expected!r}, got {actual!r}")

    def _exact(self, expected, actual, path):
        self.compared += 1
        if expected != actual:
            self.diff(path, f"expected {expected!r}, got {actual!r}")

    _bool = _int = _str = _bytes = _exact

    def _datetime(self, expected, actual, path):
        self.compared += 1
        if type(expected) is not type(actual):
            self.diff(path, f"type differs: {_type_name(expected)}")
            return
        np = sys.modules.get("numpy")
        if np is not None and isinstance(expected, (np.datetime64, np.timedelta64)):
            if np.isnat(expected) or np.isnat(actual):
                if not (np.isnat(expected) and np.isnat(actual)):
                    self.diff(path, f"expected {expected!r}, got {actual!r}")
                return
        if expected != actual:
            self.diff(path, f"expected {expected!r}, got {actual!r}")

    def _float(self, expected, actual, path):
        self._scalar_number(expected, actual, path)

    def _complex(self, expected, actual, path):
        self._scalar_number(expected, actual, path)

    def _scalar_number(self, expected, actual, path):
        self.compared += 1
        np = sys.modules.get("numpy")
        if np is not None and isinstance(expected, np.generic):
            if isinstance(actual, np.generic) and expected.dtype != actual.dtype:
                self.diff(path, f"dtype differs: {expected.dtype} vs {actual.dtype}")
                return
        rtol, atol = self.policy.tolerance_for(path)
        is_complex = isinstance(expected, complex) or (
            np is not None and isinstance(expected, np.complexfloating)
        )
        for part in ("real", "imag") if is_complex else (None,):
            e = getattr(expected, part) if part else expected
            a = getattr(actual, part) if part else actual
            e, a = float(e), float(a)
            if math.isnan(e) or math.isnan(a):
                ok = math.isnan(e) and math.isnan(a)
            elif math.isinf(e) or math.isinf(a):
                ok = e == a
            else:
                ok = abs(a - e) <= atol + rtol * abs(e)
            if not ok:
                self.diff(
                    path,
                    f"expected {expected!r}, got {actual!r} (rtol={rtol}, atol={atol})",
                )
                return

    # -- containers -------------------------------------------------------

    def _mapping(self, expected, actual, path):
        keys = []
        for side in (expected, actual):
            bad = [k for k in side if not isinstance(k, str) or "/" in k or k == ""]
            if bad:
                self.skip(path, f"keys that cannot be addressed: {bad[:5]!r}")
                return
            keys.append({k for k in side if not self.excluded(f"{path}/{k}")})
        want, got = keys
        for key in sorted(want - got):
            self.diff(f"{path}/{key}", "missing from the output")
        for key in sorted(got - want):
            self.diff(f"{path}/{key}", "not in the reference")
        for key in sorted(want & got):
            self.value(expected[key], actual[key], f"{path}/{key}")

    def _sequence(self, expected, actual, path):
        if type(expected) is not type(actual):
            self.diff(path, f"type differs: {_type_name(expected)}")
            return
        if len(expected) != len(actual):
            self.diff(path, f"length differs: {len(expected)} vs {len(actual)}")
            return
        for index, (e, a) in enumerate(zip(expected, actual)):
            self.value(e, a, f"{path}[{index}]")

    def _anndata(self, expected, actual, path):
        self._mapping(_anndata_tree(expected), _anndata_tree(actual), path)

    # -- arrays -----------------------------------------------------------

    def _array(self, expected, actual, path):
        if type(expected) is not type(actual):
            self.diff(path, f"type differs: {_type_name(expected)}")
            return
        if expected.shape != actual.shape:
            self.diff(path, f"shape differs: {expected.shape} vs {actual.shape}")
            return
        if expected.dtype != actual.dtype:
            self.diff(path, f"dtype differs: {expected.dtype} vs {actual.dtype}")
            return
        self._array_values(expected, actual, path)

    def _array_values(self, expected, actual, path):
        """Compare two arrays of equal shape and dtype."""
        np = sys.modules["numpy"]
        kind = expected.dtype.kind
        if kind in "biuUS":
            self.compared += expected.size
            unequal = np.asarray(expected != actual)
            if unequal.any():
                self.diff(
                    path,
                    f"{int(unequal.sum())} of {expected.size} values differ "
                    f"(first at index {_first(unequal)})",
                )
        elif kind in "fc":
            self.compared += expected.size
            self._float_array(np, expected, actual, path)
        elif kind in "mM":
            self.compared += expected.size
            nat_e, nat_a = np.isnat(expected), np.isnat(actual)
            if (nat_e != nat_a).any():
                self.diff(path, "missing (NaT) positions differ")
            elif (expected[~nat_e] != actual[~nat_a]).any():
                self.diff(path, "datetime values differ")
        elif kind == "O":
            flat_e, flat_a = expected.ravel(), actual.ravel()
            for index in range(flat_e.size):
                self.value(flat_e[index], flat_a[index], f"{path}[{index}]")
        else:
            self.skip(path, f"unsupported dtype {expected.dtype}")

    def _float_array(self, np, expected, actual, path):
        rtol, atol = self.policy.tolerance_for(path)
        nan_e, nan_a = np.isnan(expected), np.isnan(actual)
        if (nan_e != nan_a).any():
            self.diff(path, f"NaN positions differ (first at {_first(nan_e != nan_a)})")
            return
        inf_e, inf_a = np.isinf(expected), np.isinf(actual)
        if (inf_e != inf_a).any() or (expected[inf_e] != actual[inf_a]).any():
            self.diff(path, "infinite values differ")
            return
        finite = ~(nan_e | inf_e)
        e, a = expected[finite], actual[finite]
        with np.errstate(all="ignore"):
            error = np.abs(a - e)
            allowed = atol + rtol * np.abs(e)
            bad = ~(error <= allowed)
        if bad.any():
            worst = float(np.max(error[bad]))
            self.diff(
                path,
                f"{int(bad.sum())} of {e.size} values differ beyond "
                f"rtol={rtol}, atol={atol} (largest difference {worst:.6g})",
            )

    def _sparse(self, expected, actual, path):
        if expected.shape != actual.shape:
            self.diff(path, f"shape differs: {expected.shape} vs {actual.shape}")
            return
        if expected.dtype != actual.dtype:
            self.diff(path, f"dtype differs: {expected.dtype} vs {actual.dtype}")
            return
        if self.policy.sparse is SparsePolicy.REPRESENTATION:
            if type(expected) is not type(actual):
                self.diff(path, f"sparse type differs: {_type_name(expected)}")
                return
            fields = {"csr": "indptr indices", "csc": "indptr indices"}.get(
                expected.format, "row col" if expected.format == "coo" else None
            )
            if fields is None:
                self.skip(path, f"sparse format {expected.format} not supported")
                return
            for name in fields.split():
                self._structure(getattr(expected, name), getattr(actual, name), path)
            e_data, a_data = expected.data, actual.data
        else:
            e_canon, a_canon = _canonical_csr(expected), _canonical_csr(actual)
            for name in ("indptr", "indices"):
                if not self._structure(
                    getattr(e_canon, name), getattr(a_canon, name), path
                ):
                    return
            e_data, a_data = e_canon.data, a_canon.data
        if e_data.shape != a_data.shape:
            self.diff(path, "number of stored values differs")
            return
        before = self.compared
        self._array_values(e_data, a_data, path)
        # Every cell of the matrix was compared, stored or implicit zero.
        self.compared = before + int(expected.shape[0]) * int(expected.shape[1])

    def _structure(self, expected, actual, path) -> bool:
        np = sys.modules["numpy"]
        if expected.shape != actual.shape or not np.array_equal(expected, actual):
            self.diff(path, "positions of stored values differ")
            return False
        return True

    # -- pandas -----------------------------------------------------------

    def _dataframe(self, expected, actual, path):
        self._index(expected.index, actual.index, f"{path} index")
        for side in (expected, actual):
            if side.columns.has_duplicates:
                self.skip(path, "duplicate column labels")
                return
        want = [c for c in expected.columns if not self.excluded(f"{path}/{c}")]
        got = [c for c in actual.columns if not self.excluded(f"{path}/{c}")]
        if want != got:
            missing = [c for c in want if c not in got]
            extra = [c for c in got if c not in want]
            detail = (
                f"missing {missing!r}, unexpected {extra!r}"
                if missing or extra
                else "same columns in a different order"
            )
            self.diff(path, f"columns differ: {detail}")
        for column in want:
            if column in got:
                self._column(expected[column], actual[column], f"{path}/{column}")

    def _series(self, expected, actual, path):
        self._index(expected.index, actual.index, f"{path} index")
        if expected.name != actual.name:
            self.diff(path, f"name differs: {expected.name!r} vs {actual.name!r}")
        self._column(expected, actual, path)

    def _index(self, expected, actual, path):
        if type(expected) is not type(actual):
            self.diff(path, f"type differs: {_type_name(expected)}")
            return
        if list(expected.names) != list(actual.names):
            self.diff(path, f"names differ: {list(expected.names)!r}")
        if str(expected.dtype) != str(actual.dtype):
            self.diff(path, f"dtype differs: {expected.dtype} vs {actual.dtype}")
            return
        self._values(expected.to_numpy(), actual.to_numpy(), path)

    def _column(self, expected, actual, path):
        if str(expected.dtype) != str(actual.dtype):
            self.diff(path, f"dtype differs: {expected.dtype} vs {actual.dtype}")
            return
        if str(expected.dtype) == "category":
            if expected.cat.ordered != actual.cat.ordered:
                self.diff(path, "categorical 'ordered' flag differs")
            self._index(
                expected.cat.categories, actual.cat.categories, f"{path} categories"
            )
            self._values(
                expected.cat.codes.to_numpy(), actual.cat.codes.to_numpy(), path
            )
            return
        self._values(expected.to_numpy(), actual.to_numpy(), path)

    def _values(self, expected, actual, path):
        if expected.shape != actual.shape:
            self.diff(path, f"length differs: {expected.shape} vs {actual.shape}")
        elif expected.dtype != actual.dtype:
            self.diff(path, f"dtype differs: {expected.dtype} vs {actual.dtype}")
        else:
            self._array_values(expected, actual, path)

    # -- files ------------------------------------------------------------

    def _file(self, expected, actual, path):
        e_path, a_path = expected.path, actual.path
        for side in (e_path, a_path):
            if side.is_symlink() or not side.is_file():
                self.skip(path, f"not a regular file: {side.name}")
                return
        suffix = e_path.suffix.lower()
        if a_path.suffix.lower() != suffix:
            self.diff(path, f"file type differs: {e_path.name} vs {a_path.name}")
            return
        if suffix in (".csv", ".tsv"):
            import pandas as pd

            sep = "\t" if suffix == ".tsv" else ","
            self.value(pd.read_csv(e_path, sep=sep), pd.read_csv(a_path, sep=sep), path)
        elif suffix == ".json":
            loaded = []
            for side in (e_path, a_path):
                try:
                    loaded.append(json.loads(side.read_text(encoding="utf-8")))
                except ValueError as error:
                    self.diff(path, f"{side.name} is not valid JSON: {error}")
                    return
            self.value(loaded[0], loaded[1], path)
        elif suffix == ".h5ad":
            import anndata

            self.value(anndata.read_h5ad(e_path), anndata.read_h5ad(a_path), path)
        else:
            self.compared += 1
            if _sha256(e_path) != _sha256(a_path):
                self.diff(path, "file contents differ")


def _first(mask: Any) -> Any:
    np = sys.modules["numpy"]
    mask = np.asarray(mask)
    if mask.ndim == 0:
        return ()
    index = np.argwhere(mask)[0]
    return tuple(int(i) for i in index) if index.size > 1 else int(index[0])


def _canonical_csr(matrix: Any) -> Any:
    canon = matrix.tocsr(copy=True)
    canon.sum_duplicates()
    canon.eliminate_zeros()
    canon.sort_indices()
    return canon


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = ["MAX_FINDINGS", "OutputFile", "compare", "output_files"]
