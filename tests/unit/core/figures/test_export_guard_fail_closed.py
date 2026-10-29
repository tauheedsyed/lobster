"""The export guard fails closed: any error while classifying a target refuses it.

In :func:`assert_export_allowed` the ``except Exception: pass`` (bandit B110) only
leaves the ``try`` block. The ``else`` branch is the single way to return, and every
caught error reaches the fixed refusal raised after the handler. Errors that are not
``Exception`` (an interrupt) propagate unchanged; none of them becomes an allowed
write. The classifier is replaced here only to raise each kind of error; the real
classifier failure is covered in ``test_export_guard_contract.py``.
"""

import pytest

from lobster.core.figures import export_guard as G
from lobster.core.figures import paths as P
from lobster.core.figures.errors import ManagedPathError
from lobster.core.figures.export_guard import MANAGED_EXPORT_REASON


class _Unexpected(Exception):
    pass


ERRORS = [
    ManagedPathError("a refusal naming /private/ws/plots/fig/rev_1/x.csv"),
    OSError(13, "Permission denied", "/private/ws/exports/x.csv"),
    ValueError("embedded null byte"),
    RuntimeError("classifier state"),
    RecursionError("maximum recursion depth exceeded"),
    _Unexpected("anything else"),
]


def _target(tmp_path):
    return tmp_path / "exports" / "x.csv"


@pytest.mark.parametrize("error", ERRORS, ids=lambda e: type(e).__name__)
def test_any_classifier_error_is_the_fixed_refusal(tmp_path, monkeypatch, error):
    def failing(workspace, target):
        raise error

    monkeypatch.setattr(P, "assert_not_managed", failing)
    with pytest.raises(ManagedPathError) as refused:
        G.assert_export_allowed(tmp_path, _target(tmp_path))
    assert refused.value is not error
    assert str(refused.value) == MANAGED_EXPORT_REASON
    assert refused.value.details == {}
    # Raised after the handler: neither a cause nor a context is kept.
    assert refused.value.__cause__ is None and refused.value.__context__ is None
    assert G.export_refused(tmp_path, _target(tmp_path)) is True


def test_only_a_clean_classification_allows(tmp_path, monkeypatch):
    checked = []
    monkeypatch.setattr(P, "assert_not_managed", lambda w, t: checked.append(t))
    assert G.assert_export_allowed(tmp_path, _target(tmp_path)) is None
    assert G.export_refused(tmp_path, _target(tmp_path)) is False
    assert checked == [_target(tmp_path)] * 2


def test_an_interrupt_propagates_and_is_never_an_allowed_write(tmp_path, monkeypatch):
    def interrupted(workspace, target):
        raise KeyboardInterrupt

    monkeypatch.setattr(P, "assert_not_managed", interrupted)
    with pytest.raises(KeyboardInterrupt):
        G.assert_export_allowed(tmp_path, _target(tmp_path))
    with pytest.raises(KeyboardInterrupt):
        G.export_refused(tmp_path, _target(tmp_path))
