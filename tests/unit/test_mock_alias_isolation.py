"""Mock fixture lifetimes must not change later real consumer construction."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from lobster.config import llm_factory
from tests.conftest import _restore_llm_import_aliases, auto_mock_provider_config

pytestmark = pytest.mark.no_auto_config


def test_alias_imported_during_fixture_is_restored_without_preloading(
    monkeypatch, mocker
):
    original = llm_factory.create_llm
    fixture = auto_mock_provider_config.__wrapped__(
        SimpleNamespace(keywords={}), monkeypatch, mocker
    )
    next(fixture)
    mocked = llm_factory.create_llm
    assert mocked is not original
    consumer = ModuleType("lobster.synthetic_alias_lifetime")
    monkeypatch.setitem(sys.modules, consumer.__name__, consumer)
    # This consumer did not exist before the mock fixture. Import is real, with
    # only the provider factory mocked by the ordinary unit fixture under test.
    exec(
        "from lobster.config.llm_factory import create_llm", consumer.__dict__
    )  # nosec B102 # Fixed trusted import statement, no caller input.
    assert consumer.create_llm is mocked
    fixture.close()
    assert consumer.create_llm is original
    assert llm_factory.create_llm is original


def test_cleanup_is_identity_scoped_and_does_not_import_consumers(monkeypatch):
    mocked, original, unrelated = Mock(), object(), object()
    consumer = ModuleType("lobster.synthetic_alias_identity")
    consumer.alias = mocked
    consumer.unrelated = unrelated
    outside = ModuleType("unrelated_synthetic_alias")
    outside.alias = mocked
    monkeypatch.setitem(sys.modules, consumer.__name__, consumer)
    monkeypatch.setitem(sys.modules, outside.__name__, outside)
    before = set(sys.modules)
    _restore_llm_import_aliases(mocked, original)
    assert set(sys.modules) == before
    assert consumer.alias is original
    assert consumer.unrelated is unrelated
    assert outside.alias is mocked


def test_no_auto_config_still_yields_without_mocking(monkeypatch, mocker):
    original = llm_factory.create_llm
    fixture = auto_mock_provider_config.__wrapped__(
        SimpleNamespace(keywords={"no_auto_config": True}), monkeypatch, mocker
    )
    assert next(fixture) == {}
    assert llm_factory.create_llm is original
    fixture.close()
    assert llm_factory.create_llm is original
