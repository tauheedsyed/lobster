"""Recoverable generator failures and immutable reader provenance."""

import hashlib
import os
import threading
from types import SimpleNamespace

import pytest

from lobster.pricing import SNAPSHOT, PricingDataError, get_rates, schema

pytestmark = pytest.mark.no_auto_config


def test_content_length_mismatch_is_rejected(update_pricing):
    class Response:
        status = 200
        headers = {"Content-Length": "20"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def geturl(self):
            return schema.SOURCE_URL

        def read(self, size):
            if getattr(self, "done", False):
                return b""
            self.done = True
            return b"{}"

    with pytest.raises(update_pricing.SnapshotError, match="Content-Length"):
        update_pricing.fetch_source(urlopen=lambda *a, **kw: Response())


def test_watchdog_interrupts_a_blocking_body_read(update_pricing):
    closed = threading.Event()
    transport = SimpleNamespace(shutdown=lambda how: closed.set())

    class Response:
        status = 200
        headers = {}
        fp = SimpleNamespace(raw=SimpleNamespace(_sock=transport))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def geturl(self):
            return schema.SOURCE_URL

        def read(self, size):
            assert closed.wait(2), "deadline did not interrupt blocking read"
            return b""

    with pytest.raises(update_pricing.SnapshotError, match="exceeded"):
        update_pricing.fetch_source(urlopen=lambda *a, **kw: Response(), deadline=0.02)
    assert closed.is_set()


def test_absolute_deadline_covers_blocked_open_and_closes_late_response(update_pricing):
    release, closed = threading.Event(), threading.Event()

    class Response:
        def close(self):
            closed.set()

        def read(self, size):
            pytest.fail("body of a late response must not be read")

    def opener(*args, **kwargs):
        assert release.wait(2)
        return Response()

    try:
        with pytest.raises(update_pricing.SnapshotError, match="exceeded"):
            update_pricing.fetch_source(urlopen=opener, deadline=0.02)
    finally:
        release.set()
    assert closed.wait(2)


@pytest.mark.parametrize("inside_replace", [False, True])
def test_keyboard_interrupt_restores_outputs_and_preserves_backups(
    update_pricing, tmp_path, monkeypatch, inside_replace
):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text("old-a")
    b.write_text("old-b")
    replace = os.replace

    def interrupt(source, target):
        if inside_replace and target == a and ".backup." not in str(source):
            replace(source, target)
            raise KeyboardInterrupt()
        if not inside_replace and target == b and ".backup." not in str(source):
            raise KeyboardInterrupt()
        return replace(source, target)

    monkeypatch.setattr(update_pricing.os, "replace", interrupt)
    with pytest.raises(KeyboardInterrupt):
        update_pricing.write_atomically({a: "new-a", b: "new-b"})
    assert a.read_text() == "old-a" and b.read_text() == "old-b"
    assert {p.read_text() for p in tmp_path.glob("*.tmp")} == {"old-a", "old-b"}


@pytest.mark.parametrize("first_exists", [True, False])
def test_second_replace_failure_rolls_back_first(
    update_pricing, tmp_path, monkeypatch, first_exists
):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    if first_exists:
        a.write_text("old-a")
    b.write_text("old-b")
    replace = os.replace

    def fail_second(source, target):
        if target == b:
            raise OSError("simulated replacement failure")
        return replace(source, target)

    monkeypatch.setattr(update_pricing.os, "replace", fail_second)
    with pytest.raises(OSError, match="previous files restored"):
        update_pricing.write_atomically({a: "new-a", b: "new-b"})
    assert a.read_text() == "old-a" if first_exists else not a.exists()
    assert b.read_text() == "old-b"
    assert sorted(path.name for path in tmp_path.iterdir()) == (
        ["a.json", "b.json"] if first_exists else ["b.json"]
    )


def test_failed_rollback_is_explicit_and_retains_recovery_backup(
    update_pricing, tmp_path, monkeypatch
):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text("old-a")
    b.write_text("old-b")
    replace = os.replace

    def fail(source, target):
        if target == b or ".backup." in str(source):
            raise OSError("simulated failure")
        return replace(source, target)

    monkeypatch.setattr(update_pricing.os, "replace", fail)
    with pytest.raises(OSError, match="rollback incomplete"):
        update_pricing.write_atomically({a: "new-a", b: "new-b"})
    backups = list(tmp_path.glob(".a.json.backup.*"))
    assert len(backups) == 1 and backups[0].read_text() == "old-a"
    assert b.read_text() == "old-b"


@pytest.mark.parametrize("size", [0, -1, True, 1.5, None])
def test_context_threshold_validation_is_shared(
    update_pricing, payload, make_book, size
):
    cost = {
        "input": 1,
        "output": 2,
        "tiers": [{"input": 3, "output": 4, "tier": {"type": "context", "size": size}}],
    }
    with pytest.raises(PricingDataError, match="positive integer"):
        make_book({"openai": {"bad": cost}}).get_rates("openai", "bad")
    payload["openai"]["models"]["acme-gpt"]["cost"] = cost
    import json

    with pytest.raises(update_pricing.SnapshotError, match="positive integer"):
        update_pricing.build_snapshot(
            json.dumps(payload).encode(), "2026-01-01T00:00:00Z"
        )


def test_duplicate_context_thresholds_rejected_at_build_and_lookup(
    update_pricing, payload, make_book
):
    import json

    tier = {"input": 3, "output": 4, "tier": {"type": "context", "size": 100}}
    cost = {"input": 1, "output": 2, "tiers": [tier, tier]}
    with pytest.raises(PricingDataError, match="duplicate"):
        make_book({"openai": {"bad": cost}}).get_rates("openai", "bad")
    payload["openai"]["models"]["acme-gpt"]["cost"] = cost
    with pytest.raises(update_pricing.SnapshotError, match="duplicate"):
        update_pricing.build_snapshot(
            json.dumps(payload).encode(), "2026-01-01T00:00:00Z"
        )


def test_price_book_copies_caller_owned_nested_data(make_book):
    cost = {"input": 1, "output": 2, "context_over_200k": {"input": 3, "output": 4}}
    book = make_book({"openai": {"m": cost}})
    cost["input"] = 100
    cost["context_over_200k"]["input"] = 100
    assert book.get_rates("openai", "m").input == 1
    assert book.get_rates("openai", "m").tiers[0].rates.input == 3


def test_price_book_copies_caller_owned_override(make_book, override_entry):
    entry = override_entry(
        provider="openai", model_id="m", cost={"input": 7, "output": 21}
    )
    book = make_book({"openai": {"m": {"input": 1, "output": 2}}}, overrides=[entry])
    entry["cost"]["input"] = 100
    assert book.get_rates("openai", "m").input == 7


def test_snapshot_records_exact_override_bytes_and_engine_version():
    from importlib.resources import files

    from lobster.version import __version__

    raw = files("lobster.pricing").joinpath("overrides.json").read_bytes()
    assert SNAPSHOT.overrides_sha256 == hashlib.sha256(raw).hexdigest()
    assert SNAPSHOT.lobster_version == __version__
    assert get_rates("google", "gemini-3-pro-preview").source == "override"
