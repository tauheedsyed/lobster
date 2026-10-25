"""The price snapshot generator: filtering, verbatim costs, provenance and safety.

No test here reaches the network: downloads go through injected fake openers and
fetchers, and regeneration runs from local fixtures.
"""

from __future__ import annotations

import hashlib
import http.client
import io
import json
import ssl
import stat
import urllib.error
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from lobster.pricing import schema

pytestmark = pytest.mark.no_auto_config

FETCHED_AT = "2026-10-24T23:15:51Z"
ENGINE_PROVIDERS = [
    "anthropic",
    "azure",
    "bedrock",
    "gemini",
    "nebius",
    "openai",
    "openrouter",
]


def parse(payload) -> dict:
    """What the generator sees: the payload parsed with exact decimals."""
    return schema.loads_json(json.dumps(payload))


# ---------------------------------------------------------------------- filtering


def test_only_the_supported_providers_are_kept(snapshot):
    assert list(snapshot["providers"]) == ENGINE_PROVIDERS
    assert {k: v["modelsdev_provider"] for k, v in snapshot["providers"].items()} == {
        "anthropic": "anthropic",
        "azure": "azure",
        "bedrock": "amazon-bedrock",
        "gemini": "google",
        "nebius": "nebius",
        "openai": "openai",
        "openrouter": "openrouter",
    }
    # The same model ids exist under unsupported upstream providers, priced at 99.
    for block in snapshot["providers"].values():
        for entry in block["models"].values():
            assert 99 not in entry["cost"].values()
    rendered = json.dumps(snapshot, default=str)
    for ignored in ("google-vertex", "azure-cognitive-services", "ollama-cloud"):
        assert ignored not in rendered
    assert (
        snapshot["providers"]["gemini"]["models"]["acme-flash"]["cost"]["output"] == 5
    )


def test_each_model_keeps_only_its_cost_object_verbatim(snapshot, payload):
    expected = parse(payload)
    for engine, block in snapshot["providers"].items():
        upstream = expected[schema.PROVIDER_MAP[engine]]["models"]
        for model_id, entry in block["models"].items():
            assert list(entry) == ["cost"], "nothing but the cost is carried over"
            assert entry["cost"] == upstream[model_id]["cost"]
            assert list(entry["cost"]) == list(upstream[model_id]["cost"])  # key order


def test_models_without_a_usable_cost_are_omitted_not_zeroed(snapshot):
    anthropic = snapshot["providers"]["anthropic"]
    assert "acme-nocost" not in anthropic["models"]
    assert anthropic["source_model_count"] == 4  # counted in the source, not priced
    assert len(anthropic["models"]) == 3

    gemini = snapshot["providers"]["gemini"]
    assert list(gemini["models"]) == ["acme-flash"]  # `{}` and `null` cost dropped
    assert gemini["source_model_count"] == 3
    assert "acme-image" not in snapshot["providers"]["openai"]["models"]


def test_null_and_zero_survive_generation_and_serialisation(update_pricing, snapshot):
    text = update_pricing.render_json(snapshot, compact_depth=4)
    stored = schema.loads_json(text)["providers"]

    flash = stored["gemini"]["models"]["acme-flash"]["cost"]
    assert flash == {
        "input": None,
        "output": 5,
        "cache_read": None,
        "cache_write": None,
        "reasoning": None,
    }
    assert '"input": null' in text
    free = stored["anthropic"]["models"]["acme-free"]["cost"]
    assert free == {"input": 0, "output": 0} and free["input"] is not None
    assert stored["openrouter"]["models"]["acme/router-free"]["cost"]["input"] == 0


def test_conditional_pricing_is_carried_over_unresolved(snapshot, payload):
    expected = parse(payload)
    haiku = snapshot["providers"]["anthropic"]["models"]["acme-haiku"]["cost"]
    long_context = snapshot["providers"]["bedrock"]["models"]["eu.acme.long-v1:0"][
        "cost"
    ]

    assert (
        haiku["tiers"] == expected["anthropic"]["models"]["acme-haiku"]["cost"]["tiers"]
    )
    assert haiku["tiers"][0]["tier"] == {"type": "context", "size": 100000}
    assert haiku["input"] == Decimal("0.1")  # the base rate, untouched by the tier
    assert long_context["context_over_200k"] == {"input": 8, "output": 30}
    assert (
        long_context["input_audio"] == Decimal("2.5")
        and long_context["reasoning"] == 20
    )


def test_exact_ids_are_kept_and_ordered(snapshot):
    assert list(snapshot["providers"]["bedrock"]["models"]) == [
        "acme.sonnet-v1:0",
        "eu.acme.long-v1:0",
        "global.acme.sonnet-v1:0",
        "us.acme.sonnet-v1:0",
    ]
    assert "~acme/alias-latest" in snapshot["providers"]["openrouter"]["models"]


def test_unrelated_source_fields_and_bulk_never_reach_the_snapshot(
    update_pricing, payload
):
    payload["anthropic"]["models"]["acme-sonnet"]["description"] = "bulk " * 200_000
    raw = json.dumps(payload).encode()

    text = update_pricing.render_json(
        update_pricing.build_snapshot(raw, FETCHED_AT), compact_depth=4
    )

    assert len(raw) > 1_000_000 and len(text) < 20_000
    for dropped in (
        "bulk",
        "Name of",
        '"limit"',
        "modalities",
        "release_date",
        "EXAMPLE",
    ):
        assert dropped not in text


# ---------------------------------------------------------------------- provenance


def test_provenance_fields(update_pricing, payload_bytes, snapshot):
    digest = hashlib.sha256(payload_bytes).hexdigest()

    assert snapshot["schema_version"] == 1
    assert snapshot["unit"] == "USD per 1M tokens"
    assert snapshot["fetched_at"] == FETCHED_AT
    assert snapshot["snapshot_id"] == f"20261024T231551Z-{digest[:12]}"
    source = snapshot["source"]
    assert source["sha256"] == digest and source["size_bytes"] == len(payload_bytes)
    assert source["url"] == "https://models.dev/api.json"
    assert source["name"] == "models.dev" and source["retrieved_via"] == "https"
    assert source["license"] == "MIT"
    assert list(snapshot) == [
        "schema_version",
        "snapshot_id",
        "fetched_at",
        "unit",
        "source",
        "providers",
    ]


def test_full_mit_attribution_is_embedded(snapshot):
    source = snapshot["source"]
    text = source["license_text"]

    assert text == schema.MODELS_DEV_LICENSE_TEXT
    assert text.startswith("MIT License\n\nCopyright (c) 2025 models.dev\n")
    for clause in (
        "Permission is hereby granted, free of charge",
        "The above copyright notice and this permission notice shall be included",
        'THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND',
    ):
        assert clause in text
    assert "models.dev" in source["attribution"] and "MIT" in source["attribution"]


def test_local_files_are_labelled_as_such(update_pricing, payload_bytes):
    snapshot = update_pricing.build_snapshot(
        payload_bytes, FETCHED_AT, retrieved_via="file"
    )

    assert snapshot["source"]["retrieved_via"] == "file"


@pytest.mark.parametrize(
    "value",
    [
        FETCHED_AT,
        "2026-10-24T23:15:51+00:00",
        "2026-10-25T01:15:51+02:00",
        "2026-10-24T23:15:51.987654Z",
        datetime(2026, 10, 24, 23, 15, 51, tzinfo=timezone.utc),
        datetime(2026, 10, 24, 16, 15, 51, tzinfo=timezone(timedelta(hours=-7))),
    ],
)
def test_fetched_at_is_normalised_to_whole_second_utc(
    update_pricing, payload_bytes, value
):
    snapshot = update_pricing.build_snapshot(payload_bytes, value)

    assert snapshot["fetched_at"] == FETCHED_AT
    assert snapshot["snapshot_id"].startswith("20261024T231551Z-")


@pytest.mark.parametrize(
    "value", ["2026-10-24", "2026-10-24T23:15:51", "yesterday", "", 1729811751, None]
)
def test_fetched_at_without_a_timezone_or_unparseable_is_refused(
    update_pricing, payload_bytes, value
):
    with pytest.raises(update_pricing.SnapshotError, match="fetched_at"):
        update_pricing.build_snapshot(payload_bytes, value)
    with pytest.raises(update_pricing.SnapshotError, match="fetched_at"):
        update_pricing.build_snapshot(
            payload_bytes, datetime(2026, 10, 24, 23, 15, 51)  # naive
        )


def test_snapshot_id_tracks_both_source_bytes_and_fetch_time(
    update_pricing, payload, payload_bytes
):
    base = update_pricing.build_snapshot(payload_bytes, FETCHED_AT)["snapshot_id"]
    later = update_pricing.build_snapshot(payload_bytes, "2026-10-25T00:00:00Z")
    payload["openai"]["models"]["acme-gpt"]["cost"]["input"] = 2.6
    changed = update_pricing.build_snapshot(json.dumps(payload).encode(), FETCHED_AT)

    assert later["snapshot_id"] != base and later["snapshot_id"].endswith(base[-12:])
    assert changed["snapshot_id"] != base and changed["snapshot_id"][:16] == base[:16]


def test_derive_snapshot_id_rejects_malformed_hashes():
    for bad in ("", "abc", "G" * 64, "A" * 64, None):
        with pytest.raises(schema.PricingDataError):
            schema.derive_snapshot_id(bad, FETCHED_AT)


def test_generation_is_deterministic_and_independent_of_source_order(
    update_pricing, payload
):
    forward = json.dumps(payload).encode()
    backward = json.dumps(
        {
            provider: {**block, "models": dict(reversed(block["models"].items()))}
            for provider, block in reversed(payload.items())
        }
    ).encode()

    first = update_pricing.render_json(
        update_pricing.build_snapshot(forward, FETCHED_AT), compact_depth=4
    )
    again = update_pricing.render_json(
        update_pricing.build_snapshot(forward, FETCHED_AT), compact_depth=4
    )
    reordered = update_pricing.render_json(
        update_pricing.build_snapshot(backward, FETCHED_AT), compact_depth=4
    )

    assert first == again
    # Only the hash-derived fields may differ; every price and ordering is identical.
    assert json.loads(first)["providers"] == json.loads(reordered)["providers"]
    assert list(json.loads(first)["providers"]["bedrock"]["models"]) == sorted(
        json.loads(first)["providers"]["bedrock"]["models"]
    )


# --------------------------------------------------------------- invalid sources


def _mutated(payload, mutate) -> bytes:
    mutate(payload)
    return json.dumps(payload).encode()


@pytest.mark.parametrize(
    "case,raw,match",
    [
        ("empty", b"", "empty"),
        ("not json", b"<html><body>502 Bad Gateway</body></html>", "not usable JSON"),
        (
            "truncated",
            b'{"anthropic": {"id": "anthropic", "models": {"x"',
            "not usable JSON",
        ),
        ("list", b"[]", "JSON object"),
        ("scalar", b"3", "JSON object"),
        ("not utf-8", b'{"a": "\xff\xfe"}', "not UTF-8"),
        (
            "nan",
            b'{"anthropic": {"models": {"m": {"cost": {"input": NaN}}}}}',
            "non-finite",
        ),
        ("duplicate keys", b'{"anthropic": {}, "anthropic": {}}', "duplicate key"),
        ("empty object", b"{}", "no usable"),
    ],
)
def test_garbage_sources_are_rejected(update_pricing, case, raw, match):
    with pytest.raises(update_pricing.SnapshotError, match=match):
        update_pricing.build_snapshot(raw, FETCHED_AT)


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda p: p.pop("amazon-bedrock"), "'amazon-bedrock'"),
        (lambda p: p.pop("google"), "'google'"),
        (lambda p: p.update(nebius=["not", "a", "provider"]), "'nebius'"),
        (lambda p: p["openai"].pop("models"), "'openai'"),
        (lambda p: p["openai"].update(models=[]), "'openai'"),
        (lambda p: p["openai"].update(id="azure"), "different id"),
        (
            lambda p: p["azure"].update(
                models={"only": {"id": "only", "name": "no cost anywhere"}}
            ),
            "no priced models",
        ),
        (lambda p: p["openai"]["models"].update({"bad": "text"}), "must be an object"),
        (
            lambda p: p["openai"]["models"]["acme-gpt"].update(id="another"),
            "differs from its key",
        ),
        (
            lambda p: p["openai"]["models"]["acme-gpt"].update(cost=[1, 2]),
            "non-empty object",
        ),
        (
            lambda p: p["openai"]["models"]["acme-gpt"].update(cost="free"),
            "non-empty object",
        ),
        (
            lambda p: p["openai"]["models"]["acme-gpt"]["cost"].update(input="3"),
            "openai/acme-gpt.input",
        ),
        (
            lambda p: p["openai"]["models"]["acme-gpt"]["cost"].update(input=True),
            "openai/acme-gpt.input",
        ),
        (
            lambda p: p["openai"]["models"]["acme-gpt"]["cost"].update(output=-0.5),
            "outside 0",
        ),
        (
            lambda p: p["openai"]["models"]["acme-gpt"]["cost"].update(output=1e30),
            "outside 0",
        ),
        (
            lambda p: p["anthropic"]["models"]["acme-haiku"]["cost"]["tiers"][0].update(
                input=-1
            ),
            r"tiers\[0\].input",
        ),
        (
            lambda p: p["anthropic"]["models"]["acme-haiku"]["cost"].update(tiers={}),
            "tiers: must be a list",
        ),
        (
            lambda p: p["amazon-bedrock"]["models"]["eu.acme.long-v1:0"]["cost"].update(
                context_over_200k=[]
            ),
            "context_over_200k: must be an object",
        ),
    ],
)
def test_structurally_invalid_sources_are_rejected(
    update_pricing, payload, mutate, match
):
    raw = _mutated(payload, mutate)

    with pytest.raises(update_pricing.SnapshotError, match=match):
        update_pricing.build_snapshot(raw, FETCHED_AT)


# ------------------------------------------------------------------- serialisation


def test_each_model_is_one_line_so_a_price_change_is_a_one_line_diff(
    update_pricing, snapshot
):
    text = update_pricing.render_json(snapshot, compact_depth=4)
    lines = [line.strip().rstrip(",") for line in text.splitlines()]

    assert text.endswith("}\n") and not text.endswith("\n\n")
    assert schema.loads_json(text) == snapshot
    for block in snapshot["providers"].values():
        for model_id, entry in block["models"].items():
            # The whole model is a self-contained JSON fragment on a single line.
            fragments = [
                schema.loads_json("{" + line + "}")
                for line in lines
                if line.startswith(json.dumps(model_id) + ":")
            ]
            assert {model_id: entry} in fragments


def test_numbers_that_would_change_when_written_are_refused(update_pricing, snapshot):
    snapshot["providers"]["openai"]["models"]["acme-gpt"]["cost"]["tiers"] = [
        {"input": 1, "tier": {"type": "context", "size": Decimal("1E-400")}}
    ]

    with pytest.raises(update_pricing.SnapshotError, match="round-trip"):
        update_pricing.render_json(snapshot, compact_depth=4)


def test_non_serialisable_values_are_refused(update_pricing):
    with pytest.raises(update_pricing.SnapshotError, match="cannot serialise"):
        update_pricing.render_json({"a": object()}, compact_depth=4)
    with pytest.raises(update_pricing.SnapshotError, match="cannot serialise"):
        update_pricing.render_json({"a": float("nan")}, compact_depth=4)


def test_write_atomically_replaces_every_file_with_readable_permissions(
    update_pricing, tmp_path
):
    (tmp_path / "a.json").write_text("old", encoding="utf-8")

    update_pricing.write_atomically(
        {tmp_path / "a.json": "new-a\n", tmp_path / "b.json": "new-b\n"}
    )

    assert (tmp_path / "a.json").read_text() == "new-a\n"
    assert (tmp_path / "b.json").read_text() == "new-b\n"
    assert stat.S_IMODE((tmp_path / "b.json").stat().st_mode) == 0o644
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.json", "b.json"]


def test_write_atomically_changes_nothing_if_any_file_cannot_be_staged(
    update_pricing, tmp_path
):
    (tmp_path / "a.json").write_text("old", encoding="utf-8")

    with pytest.raises(OSError):
        update_pricing.write_atomically(
            {tmp_path / "a.json": "new", tmp_path / "missing-dir" / "b.json": "new"}
        )

    assert (tmp_path / "a.json").read_text() == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["a.json"]  # no staged leftovers


# ------------------------------------------------------------------------- download


class FakeResponse:
    """Just enough of ``http.client.HTTPResponse`` for ``fetch_source``."""

    def __init__(self, body=b"", *, status=200, headers=None, url=None):
        self._stream = io.BytesIO(body)
        self.status = status
        self.headers = headers or {}
        self._url = url or "https://models.dev/api.json"
        self.bytes_read = 0

    def read(self, size=-1):
        chunk = self._stream.read(size)
        self.bytes_read += len(chunk)
        return chunk

    def geturl(self):
        return self._url

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def opener(outcome):
    """A stand-in for ``urllib.request.urlopen`` that records how it was called."""
    calls = []

    def urlopen(request, timeout=None, context=None):
        calls.append({"request": request, "timeout": timeout, "context": context})
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    urlopen.calls = calls
    return urlopen


def test_fetch_returns_the_body_and_bounds_the_request(update_pricing):
    urlopen = opener(FakeResponse(b'{"ok": true}', headers={"Content-Length": "12"}))

    body = update_pricing.fetch_source(urlopen=urlopen)

    assert body == b'{"ok": true}'
    (call,) = urlopen.calls
    assert call["request"].full_url == "https://models.dev/api.json"
    assert call["request"].get_header("User-agent")
    assert call["timeout"] == update_pricing.FETCH_TIMEOUT_SECONDS
    assert isinstance(call["context"], ssl.SSLContext)
    assert call["context"].verify_mode == ssl.CERT_REQUIRED
    assert 0 < update_pricing.FETCH_TIMEOUT_SECONDS <= 60
    assert 0 < update_pricing.MAX_SOURCE_BYTES <= 64 * 1024 * 1024


@pytest.mark.parametrize(
    "url", ["http://models.dev/api.json", "file:///etc/hosts", "ftp://x/y", ""]
)
def test_only_https_urls_are_fetched(update_pricing, url):
    urlopen = opener(FakeResponse(b"{}"))

    with pytest.raises(update_pricing.SnapshotError, match="non-https"):
        update_pricing.fetch_source(url, urlopen=urlopen)
    assert urlopen.calls == []


def test_a_redirect_to_plain_http_is_refused(update_pricing):
    response = FakeResponse(b"{}", url="http://models.dev/api.json")

    with pytest.raises(update_pricing.SnapshotError, match="non-https"):
        update_pricing.fetch_source(urlopen=opener(response))


@pytest.mark.parametrize("status", [204, 301, 403, 500, 503])
def test_non_200_answers_are_refused(update_pricing, status):
    with pytest.raises(update_pricing.SnapshotError, match=f"HTTP {status}"):
        update_pricing.fetch_source(urlopen=opener(FakeResponse(b"{}", status=status)))


def test_an_oversized_declared_length_is_refused_before_reading(update_pricing):
    response = FakeResponse(b"x" * 100, headers={"Content-Length": "100"})

    with pytest.raises(update_pricing.SnapshotError, match="byte limit"):
        update_pricing.fetch_source(urlopen=opener(response), max_bytes=50)
    assert response.bytes_read == 0


def test_an_oversized_stream_without_a_length_is_cut_off(update_pricing):
    response = FakeResponse(b"x" * 1_000_000)

    with pytest.raises(update_pricing.SnapshotError, match="exceeds"):
        update_pricing.fetch_source(urlopen=opener(response), max_bytes=100)
    assert response.bytes_read <= 100 + 64 * 1024  # never read the whole payload


def test_a_slow_download_is_abandoned_at_the_deadline(update_pricing):
    ticks = iter(range(0, 10_000, 100))  # every clock read advances 100 s

    with pytest.raises(update_pricing.SnapshotError, match="exceeded"):
        update_pricing.fetch_source(
            urlopen=opener(FakeResponse(b"x" * 200_000)),
            deadline=150,
            clock=lambda: next(ticks),
        )


@pytest.mark.parametrize(
    "failure",
    [
        urllib.error.URLError("name resolution failed"),
        urllib.error.HTTPError("https://models.dev/api.json", 503, "down", {}, None),
        TimeoutError("timed out"),
        ConnectionResetError("reset"),
        ssl.SSLError("certificate verify failed"),
        http.client.IncompleteRead(b"partial"),
        http.client.RemoteDisconnected("closed"),
    ],
    ids=lambda failure: type(failure).__name__,
)
def test_network_failures_become_snapshot_errors(update_pricing, failure):
    with pytest.raises(update_pricing.SnapshotError, match="could not fetch"):
        update_pricing.fetch_source(urlopen=opener(failure))


def test_local_sources_are_bounded_too(update_pricing, tmp_path):
    capture = tmp_path / "capture.json"
    capture.write_bytes(b"x" * 100)

    assert update_pricing.read_local_source(capture, max_bytes=100) == b"x" * 100
    with pytest.raises(update_pricing.SnapshotError, match="exceeds"):
        update_pricing.read_local_source(capture, max_bytes=99)
    with pytest.raises(update_pricing.SnapshotError, match="cannot read source"):
        update_pricing.read_local_source(tmp_path / "missing.json")
    with pytest.raises(update_pricing.SnapshotError, match="cannot read source"):
        update_pricing.read_local_source(tmp_path)


# ------------------------------------------------------------------------------ CLI


@dataclass
class Work:
    """An output directory, a local capture and a providers directory."""

    output: Path
    source: Path
    providers: Path

    def args(self, *extra: str) -> list[str]:
        return [
            "--output-dir",
            str(self.output),
            "--providers-dir",
            str(self.providers),
            *extra,
        ]

    def snapshot_args(self) -> list[str]:
        return self.args("--source", str(self.source), "--fetched-at", FETCHED_AT)


@pytest.fixture
def workdir(tmp_path, payload_bytes, providers_dir) -> Work:
    output = tmp_path / "out"
    output.mkdir()
    capture = tmp_path / "capture.json"
    capture.write_bytes(payload_bytes)
    return Work(output, capture, providers_dir)


def test_local_source_regeneration_is_deterministic(update_pricing, workdir, capsys):
    assert update_pricing.main(workdir.snapshot_args()) == 0
    first = {p.name: p.read_bytes() for p in workdir.output.iterdir()}
    assert update_pricing.main(workdir.snapshot_args()) == 0
    second = {p.name: p.read_bytes() for p in workdir.output.iterdir()}

    assert sorted(first) == ["coverage.json", "prices.json"]
    assert first == second
    prices = schema.loads_json(first["prices.json"].decode())
    assert prices["source"]["retrieved_via"] == "file"
    assert prices["fetched_at"] == FETCHED_AT
    coverage = json.loads(first["coverage.json"])
    assert coverage["snapshot_id"] == prices["snapshot_id"]
    assert "snapshot 20261024T231551Z-" in capsys.readouterr().out


def test_default_mode_fetches_and_stamps_the_clock(
    update_pricing, workdir, payload_bytes
):
    moment = datetime(2026, 10, 25, 1, 2, 3, tzinfo=timezone.utc)

    code = update_pricing.main(
        workdir.args(), fetch=lambda: payload_bytes, now=lambda: moment
    )

    prices = schema.loads_json((workdir.output / "prices.json").read_text())
    assert code == 0
    assert prices["fetched_at"] == "2026-10-25T01:02:03Z"
    assert prices["source"]["retrieved_via"] == "https"
    assert prices["snapshot_id"].startswith("20261025T010203Z-")


def test_the_output_directory_receives_exactly_the_two_generated_files(
    update_pricing, workdir
):
    (workdir.output / "overrides.json").write_text(
        '{"schema_version": 1, "overrides": []}', encoding="utf-8"
    )

    assert update_pricing.main(workdir.snapshot_args()) == 0

    assert sorted(p.name for p in workdir.output.iterdir()) == [
        "coverage.json",
        "overrides.json",
        "prices.json",
    ]
    assert (workdir.output / "overrides.json").read_text() == (
        '{"schema_version": 1, "overrides": []}'
    )


SENTINELS = {"prices.json": b"previous prices", "coverage.json": b"previous coverage"}


def _seed(workdir):
    for name, content in SENTINELS.items():
        (workdir.output / name).write_bytes(content)


def _untouched(workdir):
    found = {p.name: p.read_bytes() for p in workdir.output.iterdir()}
    return found == SENTINELS


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b'{"anthropic": {"models": {"x"',
        b"<html>maintenance</html>",
        b"{}",
        b"[]",
        b"\xff\xfe\x00",
    ],
    ids=["empty", "truncated", "html", "no-providers", "list", "binary"],
)
def test_an_invalid_local_source_leaves_the_existing_files_untouched(
    update_pricing, workdir, capsys, raw
):
    _seed(workdir)
    workdir.source.write_bytes(raw)

    code = update_pricing.main(workdir.snapshot_args())

    assert code == 1
    assert _untouched(workdir)
    assert "update_pricing: error" in capsys.readouterr().err


def test_an_invalid_price_in_the_source_leaves_the_existing_files_untouched(
    update_pricing, workdir, payload
):
    _seed(workdir)
    payload["openai"]["models"]["acme-gpt"]["cost"]["input"] = -3
    workdir.source.write_text(json.dumps(payload), encoding="utf-8")

    assert update_pricing.main(workdir.snapshot_args()) == 1
    assert _untouched(workdir)


def test_a_failed_or_truncated_download_leaves_the_existing_files_untouched(
    update_pricing, workdir, payload_bytes, capsys
):
    _seed(workdir)

    def unreachable():
        raise update_pricing.SnapshotError("could not fetch: connection refused")

    assert update_pricing.main(workdir.args(), fetch=unreachable) == 1
    assert update_pricing.main(workdir.args(), fetch=lambda: payload_bytes[:-100]) == 1
    assert _untouched(workdir)
    assert "connection refused" in capsys.readouterr().err


def test_invalid_overrides_or_providers_leave_the_existing_files_untouched(
    update_pricing, workdir, tmp_path
):
    _seed(workdir)
    (workdir.output / "overrides.json").write_text(
        '{"schema_version": 1, "overrides": [{"provider": "bedrock"}]}',
        encoding="utf-8",
    )
    assert update_pricing.main(workdir.snapshot_args()) == 1

    (workdir.output / "overrides.json").unlink()
    broken = tmp_path / "broken-providers"
    broken.mkdir()
    (broken / "x_provider.py").write_text("KNOWN_MODELS = [", encoding="utf-8")
    code = update_pricing.main(
        ["--output-dir", str(workdir.output), "--providers-dir", str(broken)]
        + ["--source", str(workdir.source), "--fetched-at", FETCHED_AT]
    )
    assert code == 1
    assert _untouched(workdir)


def test_an_unwritable_output_directory_fails_cleanly(
    update_pricing, workdir, tmp_path
):
    code = update_pricing.main(
        ["--output-dir", str(tmp_path / "does-not-exist"), "--providers-dir"]
        + [str(workdir.providers), "--source", str(workdir.source)]
        + ["--fetched-at", FETCHED_AT]
    )

    assert code == 1


@pytest.mark.parametrize(
    "extra",
    [["--source", "capture.json"], ["--fetched-at", FETCHED_AT]],
    ids=["source-only", "fetched-at-only"],
)
def test_source_and_fetched_at_must_be_given_together(update_pricing, workdir, extra):
    with pytest.raises(SystemExit) as exit_info:
        update_pricing.main(workdir.args(*extra))

    assert exit_info.value.code == 2
    assert list(workdir.output.iterdir()) == []
