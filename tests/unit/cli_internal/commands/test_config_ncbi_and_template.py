"""`lobster config test` NCBI probe and the generated .env template.

The probe reads JSON (no XML parser on network data), sends the key as a query
parameter, and never echoes the key back. No network: httpx.get is replaced.
"""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from lobster.cli_internal.commands.light import config_commands
from lobster.cli_internal.commands.light.config_commands import (
    _NCBI_ESEARCH_URL,
    _check_ncbi_eutils,
)

KEY = "test-ncbi-key-1234567890"


def _reply(status_code=200, payload=None, json_error=None):
    def json():
        if json_error is not None:
            raise json_error
        return payload

    return SimpleNamespace(status_code=status_code, json=json)


@pytest.fixture
def calls(monkeypatch):
    seen = []

    def install(reply=None, raises=None):
        def fake_get(url, params=None, timeout=None):
            seen.append({"url": url, "params": dict(params), "timeout": timeout})
            if raises is not None:
                raise raises
            return reply

        monkeypatch.setattr(httpx, "get", fake_get)
        return seen

    return install


def test_pass_on_a_normal_search_reply(calls):
    seen = calls(_reply(payload={"esearchresult": {"count": "1", "idlist": ["1"]}}))
    assert _check_ncbi_eutils(KEY) == ("pass", "Connected")
    assert seen[0]["url"] == _NCBI_ESEARCH_URL
    assert seen[0]["params"]["retmode"] == "json"
    assert seen[0]["params"]["api_key"] == KEY
    assert seen[0]["timeout"] == 10.0


def test_no_api_key_param_without_a_key(calls):
    seen = calls(_reply(payload={"esearchresult": {"count": "0"}}))
    assert _check_ncbi_eutils(None)[0] == "pass"
    assert "api_key" not in seen[0]["params"]


def test_search_error_inside_the_result_fails(calls):
    calls(_reply(payload={"esearchresult": {"ERROR": "Invalid query"}}))
    assert _check_ncbi_eutils(None) == ("fail", "Invalid query")


def test_invalid_key_reply_fails_and_masks_the_key(calls):
    calls(_reply(400, {"error": f"API key invalid: {KEY}", "type": "invalid"}))
    status, message = _check_ncbi_eutils(KEY)
    assert status == "fail"
    assert KEY not in message
    assert "API key invalid" in message


def test_transport_error_fails_and_masks_the_key(calls):
    calls(raises=httpx.ConnectError(f"cannot reach host for api_key={KEY}"))
    status, message = _check_ncbi_eutils(KEY)
    assert status == "fail"
    assert KEY not in message and "***" in message


def test_client_setup_error_is_a_failed_check(calls):
    calls(raises=FileNotFoundError("CA bundle not found"))
    assert _check_ncbi_eutils(None) == ("fail", "CA bundle not found")


def test_non_json_reply_fails(calls):
    calls(_reply(502, json_error=ValueError("Expecting value")))
    assert _check_ncbi_eutils(None) == ("fail", "Expecting value")


@pytest.mark.parametrize("payload", [["not", "a", "dict"], {"header": {}}])
def test_unexpected_shapes_fail_with_the_status(calls, payload):
    calls(_reply(200, payload))
    assert _check_ncbi_eutils(None) == ("fail", "HTTP 200: unexpected response")


def test_generated_env_template_names_only_live_model_settings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "lobster.ui.console_manager.get_console_manager",
        lambda: SimpleNamespace(console=SimpleNamespace(print=lambda *a, **k: None)),
    )
    config_commands.generate_env_impl()
    text = (tmp_path / ".env.template").read_text()
    assert "# LOBSTER_BEDROCK_MODEL_ID=" in text
    for dead in (
        "LOBSTER_SUPERVISOR_MODEL",
        "LOBSTER_TRANSCRIPTOMICS_EXPERT_MODEL",
        "LOBSTER_METHOD_AGENT_MODEL",
        "LOBSTER_GENERAL_CONVERSATION_MODEL",
        "LOBSTER_GLOBAL_MODEL",
    ):
        assert dead not in text
