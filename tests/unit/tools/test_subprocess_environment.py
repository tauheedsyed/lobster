"""Real tool subprocesses receive the established minimal environment policy."""

import json
import os
from unittest.mock import Mock
from uuid import uuid4

import pytest


def _original_custom_environment():
    """Snapshot of the pre-extraction custom-code child environment contract."""
    return {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "USER": os.environ.get("USER", ""),
        "TMPDIR": os.environ.get(
            "TMPDIR", "/tmp"
        ),  # nosec B108 # Environment fallback only, not file creation.
        "LANG": os.environ.get("LANG", "en_US.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", ""),
        "PYTHONPATH": "",
        "PYTHONHOME": os.environ.get("PYTHONHOME", ""),
    }


@pytest.mark.asyncio
async def test_real_shell_does_not_return_parent_credentials(tmp_path, monkeypatch):
    from lobster.tools.filesystem_tools import create_filesystem_tools

    names = (
        "OMICS_OS_API_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "FOO_API_KEY",
    )
    canaries = [uuid4().hex for _ in names]
    for name, value in zip(names, canaries):
        monkeypatch.setenv(name, value)
    tools = {tool.name: tool for tool in create_filesystem_tools(tmp_path)}
    result = await tools["shell_execute"].ainvoke(
        {
            "command": 'printf \'keys:%s|%s|%s|%s:end\\n\' "$OMICS_OS_API_KEY" "$AWS_SECRET_ACCESS_KEY" "$AWS_SESSION_TOKEN" "$FOO_API_KEY"'
        }
    )
    assert result == "keys:|||:end\n"
    assert all(value not in result for value in canaries)


@pytest.mark.asyncio
async def test_real_shell_preserves_path_lookup(tmp_path, monkeypatch):
    from lobster.tools.filesystem_tools import create_filesystem_tools

    bindir = tmp_path / "bin"
    bindir.mkdir()
    probe = bindir / "child-env-probe"
    probe.write_text("#!/bin/sh\nprintf 'path-works\\n'\n")
    probe.chmod(0o700)
    monkeypatch.setenv("PATH", str(bindir))
    tools = {tool.name: tool for tool in create_filesystem_tools(tmp_path)}
    assert (
        await tools["shell_execute"].ainvoke({"command": "child-env-probe"})
        == "path-works\n"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("home", [None, "", "synthetic-home"])
async def test_real_shell_preserves_home_fallback(tmp_path, monkeypatch, home):
    from lobster.tools.filesystem_tools import create_filesystem_tools

    if home is None:
        monkeypatch.delenv("HOME", raising=False)
    else:
        monkeypatch.setenv("HOME", home)
    tools = {tool.name: tool for tool in create_filesystem_tools(tmp_path)}
    result = await tools["shell_execute"].ainvoke({"command": "printf '%s' \"$HOME\""})
    assert result == (
        "/tmp"
        if home is None
        else home or "(no output)"  # nosec B108 # Expected legacy environment fallback.
    )


@pytest.mark.parametrize(
    "environment",
    [
        {},
        {
            name: ""
            for name in (
                "PATH",
                "HOME",
                "USER",
                "TMPDIR",
                "LANG",
                "LC_ALL",
                "PYTHONPATH",
                "PYTHONHOME",
            )
        },
        {
            name: "fixture-" + name
            for name in (
                "PATH",
                "HOME",
                "USER",
                "TMPDIR",
                "LANG",
                "LC_ALL",
                "PYTHONPATH",
                "PYTHONHOME",
                "FOO_API_KEY",
                "OTHER",
            )
        },
    ],
)
def test_helper_is_byte_identical_to_original_custom_environment(
    environment, monkeypatch
):
    from lobster.utils.subprocess_env import minimal_child_env

    monkeypatch.setattr(os, "environ", environment)
    assert json.dumps(minimal_child_env()) == json.dumps(_original_custom_environment())
    assert "FOO_API_KEY" not in minimal_child_env()
    assert minimal_child_env()["PYTHONPATH"] == ""


def test_real_custom_code_receives_unchanged_environment(tmp_path, monkeypatch):
    from lobster.core.data_manager_v2 import DataManagerV2
    from lobster.services.execution import custom_code_execution_service as module

    expected = _original_custom_environment()
    run = module.subprocess.run
    captured = []

    def observe(*args, **kwargs):
        captured.append(kwargs["env"].copy())
        return run(*args, **kwargs)

    monkeypatch.setattr(module.subprocess, "run", observe)
    data_manager = Mock(spec=DataManagerV2)
    data_manager.workspace_path = tmp_path
    data_manager.data_dir = tmp_path / "data"
    data_manager.list_modalities.return_value = []
    data_manager.locality_provider = Mock()
    data_manager.locality_provider.list_available.return_value = []
    service = module.CustomCodeExecutionService(data_manager)
    result, stats, _ = service.execute("result = 42", persist=False)
    assert result == 42 and stats["success"] is True
    assert len(captured) == 1
    assert json.dumps(captured[0]) == json.dumps(expected)
