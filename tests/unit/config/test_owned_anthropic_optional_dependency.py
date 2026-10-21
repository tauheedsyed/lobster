"""The optional Messages factory fails safely outside its qualified SDK domain."""

import os
import subprocess  # nosec B404 # Trusted interpreter runs only fixed offline import fixtures.
import sys

import pytest


def _run_import_fixture(script):
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(
            ("AWS_", "ANTHROPIC_", "OPENAI_", "LANGCHAIN_API", "LANGSMITH_API")
        )
    }
    environment.update(
        AWS_EC2_METADATA_DISABLED="true",
        LANGSMITH_TRACING="false",
        OTEL_SDK_DISABLED="true",
    )
    result = subprocess.run(  # nosec B603 # Fixed interpreter/fixture argv, no user input or shell.
        [sys.executable, "-I", "-c", script],
        env=environment,
        text=True,
        capture_output=True,
        timeout=45,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_missing_httpx2_is_a_clear_typed_context_refusal_in_fresh_process():
    script = r"""
import sys
from lobster.core.exceptions import InferenceRefused
assert "httpx2" not in sys.modules
class MissingTransport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "httpx2":
            raise ModuleNotFoundError("Synthetic missing optional transport", name="httpx2")
sys.meta_path.insert(0, MissingTransport())
try:
    from lobster.config.providers.owned_anthropic_factory import OwnedAnthropicFactory
except InferenceRefused as refusal:
    assert refusal.category == "context" and refusal.retryable is False
    assert "lobster-ai[anthropic]" in "\n".join(refusal.__notes__)
    assert "Synthetic missing" not in str(refusal)
    assert refusal.__suppress_context__
    print("typed optional-extra refusal")
else:
    raise AssertionError("Missing optional transport did not refuse")
"""
    assert "typed optional-extra refusal" in _run_import_fixture(script)


@pytest.mark.parametrize(
    "sdk_version", ["0.79.0", "1.6.9", "1.8.0", "2.0.0", "1.7.0rc1", "invalid"]
)
def test_unqualified_sdk_refuses_before_importing_native_consumer(sdk_version):
    script = r"""
import sys
import importlib.metadata
from lobster.core.exceptions import InferenceRefused
original_version = importlib.metadata.version
importlib.metadata.version = lambda name: sys.argv[1] if name == "anthropic" else original_version(name)
class NativeConsumerGuard:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "lobster.config.providers.controlled_anthropic_client":
            raise AssertionError("Unqualified SDK reached native consumer import")
sys.meta_path.insert(0, NativeConsumerGuard())
try:
    from lobster.config.providers.owned_anthropic_factory import OwnedAnthropicFactory
except InferenceRefused as refusal:
    assert refusal.category == "context" and refusal.retryable is False
    notes = "\n".join(refusal.__notes__)
    assert "anthropic>=1.7.0,<1.8" in notes
    assert "lobster-ai[anthropic]" in notes
    assert refusal.__suppress_context__
    print("typed SDK qualification refusal")
else:
    raise AssertionError("Unqualified SDK did not refuse")
"""
    # The version is a bounded fixed parametrized fixture, not executable input.
    script = "import sys; sys.argv = ['fixture', " + repr(sdk_version) + "]\n" + script
    assert "typed SDK qualification refusal" in _run_import_fixture(script)


@pytest.mark.parametrize("sdk_version", ["1.7.0", "1.7.1", "1.7.99"])
def test_patch_domain_accepts_import_without_claiming_patch_runtime_qualification(
    sdk_version,
):
    script = r"""
import sys
import importlib.metadata
original_version = importlib.metadata.version
importlib.metadata.version = lambda name: sys.argv[1] if name == "anthropic" else original_version(name)
from lobster.config.providers.owned_anthropic_factory import OwnedAnthropicFactory
print("accepted patch-domain import")
"""
    script = "import sys; sys.argv = ['fixture', " + repr(sdk_version) + "]\n" + script
    assert "accepted patch-domain import" in _run_import_fixture(script)
