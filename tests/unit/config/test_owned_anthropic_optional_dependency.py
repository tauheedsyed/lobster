"""Optional native dependencies are validated at use, never during core imports."""

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


_FACTORY = r"""
from types import SimpleNamespace
from uuid import uuid4
from lobster.config.inference_context import INFERENCE_BINDING_VERSION, InferenceContext
from lobster.config.providers.owned_anthropic_factory import OwnedAnthropicFactory
context = InferenceContext(provider="anthropic", model_id="synthetic", binding_id=str(uuid4()),
    binding_version=INFERENCE_BINDING_VERSION, context_window=200000, max_output_tokens=8192)
def must_not_authorize(*args):
    raise AssertionError("Missing SDK reached authorization")
factory = OwnedAnthropicFactory(context=context, base_url="https://example.invalid",
    request_auth=SimpleNamespace(get=lambda: None, aget=lambda: None, stop=lambda: None, aclose=lambda: None),
    authorizer_factory=must_not_authorize, validate_request=lambda *args: None,
    describe_request=lambda *args: None, error_decoder=lambda *args: None, request_headers=lambda: {})
"""


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
"""
    script += _FACTORY + r"""
assert "httpx2" not in sys.modules
try:
    factory.create(component="helper", context=context)
except InferenceRefused as refusal:
    assert refusal.category == "context" and refusal.retryable is False
    assert "lobster-ai[anthropic]" in "\n".join(refusal.__notes__)
    assert "Synthetic missing" not in str(refusal)
    assert refusal.__suppress_context__
    print("typed optional-extra refusal")
else:
    raise AssertionError("Missing optional transport did not refuse on use")
"""
    assert "typed optional-extra refusal" in _run_import_fixture(script)


@pytest.mark.parametrize("sdk_version", ["0.79.0", "1.6.9", "1.7.0rc1", "invalid"])
def test_unqualified_sdk_refuses_before_importing_native_consumer(sdk_version):
    script = r"""
import sys, types, importlib.metadata
from lobster.core.exceptions import InferenceRefused
original_version = importlib.metadata.version
importlib.metadata.version = lambda name: sys.argv[1] if name == "anthropic" else original_version(name)
# Make transport availability independent of optional packages in the test env.
sys.modules["httpx2"] = types.ModuleType("httpx2")
class NativeConsumerGuard:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "lobster.config.providers.controlled_anthropic_client":
            raise AssertionError("Unqualified SDK reached native consumer import")
sys.meta_path.insert(0, NativeConsumerGuard())
"""
    script += _FACTORY + r"""
try:
    factory.create(component="helper", context=context)
except InferenceRefused as refusal:
    assert refusal.category == "context" and refusal.retryable is False
    assert "anthropic>=1.7.0" in "\n".join(refusal.__notes__)
    assert refusal.__suppress_context__
    print("typed SDK qualification refusal")
else:
    raise AssertionError("Unqualified SDK did not refuse on use")
"""
    script = "import sys; sys.argv = ['fixture', " + repr(sdk_version) + "]\n" + script
    assert "typed SDK qualification refusal" in _run_import_fixture(script)


@pytest.mark.parametrize(
    "sdk_version", ["1.7.0", "1.7.1", "1.7.99", "1.8.0", "1.17.0", "2.0.0"]
)
def test_patch_domain_accepts_import_without_claiming_patch_runtime_qualification(
    sdk_version,
):
    script = r"""
import sys, types, importlib.metadata
original_version = importlib.metadata.version
importlib.metadata.version = lambda name: sys.argv[1] if name == "anthropic" else original_version(name)
transport = types.ModuleType("httpx2")
sys.modules["httpx2"] = transport
consumer = types.ModuleType("lobster.config.providers.controlled_anthropic_client")
consumer.ControlledAnthropicClient = object
consumer.ControlledChatAnthropic = object
sys.modules[consumer.__name__] = consumer
from lobster.config.providers.owned_anthropic_factory import _owned_sdk
assert _owned_sdk() == (transport, object, object)
print("accepted patch-domain import")
"""
    script = "import sys; sys.argv = ['fixture', " + repr(sdk_version) + "]\n" + script
    assert "accepted patch-domain import" in _run_import_fixture(script)


@pytest.mark.parametrize("entrypoint", ["imports", "help"])
def test_core_entrypoints_never_import_owned_factory(entrypoint):
    script = r"""
import sys
name = "lobster.config.providers.owned_anthropic_factory"
class ForbidOwnedFactory:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == name:
            raise AssertionError("Core entrypoint eagerly imported optional owned factory")
sys.meta_path.insert(0, ForbidOwnedFactory())
if sys.argv[1] == "imports":
    import lobster
    import lobster.agents.supervisor
    import lobster.agents.graph
    import lobster.core.client
else:
    sys.argv = ["lobster", "--help"]
    from lobster.cli import app
    try:
        app()
    except SystemExit as exit:
        assert exit.code in (None, 0)
assert name not in sys.modules
print("core import boundary preserved")
"""
    script = "import sys; sys.argv = ['fixture', " + repr(entrypoint) + "]\n" + script
    assert "core import boundary preserved" in _run_import_fixture(script)
