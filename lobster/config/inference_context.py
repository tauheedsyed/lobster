"""Explicit, immutable inference bindings supplied by an application host.

The engine does not resolve credentials or access policy for supplied bindings.
A factory owns its native request clients and their per-request authorization.
"""

import inspect
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from lobster.core.exceptions import InferenceRefused, reraise_inference_refusal

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

INFERENCE_BINDING_VERSION = "inference-binding-v1"

_GENERATION_FIELDS = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "thinking",
        "stop_sequences",
        "stop",
        "reasoning_effort",
        "betas",
        "output_config",
        "additional_model_request_fields",
    }
)
_BINDING_FIELDS = frozenset(
    {
        "api_key",
        "access_token",
        "refresh_token",
        "auth_token",
        "token",
        "secret",
        "password",
        "credentials",
        "credential",
        "authorization",
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_session_token",
        "credentials_profile_name",
        "endpoint",
        "endpoint_url",
        "base_url",
        "anthropic_api_url",
        "anthropic_proxy",
        "client",
        "bedrock_client",
        "http_client",
        "http_async_client",
        "headers",
        "default_headers",
    }
)


def _immutable_generation_value(value: Any, depth: int = 0) -> None:
    if depth > 32:
        raise InferenceRefused("context")
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is tuple:
        if (
            len(value) == 2
            and type(value[0]) is str
            and value[0].lower() in _BINDING_FIELDS
        ):
            raise InferenceRefused("context")
        for item in value:
            # Nested option pairs cannot smuggle authentication or transport.
            if (
                type(item) is tuple
                and len(item) == 2
                and type(item[0]) is str
                and item[0].lower() in _BINDING_FIELDS
            ):
                raise InferenceRefused("context")
            _immutable_generation_value(item, depth + 1)
        return
    raise InferenceRefused("context")


@dataclass(frozen=True, slots=True)
class InferenceContext:
    """Nonsecret generation identity; credentials remain inside the host factory."""

    provider: str
    model_id: str
    binding_id: str
    binding_version: str
    context_window: int
    max_output_tokens: int
    generation_options: tuple = ()

    def __post_init__(self) -> None:
        for value in (
            self.provider,
            self.model_id,
            self.binding_id,
            self.binding_version,
        ):
            if (
                type(value) is not str
                or not value
                or value != value.strip()
                or len(value) > 512
            ):
                raise InferenceRefused("context")
        for bound in (self.context_window, self.max_output_tokens):
            if type(bound) is not int or bound <= 0:
                raise InferenceRefused("context")
        if type(self.generation_options) is not tuple:
            raise InferenceRefused("context")
        names = set()
        for pair in self.generation_options:
            if type(pair) is not tuple or len(pair) != 2:
                raise InferenceRefused("context")
            name, value = pair
            if type(name) is not str or name not in _GENERATION_FIELDS or name in names:
                raise InferenceRefused("context")
            names.add(name)
            _immutable_generation_value(value)


class BoundModelFactory(Protocol):
    """Produces real models with controlled request clients and no response cache."""

    capability_version: str

    def create(
        self, *, component: str, context: InferenceContext
    ) -> "BaseChatModel": ...


class InferenceAuthorizer(Protocol):
    """Native request hooks; descriptor, permits and receipts are host-owned."""

    def before_request(self, descriptor: Any) -> Any: ...
    def after_request(self, permit: Any, outcome: Any) -> Any: ...
    async def abefore_request(self, descriptor: Any) -> Any: ...
    async def aafter_request(self, permit: Any, outcome: Any) -> Any: ...


def validate_inference_binding(
    inference_context: InferenceContext | None,
    model_factory: BoundModelFactory | None,
) -> bool:
    """Return whether strict binding is active, refusing partial/unsupported pairs."""
    if inference_context is None and model_factory is None:
        return False
    if not isinstance(inference_context, InferenceContext) or model_factory is None:
        raise InferenceRefused("context")
    try:
        supported = (
            getattr(model_factory, "capability_version", None)
            == INFERENCE_BINDING_VERSION
        )
        create = getattr(model_factory, "create", None)
    except Exception as error:
        reraise_inference_refusal(error)
        raise InferenceRefused("context") from None
    if not supported or not callable(create) or inspect.iscoroutinefunction(create):
        raise InferenceRefused("context")
    return True


def _reject_credential_config(value: Any, depth: int = 0) -> None:
    if depth > 32:
        raise InferenceRefused("context")
    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str) and key.lower() in _BINDING_FIELDS:
                raise InferenceRefused("context")
            _reject_credential_config(child, depth + 1)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _reject_credential_config(child, depth + 1)


def create_bound_model(
    *,
    component: str,
    inference_context: InferenceContext,
    model_factory: BoundModelFactory,
    model_config: Mapping[str, Any] | None = None,
    provider_override: str | None = None,
    model_override: str | None = None,
) -> "BaseChatModel":
    """Use only the supplied binding; never consult registry, files or environment.

    Legacy generation defaults are subordinate to the immutable context. Explicit
    identity disagreements and credential/transport arguments are refused.
    """
    validate_inference_binding(inference_context, model_factory)
    if type(component) is not str or not component or len(component) > 256:
        raise InferenceRefused("context")
    if (
        provider_override is not None
        and provider_override != inference_context.provider
    ):
        raise InferenceRefused("context")
    if model_override is not None and model_override != inference_context.model_id:
        raise InferenceRefused("context")
    if model_config is not None:
        if not isinstance(model_config, Mapping):
            raise InferenceRefused("context")
        _reject_credential_config(model_config)
        if (
            "model_id" in model_config
            and model_config["model_id"] != inference_context.model_id
        ):
            raise InferenceRefused("context")
    from langchain_core.language_models import BaseChatModel

    try:
        model = model_factory.create(component=component, context=inference_context)
    except Exception as error:
        reraise_inference_refusal(error)
        raise InferenceRefused("provider") from None
    if not isinstance(model, BaseChatModel) or model.cache is not False:
        raise InferenceRefused("context")
    return model
