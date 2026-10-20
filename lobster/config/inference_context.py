"""Explicit, immutable inference bindings supplied by an application host.

The engine does not resolve credentials or access policy for supplied bindings.
A factory owns its native request clients and their per-request authorization.
"""

import inspect
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol
from uuid import UUID

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


@dataclass(frozen=True, slots=True)
class InferenceModelSelection:
    """An explicit native model pair already canonicalized by the trusted host.

    Selection is an intent check, never a replacement for a resolved context.
    Absence means inherit the supplied context, not local/provider defaults.
    """

    provider: str
    model_id: str

    def __post_init__(self):
        for value in (self.provider, self.model_id):
            if (
                type(value) is not str
                or not value
                or value != value.strip()
                or len(value) > 512
            ):
                raise InferenceRefused("context")


def resolve_inference_model(
    context: InferenceContext, selection: InferenceModelSelection | None = None
) -> tuple[str, str]:
    """Inherit or verify an exact explicit pair without ambient resolution."""
    if not isinstance(context, InferenceContext):
        raise InferenceRefused("context")
    pair = context.provider, context.model_id
    if selection is not None and (
        not isinstance(selection, InferenceModelSelection)
        or (selection.provider, selection.model_id) != pair
    ):
        raise InferenceRefused("context")
    return pair


@dataclass(frozen=True, slots=True)
class RequestDescriptor:
    """Nonsecret identity and conservative bounds of one physical request."""

    provider: str
    model_id: str
    component: str
    binding_id: str
    binding_version: str
    invocation_id: str
    attempt_id: str
    max_output_tokens: int
    input_token_upper_bound: int
    deadline_at: float

    def __post_init__(self):
        for name in (
            "provider",
            "model_id",
            "component",
            "binding_id",
            "binding_version",
        ):
            value = getattr(self, name)
            if (
                type(value) is not str
                or not value
                or value != value.strip()
                or len(value) > 512
            ):
                raise InferenceRefused("context")
        for value in (self.invocation_id, self.attempt_id):
            try:
                if type(value) is not str or str(UUID(value)) != value:
                    raise ValueError
            except (ValueError, TypeError, AttributeError):
                raise InferenceRefused("context") from None
        for value in (self.max_output_tokens, self.input_token_upper_bound):
            if type(value) is not int or value <= 0:
                raise InferenceRefused("context")
        if (
            type(self.deadline_at) is not float
            or not math.isfinite(self.deadline_at)
            or not 0 < self.deadline_at < 253402300800
        ):
            raise InferenceRefused("context")


_RAW_USAGE_FIELDS = frozenset(
    {
        "inputTokens",
        "outputTokens",
        "totalTokens",
        "cacheReadInputTokens",
        "cacheWriteInputTokens",
        "input_tokens",
        "output_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
        "ephemeral_5m_input_tokens",
        "ephemeral_1h_input_tokens",
        "cache_ttl",
    }
)


@dataclass(frozen=True, slots=True)
class UsageOutcome:
    """Usage evidence only, never provider text, prompts, keys or receipts."""

    raw_usage: tuple = ()
    normalized_input_tokens: int | None = None
    normalized_output_tokens: int | None = None
    finish_reason: str | None = None
    completeness: str = "unknown"
    request_issued: bool = False
    outcome: str = "unknown"
    actual_model_id: str | None = None

    def __post_init__(self):
        if (
            type(self.completeness) is not str
            or self.completeness not in {"complete", "partial", "unknown"}
            or type(self.outcome) is not str
            or self.outcome
            not in {"completed", "refused", "cancelled", "failed", "unknown"}
            or type(self.request_issued) is not bool
        ):
            raise InferenceRefused("accounting")
        for value in (self.normalized_input_tokens, self.normalized_output_tokens):
            if value is not None and (type(value) is not int or value < 0):
                raise InferenceRefused("accounting")
        if type(self.raw_usage) is not tuple:
            raise InferenceRefused("accounting")
        names = set()
        for pair in self.raw_usage:
            if type(pair) is not tuple or len(pair) != 2:
                raise InferenceRefused("accounting")
            name, value = pair
            if type(name) is not str or name not in _RAW_USAGE_FIELDS or name in names:
                raise InferenceRefused("accounting")
            names.add(name)
            if name == "cache_ttl":
                if type(value) is not str or value not in {"5m", "1h"}:
                    raise InferenceRefused("accounting")
            elif type(value) is not int or value < 0:
                raise InferenceRefused("accounting")
        for value in (self.finish_reason, self.actual_model_id):
            if value is not None and (
                type(value) is not str or not value or len(value) > 512
            ):
                raise InferenceRefused("accounting")


def _freeze_native_value(value, depth=0):
    if depth > 64:
        raise InferenceRefused("context")
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise InferenceRefused("context")
        return MappingProxyType(
            {
                key: _freeze_native_value(child, depth + 1)
                for key, child in value.items()
            }
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_native_value(child, depth + 1) for child in value)
    if value is None or type(value) in (str, bytes, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise InferenceRefused("context")


@dataclass(frozen=True, slots=True, repr=False)
class NativeRequestView:
    """Ephemeral native parameters for validation; never logged or serialized."""

    operation: str
    endpoint: str
    parameters: Mapping[str, Any]

    def __post_init__(self):
        if (
            type(self.operation) is not str
            or not self.operation
            or type(self.endpoint) is not str
            or not self.endpoint
            or not isinstance(self.parameters, Mapping)
        ):
            raise InferenceRefused("context")
        object.__setattr__(self, "parameters", _freeze_native_value(self.parameters))

    def __repr__(self):
        return "NativeRequestView(<ephemeral>)"

    def __reduce__(self):
        raise TypeError("Native request views cannot be serialized")


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
