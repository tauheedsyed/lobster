"""Generic explicit-endpoint Messages factory with host-owned request authority."""

import asyncio
from importlib.metadata import version
from threading import RLock
from urllib.parse import urlsplit

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion

from lobster.config.inference_context import INFERENCE_BINDING_VERSION, InferenceContext
from lobster.config.providers.request_auth import canonical_refusal, refusal_spec
from lobster.core.exceptions import InferenceRefused

# Qualification baseline: Cloud constraints-prod-*.txt declares
# anthropic==1.7.0 and httpx2==2.13.0. Patch updates retain this minor contract.
QUALIFIED_ANTHROPIC_SDK = ">=1.7.0,<1.8"

# Check the optional transport and SDK domain before importing SDK consumers.
try:
    import httpx2

    if not SpecifierSet(QUALIFIED_ANTHROPIC_SDK).contains(version("anthropic")):
        raise ImportError("Unqualified Anthropic SDK version")
    from lobster.config.providers.controlled_anthropic_client import (
        ControlledAnthropicClient,
        ControlledChatAnthropic,
    )
except (ImportError, InvalidVersion):
    refusal = InferenceRefused("context")
    refusal.add_note(
        "Owned Anthropic inference requires lobster-ai[anthropic] "
        f"with anthropic{QUALIFIED_ANTHROPIC_SDK} and httpx2>=2.0.0,<3."
    )
    raise refusal from None


def _options(value):
    if type(value) is tuple:
        if value and all(
            type(pair) is tuple and len(pair) == 2 and type(pair[0]) is str
            for pair in value
        ):
            return {name: _options(child) for name, child in value}
        return [_options(child) for child in value]
    return value


class OwnedAnthropicFactory:
    """Create real bound models, never resolve ambient credentials or policy.

    Every component owns separate sync/async HTTP clients. The host supplies
    request validators/descriptors, transport-only or accounting hooks, renewing
    authentication, safe native-error decoding and nonsecret request headers.
    No application identity, bearer value or endpoint enters InferenceContext.
    Call ``await aclose()`` before replacing this factory or rebinding a graph.
    """

    capability_version = INFERENCE_BINDING_VERSION

    def __init__(
        self,
        *,
        context,
        base_url,
        request_auth,
        authorizer_factory,
        validate_request,
        describe_request,
        error_decoder,
        request_headers,
        transport_factory=None,
    ):
        try:
            url = urlsplit(base_url)
            valid_url = (
                url.scheme == "https"
                and url.hostname
                and not url.username
                and not url.password
                and not url.query
                and not url.fragment
                and "%" not in url.path
                and not any(part in {".", ".."} for part in url.path.split("/"))
                and url.port in {None, 443}
            )
        except (TypeError, ValueError):
            valid_url = False
        if (
            not isinstance(context, InferenceContext)
            or context.provider != "anthropic"
            or not valid_url
            or any(
                not callable(callback)
                for callback in (
                    authorizer_factory,
                    validate_request,
                    describe_request,
                    error_decoder,
                    request_headers,
                )
            )
            or not callable(getattr(request_auth, "get", None))
            or not callable(getattr(request_auth, "aget", None))
            or not callable(getattr(request_auth, "stop", None))
            or not callable(getattr(request_auth, "aclose", None))
        ):
            raise InferenceRefused("context")
        self._context, self._base_url = context, base_url.rstrip("/")
        self._auth, self._authorizer_factory = request_auth, authorizer_factory
        self._validate, self._describe = validate_request, describe_request
        self._decode, self._headers = error_decoder, request_headers
        self._transports = transport_factory
        self._owners = []
        self._http_clients = []
        self._lock = RLock()
        self._closed = False
        self._refusal = None
        self._closing = None

    def __repr__(self):
        return "OwnedAnthropicFactory(<owned>)"

    def __reduce__(self):
        raise TypeError("Owned inference factories cannot be serialized")

    def _validate_model_request(self, component, view):
        with self._lock:
            if self._refusal is not None:
                raise InferenceRefused(self._refusal[0], retryable=self._refusal[1])
            if self._closed:
                raise InferenceRefused("context")
        return self._validate(component, view)

    def _before_issue(self, mark):
        with self._lock:
            if self._refusal is not None:
                raise InferenceRefused(self._refusal[0], retryable=self._refusal[1])
            if self._closed:
                raise InferenceRefused("context")
            mark()

    def _decode_error(self, view):
        facts = None
        try:
            decoded = self._decode(view)
            if isinstance(decoded, InferenceRefused):
                facts = canonical_refusal(
                    decoded.category, decoded.retryable, "provider"
                )
        except Exception as error:
            facts = refusal_spec(error, "provider")
        if facts is not None:
            if facts[0] != "provider" or not facts[1]:
                with self._lock:
                    self._refusal = facts
            return InferenceRefused(facts[0], retryable=facts[1])
        return None

    def create(self, *, component, context):
        with self._lock:
            if self._refusal is not None:
                raise InferenceRefused(self._refusal[0], retryable=self._refusal[1])
            if (
                self._closed
                or not isinstance(context, InferenceContext)
                or context != self._context
                or type(component) is not str
                or not component.strip()
                or len(component) > 256
            ):
                raise InferenceRefused("context")
            options = {
                name: _options(value) for name, value in context.generation_options
            }
            if "additional_model_request_fields" in options:
                raise InferenceRefused("context")
            authorizer = self._authorizer_factory(component, context)

            sync_transport, async_transport = (
                self._transports(component) if self._transports else (None, None)
            )
            sync = httpx2.Client(
                transport=sync_transport,
                trust_env=False,
                follow_redirects=False,
                timeout=httpx2.Timeout(60.0, connect=3.0, write=10.0, pool=10.0),
            )
            self._http_clients.append(sync)
            asynchronous = httpx2.AsyncClient(
                transport=async_transport,
                trust_env=False,
                follow_redirects=False,
                timeout=httpx2.Timeout(60.0, connect=3.0, write=10.0, pool=10.0),
            )
            self._http_clients.append(asynchronous)
            owner = ControlledAnthropicClient(
                context=context,
                component=component,
                authorizer=authorizer,
                validate_request=lambda view: self._validate_model_request(
                    component, view
                ),
                describe_request=lambda view: self._describe(component, view),
                api_key="explicit-owned-request-auth",
                base_url=self._base_url,
                http_client=sync,
                http_async_client=asynchronous,
                request_auth=self._auth,
                error_decoder=self._decode_error,
                request_headers=self._headers,
                before_issue=self._before_issue,
            )
            self._owners.append(owner)
            return ControlledChatAnthropic(
                model=context.model_id,
                api_key="explicit-owned-request-auth",
                base_url=self._base_url,
                max_tokens=context.max_output_tokens,
                max_retries=0,
                cache=False,
                **options,
            ).attach_request_owner(owner)

    async def _dispose(self):
        raw_results = await asyncio.gather(
            self._auth.aclose(),
            *(owner.drain() for owner in self._owners),
            return_exceptions=True,
        )
        results = [
            refusal_spec(item, "context")
            for item in raw_results
            if isinstance(item, BaseException)
        ]
        raw_results = None
        for owner in self._owners:
            try:
                owner.close()
            except BaseException as error:
                results.append(refusal_spec(error, "context"))
            try:
                await owner.aclose()
            except BaseException as error:
                results.append(refusal_spec(error, "context"))
        for client in self._http_clients:
            try:
                if hasattr(client, "aclose"):
                    await client.aclose()
                else:
                    client.close()
            except BaseException as error:
                results.append(refusal_spec(error, "context"))
        # A durable closing task carries primitives, never a failed SDK close's
        # exception/traceback. Public waiters construct their own safe refusal.
        return next(
            (item for item in results if item[0] == "accounting"),
            results[0] if results else None,
        )

    async def aclose(self):
        with self._lock:
            self._closed = True
            self._auth.stop()
            for owner in self._owners:
                owner.stop()
            if self._closing is None:
                self._closing = asyncio.create_task(self._dispose())
            task = self._closing
        cancelled = False
        while True:
            try:
                result = await asyncio.shield(task)
                break
            except asyncio.CancelledError:
                cancelled = True
                if task.done():
                    result = task.result()
                    break
        if result is not None:
            raise InferenceRefused(result[0], retryable=result[1]) from None
        if cancelled:
            raise asyncio.CancelledError
