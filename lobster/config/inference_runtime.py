"""Runtime-only inference ownership for one data manager and its lazy helpers.

Bindings never enter workspace snapshots. Replacement is an explicit graph/run
boundary, not a way to change an active request or resume an interrupted run.
"""

from contextlib import contextmanager, nullcontext
from threading import RLock
from weakref import WeakSet

from langchain_core.runnables import Runnable

from lobster.config.inference_context import (
    BoundModelFactory,
    InferenceContext,
    validate_inference_binding,
)
from lobster.core.exceptions import InferenceRefused, reraise_inference_refusal


class InferenceBinding:
    """An ephemeral lease; identity is immutable while the lease is usable."""

    def __init__(self, owner, context, factory):
        self._owner = owner
        self._context = context
        self._factory = factory
        self._lock = RLock()
        self._helpers = WeakSet()
        self._active = 0
        self._retired = False
        self._suspended = set()

    def __reduce__(self):
        raise TypeError("Runtime inference bindings cannot be serialized")

    def check(self):
        with self._lock:
            if self._retired:
                raise InferenceRefused("context")
            if (
                self._owner is not None
                and vars(self._owner).get("_inference_binding") is not self
            ):
                raise InferenceRefused("context")
            return self._context, self._factory

    def register_helper(self, helper):
        with self._lock:
            self.check()
            self._helpers.add(helper)

    @contextmanager
    def use(self):
        with self._lock:
            self.check()
            self._active += 1
        try:
            yield
        finally:
            with self._lock:
                self._active -= 1

    def suspend(self, suspended, thread_id):
        with self._lock:
            self.check()
            if suspended:
                self._suspended.add(thread_id)
            else:
                self._suspended.discard(thread_id)

    def retire(self):
        with self._lock:
            if self._active or self._suspended:
                raise InferenceRefused("context")
            self._retired = True
            for helper in self._helpers:
                helper._llm = None
            self._helpers.clear()
            self._context = None
            self._factory = None
            self._owner = None


def capture_inference_binding(
    data_manager,
    inference_context: InferenceContext | None = None,
    model_factory: BoundModelFactory | None = None,
    *,
    inherit: bool = True,
    replace: bool = False,
) -> InferenceBinding | None:
    """Capture a supplied pair, or the owner's pair for a legacy helper.

    Primary factories use ``inherit=False``: absent parameters must not create
    an ambient primary model alongside already bound helpers. Graph creation
    uses ``replace=True`` even for an identical identity, after prior work drains.
    """
    supplied = validate_inference_binding(inference_context, model_factory)
    if data_manager is None:
        return (
            InferenceBinding(None, inference_context, model_factory)
            if supplied
            else None
        )
    owner_vars = vars(data_manager)
    lock = owner_vars.setdefault("_inference_binding_lock", RLock())
    with lock:
        current = owner_vars.get("_inference_binding")
        if current is not None and not isinstance(current, InferenceBinding):
            raise InferenceRefused("context")
        if not supplied:
            if current is None:
                return None
            if not inherit or replace:
                raise InferenceRefused("context")
            current.check()
            return current
        if current is not None:
            if replace:
                current.retire()
            else:
                context, factory = current.check()
                if context != inference_context or factory is not model_factory:
                    raise InferenceRefused("context")
                return current
        binding = InferenceBinding(data_manager, inference_context, model_factory)
        owner_vars["_inference_binding"] = binding
        return binding


class InferenceBoundHelper:
    """Shared lease capture and stale-cache checks for lazy model helpers."""

    def _capture_binding(self, data_manager, inference_context, model_factory):
        self.data_manager = data_manager
        self._inference_binding = capture_inference_binding(
            data_manager, inference_context, model_factory
        )
        if self._inference_binding is not None:
            self._inference_binding.register_helper(self)

    def _check_binding(self):
        binding = self._inference_binding
        if binding is not None:
            return binding.check()
        if (
            self.data_manager is not None
            and vars(self.data_manager).get("_inference_binding") is not None
        ):
            raise InferenceRefused("context")
        return None, None

    def _inference_scope(self):
        self._check_binding()
        return (
            self._inference_binding.use()
            if self._inference_binding is not None
            else nullcontext()
        )


class InferenceBoundGraph(Runnable):
    """Delegate graph APIs while pinning the binding throughout consumption.

    A generator owns its lease until exhaustion or close. An actual checkpoint
    interrupt keeps replacement disabled until the original graph is resumed.
    """

    def __init__(self, graph, binding):
        self._graph = graph
        self._binding = binding

    def __getattr__(self, name):
        return getattr(self._graph, name)

    def get_input_schema(self, config=None):
        return self._graph.get_input_schema(config)

    def get_output_schema(self, config=None):
        return self._graph.get_output_schema(config)

    def get_graph(self, config=None):
        return self._graph.get_graph(config)

    @property
    def config_specs(self):
        return self._graph.config_specs

    @contextmanager
    def _scope(self, config):
        with self._binding.use():
            try:
                yield
            finally:
                if getattr(self._graph, "checkpointer", None) is not None and config:
                    thread_id = config.get("configurable", {}).get("thread_id")
                    try:
                        state = self._graph.get_state(config)
                    except Exception as error:
                        # Unknown checkpoint health must not unlock replacement.
                        self._binding.suspend(True, thread_id)
                        reraise_inference_refusal(error)
                        raise InferenceRefused("context") from None
                    self._binding.suspend(
                        any(task.interrupts for task in state.tasks), thread_id
                    )

    def invoke(self, input, config=None, **kwargs):
        with self._scope(config):
            return self._graph.invoke(input, config, **kwargs)

    async def ainvoke(self, input, config=None, **kwargs):
        with self._scope(config):
            return await self._graph.ainvoke(input, config, **kwargs)

    def stream(self, input, config=None, **kwargs):
        with self._scope(config):
            stream = self._graph.stream(input, config, **kwargs)
            try:
                yield from stream
            finally:
                stream.close()

    async def astream(self, input, config=None, **kwargs):
        with self._scope(config):
            stream = self._graph.astream(input, config, **kwargs)
            try:
                async for chunk in stream:
                    yield chunk
            finally:
                await stream.aclose()

    async def astream_events(self, input, config=None, **kwargs):
        with self._scope(config):
            stream = self._graph.astream_events(input, config, **kwargs)
            try:
                async for chunk in stream:
                    yield chunk
            finally:
                await stream.aclose()
