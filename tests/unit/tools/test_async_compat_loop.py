"""The sync fallback must reuse one event loop across calls.

SDK clients (e.g. the OpenAI async client) cache an HTTP client bound to the
loop of their first call. A fresh loop per call (``asyncio.run``) closes that
loop, so the second delegation failed with "Event loop is closed".
"""

import asyncio

from langchain_core.tools import tool

from lobster.tools._async_compat import enable_sync_fallback, run_coroutine_sync


def test_sync_fallback_reuses_one_loop():
    loops = []

    @tool
    async def probe(x: int) -> int:
        """Record the running loop."""
        loops.append(asyncio.get_running_loop())
        return x + 1

    enable_sync_fallback(probe)
    assert [probe.invoke({"x": i}) for i in range(3)] == [1, 2, 3]
    assert len({id(loop) for loop in loops}) == 1
    assert not loops[0].is_closed()


def test_loop_bound_resource_survives_repeated_calls():
    """Mimic an SDK client that caches a loop-bound object on first use."""
    cache = {}

    async def call():
        loop = asyncio.get_running_loop()
        bound = cache.setdefault("loop", loop)
        if bound is not loop or bound.is_closed():
            raise RuntimeError("Event loop is closed")
        await asyncio.sleep(0)
        return "ok"

    assert [run_coroutine_sync(call()) for _ in range(3)] == ["ok"] * 3


def test_called_from_running_loop_does_not_deadlock():
    async def inner():
        return "inner"

    async def outer():
        return run_coroutine_sync(inner())

    assert asyncio.run(outer()) == "inner"


def test_nested_call_on_shared_loop_does_not_deadlock():
    async def inner():
        return "inner"

    async def outer():
        return run_coroutine_sync(inner())

    assert run_coroutine_sync(outer()) == "inner"


def test_exceptions_propagate():
    async def boom():
        raise ValueError("bad")

    import pytest

    with pytest.raises(ValueError, match="bad"):
        run_coroutine_sync(boom())
