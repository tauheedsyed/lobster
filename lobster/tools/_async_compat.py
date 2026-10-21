"""Dual-mode async/sync compatibility for LangChain tools.

When a @tool is defined as `async def`, LangChain creates a StructuredTool
with `func=None` and `coroutine=<the async fn>`. This means `.invoke()` raises
`NotImplementedError: StructuredTool does not support sync invocation`.

Cloud uses `graph.astream()` which calls `tool.ainvoke()` — works fine.
CLI uses `graph.stream()` which calls `tool.invoke()` — breaks.

`enable_sync_fallback(tool)` patches `tool.func` with a sync wrapper that runs
the coroutine on one persistent background event loop. A fresh loop per call
(`asyncio.run`) breaks SDK clients that cache an async HTTP client bound to the
first loop: the second call fails with "Event loop is closed". If the wrapper
is called from that background loop itself, it falls back to a private
thread so it cannot deadlock.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from typing import Any, Optional

from langchain_core.tools import BaseTool

_loop: Optional[asyncio.AbstractEventLoop] = None
_loop_lock = threading.Lock()


def _background_loop() -> asyncio.AbstractEventLoop:
    """Return the shared background event loop, starting it once."""
    global _loop
    with _loop_lock:
        if _loop is None or _loop.is_closed():
            loop = asyncio.new_event_loop()
            thread = threading.Thread(
                target=loop.run_forever,
                name="lobster-sync-tool-loop",
                daemon=True,
            )
            thread.start()
            _loop = loop
        return _loop


def run_coroutine_sync(coro: Any) -> Any:
    """Run a coroutine to completion from synchronous code."""
    loop = _background_loop()
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is loop:
        # Called from the shared loop's own thread: blocking on it would deadlock.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return future.result()
    except BaseException:
        future.cancel()
        raise


def enable_sync_fallback(tool: BaseTool) -> BaseTool:
    """Patch an async-only tool to also support sync .invoke().

    Idempotent — safe to call multiple times on the same tool.
    """
    if tool.func is not None:
        return tool

    coroutine = tool.coroutine
    if coroutine is None:
        return tool

    def _sync_fallback(*args: Any, **kwargs: Any) -> Any:
        return run_coroutine_sync(coroutine(*args, **kwargs))

    tool.func = _sync_fallback
    return tool
