"""Dual-mode async/sync compatibility for LangChain tools.

When a @tool is defined as `async def`, LangChain creates a StructuredTool
with `func=None` and `coroutine=<the async fn>`. This means `.invoke()` raises
`NotImplementedError: StructuredTool does not support sync invocation`.

Cloud uses `graph.astream()` which calls `tool.ainvoke()` — works fine.
CLI uses `graph.stream()` which calls `tool.invoke()` — breaks.

`enable_sync_fallback(tool)` patches `tool.func` with a sync wrapper:
- No loop running (CLI): run on one persistent background event loop. A fresh
  loop per call (`asyncio.run`) breaks SDK clients that cache an async HTTP
  client bound to the first loop: the second call failed with
  "Event loop is closed".
- Loop already running in this thread: offload `asyncio.run` to a private
  thread, as before. Never block on the shared loop from a running loop, which
  could deadlock nested calls.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import threading
from typing import Any, Optional

from langchain_core.tools import BaseTool

_loop: Optional[asyncio.AbstractEventLoop] = None
_loop_lock = threading.Lock()


def _reset_after_fork() -> None:
    # A forked child inherits the loop object but not its thread.
    global _loop, _loop_lock
    _loop = None
    _loop_lock = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)


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
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        # A loop is running in this thread: blocking it on the shared loop could
        # deadlock nested calls, so keep the previous private-thread behaviour.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    # Caller context variables propagate: the task is created in a copy of them.
    future = asyncio.run_coroutine_threadsafe(coro, _background_loop())
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
