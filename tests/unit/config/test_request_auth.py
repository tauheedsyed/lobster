"""No-network host callback auth renewal races, refusals and cancellation."""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from uuid import uuid4

import pytest

from lobster.config.providers.request_auth import ExpiringRequestAuth, RequestAuth
from lobster.core.exceptions import InferenceRefused

pytestmark = pytest.mark.no_auto_config


def supplier(*, ttl=300.0, refresh=None, arefresh=None, deadline=None):
    clock = [time.time()]
    values = [uuid4().hex, uuid4().hex]
    calls = []

    def renew(old):
        calls.append(old.generation)
        return RequestAuth(values[1], expires_at=clock[0] + 300.0, generation=2)

    async def arenew(old):
        return renew(old)

    auth = ExpiringRequestAuth(
        initial=RequestAuth(values[0], expires_at=clock[0] + ttl),
        refresh=refresh or renew,
        arefresh=arefresh or arenew,
        clock=lambda: clock[0],
        renewal_deadline=deadline,
    )
    return auth, clock, calls


@pytest.mark.parametrize(
    "value", ["", "has space", "has\nnewline", "unicode\u03bb", "\x7f", "x" * 4097]
)
def test_auth_header_shape_refuses_without_echo(value):
    with pytest.raises(InferenceRefused) as error:
        RequestAuth(value, expires_at=time.time() + 300.0)
    assert error.value.category == "auth"
    assert str(error.value) == "Inference refused (auth)."


@pytest.mark.parametrize("expiry", [0.0, -1.0, float("inf"), float("nan"), True, 300])
def test_auth_expiry_is_finite_strict_epoch(expiry):
    with pytest.raises(InferenceRefused):
        RequestAuth(uuid4().hex, expires_at=expiry)


@pytest.mark.asyncio
async def test_expired_grant_does_not_invoke_refresh():
    auth, clock, calls = supplier()
    clock[0] += 301.0
    with pytest.raises(InferenceRefused) as caught:
        await auth.aget()
    assert caught.value.category == "auth"
    assert not calls
    await auth.aclose()


@pytest.mark.asyncio
async def test_sync_thread_refresh_shares_one_result_with_async_waiter():
    entered, release = Event(), Event()
    clock = [time.time()]
    current = RequestAuth(uuid4().hex, expires_at=clock[0] + 100.0)
    replacement = RequestAuth(uuid4().hex, expires_at=clock[0] + 300.0, generation=2)
    calls = []

    def renew(old):
        calls.append(old.generation)
        entered.set()
        assert release.wait(5)
        return replacement

    async def unexpected(old):
        raise AssertionError("an async waiter must not issue a second refresh")

    auth = ExpiringRequestAuth(
        initial=current, refresh=renew, arefresh=unexpected, clock=lambda: clock[0]
    )
    with ThreadPoolExecutor(max_workers=4) as pool:
        first = pool.submit(auth.get)
        assert await asyncio.to_thread(entered.wait, 5)
        followers = [pool.submit(auth.get) for _ in range(2)]
        asynchronous = asyncio.create_task(auth.aget())
        await asyncio.sleep(0)
        release.set()
        assert await asynchronous is replacement
        assert first.result() is replacement
        assert all(future.result() is replacement for future in followers)
    assert calls == [1]
    await auth.aclose()


@pytest.mark.asyncio
async def test_cancelled_async_refresh_waits_and_preserves_shared_result():
    entered, release = asyncio.Event(), asyncio.Event()
    clock = [time.time()]
    replacement = RequestAuth(uuid4().hex, expires_at=clock[0] + 300.0, generation=2)
    calls = []

    async def renew(old):
        calls.append(old.generation)
        entered.set()
        await release.wait()
        return replacement

    auth, _, _ = supplier(ttl=100.0, arefresh=renew)
    task = asyncio.create_task(auth.aget())
    await entered.wait()
    follower = asyncio.create_task(auth.aget())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not follower.done()
    release.set()
    assert await follower is replacement
    assert await auth.aget() is replacement
    assert calls == [1]
    await auth.aclose()


@pytest.mark.asyncio
async def test_refresh_failure_latches_exact_typed_refusal_for_all_waiters():
    refusal = InferenceRefused("accounting")
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def renew(old):
        calls.append(old.generation)
        entered.set()
        await release.wait()
        raise refusal

    auth, _, _ = supplier(ttl=100.0, arefresh=renew)
    first = asyncio.create_task(auth.aget())
    await entered.wait()
    second = asyncio.create_task(auth.aget())
    release.set()
    for task in (first, second):
        with pytest.raises(InferenceRefused) as caught:
            await task
        assert caught.value.category == refusal.category
        assert caught.value.retryable == refusal.retryable
        assert caught.value is not refusal
        assert caught.value.__context__ is None
    with pytest.raises(InferenceRefused) as caught:
        auth.get()
    assert caught.value.category == refusal.category
    assert auth._refusal == ("accounting", False)
    assert calls == [1]
    await auth.aclose()


@pytest.mark.asyncio
async def test_close_during_refresh_drains_and_never_installs_new_authority():
    entered, release = asyncio.Event(), asyncio.Event()

    async def renew(old):
        entered.set()
        await release.wait()
        return RequestAuth(uuid4().hex, expires_at=time.time() + 300.0, generation=2)

    auth, _, _ = supplier(ttl=100.0, arefresh=renew)
    request = asyncio.create_task(auth.aget())
    await entered.wait()
    close = asyncio.create_task(auth.aclose())
    await asyncio.sleep(0)
    assert not close.done()
    release.set()
    with pytest.raises(InferenceRefused):
        await request
    await close
    with pytest.raises(InferenceRefused):
        await auth.aget()


@pytest.mark.asyncio
async def test_direct_close_cancellation_still_erases_and_drains():
    entered, release = asyncio.Event(), asyncio.Event()

    async def renew(old):
        entered.set()
        await release.wait()
        return RequestAuth(uuid4().hex, expires_at=time.time() + 300.0, generation=2)

    auth, _, _ = supplier(ttl=100.0, arefresh=renew)
    waiting = asyncio.create_task(auth.aget())
    await entered.wait()
    closing = asyncio.create_task(auth.aclose())
    await asyncio.sleep(0)
    closing.cancel()
    await asyncio.sleep(0)
    closing.cancel()
    assert not closing.done()
    release.set()
    with pytest.raises(InferenceRefused):
        await waiting
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert auth._current is None


@pytest.mark.asyncio
async def test_async_refresh_deadline_fails_closed_without_stale_fallback():
    started = asyncio.Event()

    async def hung(old):
        started.set()
        await asyncio.Event().wait()

    auth, _, calls = supplier(ttl=100.0, arefresh=hung)
    auth._timeout = 0.01
    with pytest.raises(InferenceRefused) as caught:
        await asyncio.wait_for(auth.aget(), 1)
    assert caught.value.category == "auth"
    assert auth._refusal == ("auth", False)
    assert auth._future is None
    with pytest.raises(InferenceRefused):
        auth.get()
    assert not calls
    await auth.aclose()


@pytest.mark.asyncio
async def test_callback_failure_is_plain_facts_not_retained_exception_graph():
    canary = uuid4().hex
    original = ValueError(canary)

    async def fail(old):
        raise original

    auth, _, _ = supplier(ttl=100.0, arefresh=fail)
    with pytest.raises(InferenceRefused) as caught:
        await auth.aget()
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None
    assert auth._refusal == ("auth", False)
    assert auth._future is None
    assert original is not auth._refusal
    assert canary not in str(caught.value)
    await auth.aclose()


@pytest.mark.asyncio
async def test_successful_async_refresh_is_not_retained_after_close():
    auth, _, _ = supplier(ttl=100.0)
    replacement = await auth.aget()
    assert replacement.generation == 2
    assert auth._renewal_task.result() is None
    await auth.aclose()
    assert auth._current is None
    assert auth._future is None
    assert auth._renewal_task is None
    assert auth._closing.result() is None


@pytest.mark.parametrize("attribute", ["category", "retryable"])
@pytest.mark.asyncio
async def test_refresh_malformed_typed_fields_are_not_retained(attribute):
    original = InferenceRefused("budget")
    if attribute == "retryable":
        original.retryable = ValueError(uuid4().hex)
    else:

        class UnsafeCategory(str):
            pass

        original.category = UnsafeCategory("budget")
        original.category.payload = ValueError(uuid4().hex)

    async def fail(old):
        raise original

    auth, _, _ = supplier(ttl=100.0, arefresh=fail)
    with pytest.raises(InferenceRefused) as caught:
        await auth.aget()
    assert caught.value.category == "auth"
    assert type(caught.value.category) is str
    assert type(caught.value.retryable) is bool
    assert auth._refusal == ("auth", False)
    await auth.aclose()


def test_refresh_timeout_cannot_exceed_ten_seconds():
    current = RequestAuth(uuid4().hex, expires_at=time.time() + 300.0)
    with pytest.raises(InferenceRefused):
        ExpiringRequestAuth(
            initial=current,
            refresh=lambda value: value,
            arefresh=lambda value: value,
            refresh_timeout=10.1,
        )


@pytest.mark.asyncio
async def test_no_refresh_of_valid_token_at_authoritative_renewal_deadline():
    now = time.time()
    deadline = now + 100.0
    calls = []

    def unexpected(old):
        calls.append(True)
        raise AssertionError("a valid token already reaches the admitted end")

    async def aunexpected(old):
        return unexpected(old)

    original = RequestAuth(uuid4().hex, expires_at=deadline)
    auth = ExpiringRequestAuth(
        initial=original,
        refresh=unexpected,
        arefresh=aunexpected,
        renewal_deadline=deadline,
        clock=lambda: now,
    )
    assert auth.get() is original
    assert await auth.aget() is original
    assert not calls
    await auth.aclose()
