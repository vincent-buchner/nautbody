import asyncio

import pytest
from src.core.asynchronous.buffer_queue import AsyncBufferQueue

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def test_put_buffer_does_not_release_items_until_release_buffer_called() -> None:
    queue = AsyncBufferQueue[str]()
    queue.put_buffer("a")

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(queue.pull(), timeout=0.05)


async def test_release_buffer_pushes_buffered_items_into_the_queue() -> None:
    queue = AsyncBufferQueue[str]()
    queue.put_buffer("a")
    queue.put_buffer("b")

    await queue.release_buffer()

    first = await queue.pull()
    second = await queue.pull()

    assert [first, second] == ["a", "b"]


async def test_release_buffer_preserves_fifo_order() -> None:
    queue = AsyncBufferQueue[int]()
    for i in range(5):
        queue.put_buffer(i)

    await queue.release_buffer()

    results = [await queue.pull() for _ in range(5)]

    assert results == [0, 1, 2, 3, 4]


async def test_release_buffer_with_empty_buffer_is_a_noop() -> None:
    queue = AsyncBufferQueue[str]()

    await queue.release_buffer()

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(queue.pull(), timeout=0.05)


async def test_release_buffer_can_be_called_multiple_times() -> None:
    queue = AsyncBufferQueue[str]()
    queue.put_buffer("a")
    await queue.release_buffer()

    queue.put_buffer("b")
    await queue.release_buffer()

    first = await queue.pull()
    second = await queue.pull()

    assert [first, second] == ["a", "b"]


async def test_pull_returns_the_item_directly() -> None:
    queue = AsyncBufferQueue[str]()
    queue.put_buffer("a")
    await queue.release_buffer()

    pulled = await queue.pull()

    assert pulled == "a"
