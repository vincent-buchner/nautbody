import asyncio
from collections.abc import Awaitable


class AsyncBufferQueue[T]:
    async def __init__(self) -> None:
        self._queue = asyncio.Queue[T]()
        self._buffer: list[T] = []

    def put_buffer(self, item: T):
        self._buffer.append(item)

    async def release_buffer(self) -> None:
        for item in self._buffer:
            await self._queue.put(item)

    async def pull(self) -> Awaitable[T]:
        return self._queue.get()
