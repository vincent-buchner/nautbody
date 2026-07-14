import asyncio
import numpy as np


# ================== Q / A ==================
# Q: What is the sample rate in audio processing? What does it do?
# ===========================================


class AudioBufferQueue:
    def __init__(self, sample_rate: int, target_chunk_duration: float = 3) -> None:
        self.sample_rate = sample_rate
        self._audio_queue: asyncio.Queue[np.ndarray] = asyncio.Queue()

        self._buffer = []
        self._target_buffer_size = sample_rate * target_chunk_duration
        self._current_buffer_size = 0

        self.buffer_full_event = asyncio.Event()

    def __put_buffer(self, audio_bytes: np.ndarray) -> bool:
        self._buffer.append(audio_bytes.copy())
        self._current_buffer_size += len(audio_bytes)

        isCurrentGreaterThanTargetBufferSize = (
            self._current_buffer_size > self._target_buffer_size
        )

        if isCurrentGreaterThanTargetBufferSize:
            self.buffer_full_event.set()

        return isCurrentGreaterThanTargetBufferSize

    async def __get_buffer(self) -> np.ndarray:
        # This makes sure that the audio buffer is actually
        # full first before continuing
        await self.buffer_full_event.wait()

        concatenated_audio_bytes = np.concatenate(self._buffer)

        result = concatenated_audio_bytes[: self._target_buffer_size]

        if len(concatenated_audio_bytes) > self._target_buffer_size:
            self._buffer = [concatenated_audio_bytes[self._target_buffer_size :]]
            self._current_buffer_size = len(self._buffer[0])
        else:
            self._buffer = []
            self._current_buffer_size = 0

        self.buffer_full_event.clear()

        return result

    async def put(self, audio_bytes: np.ndarray) -> None:
        isBufferFull = self.__put_buffer(audio_bytes)

        if isBufferFull:
            audio_buffer = await self.__get_buffer()
            await self._audio_queue.put(audio_buffer)

    async def get(self) -> np.ndarray:
        return await self._audio_queue.get()
