import asyncio
from collections.abc import AsyncIterator

import numpy as np

from application.conversation.ports.ILLMProvider import ILLMProvider
from application.conversation.ports.ISTT import ISTT
from application.conversation.ports.ITTS import ITTS
from application.conversation.ports.IVAD import IVAD


class StartConversation:
    def __init__(
        self,
        llm_provider: ILLMProvider,
        stt: ISTT,
        tts: ITTS,
        vad: IVAD,
        audio_in_buffer_queue: asyncio.Queue[bytes],
        audio_out_buffer_queue: asyncio.Queue[bytes],
    ) -> None:
        self._llm_provider = llm_provider
        self._stt = stt
        self._tts = tts
        self._vad = vad

        self._loop = asyncio.get_running_loop()
        self._audio_in_buffer_queue = audio_in_buffer_queue
        self._audio_out_buffer_queue = audio_out_buffer_queue
        self._audio_to_text_buffer: list[np.ndarray] = []

        self._is_user_speaking = False

    def execute(self) -> None:
        self._loop.create_task(self._run())

    async def _run(self) -> None:
        async for chunk in self._process():
            ndarray_chunk = np.frombuffer(chunk, dtype=np.float32).copy()
            vad_result = self._vad.process_audio_chunk(ndarray_chunk)

            if vad_result is None:
                self._handle_intermediate_event(ndarray_chunk)
                continue

            if vad_result.get("start"):
                self._handle_start_event()

            if vad_result.get("end"):
                await self._handle_end_event()

    async def _process(self) -> AsyncIterator[bytes]:
        while True:
            chunk = await self._audio_in_buffer_queue.get()
            yield chunk

    def _handle_intermediate_event(self, audio_chunk: np.ndarray) -> None:
        if self._is_user_speaking:
            self._audio_to_text_buffer.append(audio_chunk)

    def _handle_start_event(self) -> None:
        self._is_user_speaking = True
        print("Start Speaking")

    async def _handle_end_event(self) -> None:

        self._is_user_speaking = False
        built_up_audio = np.array(self._audio_to_text_buffer).flatten()
        self._audio_to_text_buffer.clear()
        print("Stop Speaking")

        user_text = self._stt.generate_text(built_up_audio)
        print(f"You said: {user_text}")
        if not user_text.strip():
            return

        llm_response = self._llm_provider.generate_response(user_text)
        print(f"LLM said: {llm_response}")
        if llm_response is None:
            return

        audio = self._tts.generate_audio(llm_response)
        await self._audio_out_buffer_queue.put(audio)
