import asyncio
from collections.abc import AsyncIterator

import numpy as np

from application.conversation.domain.context.context import (
    ConversationContext,
)
from application.conversation.domain.deducers.llm_event_deducer import (
    LLMEventDeducer,
)
from application.conversation.domain.deducers.user_event_deducer import (
    UserEventDeducer,
)
from application.conversation.domain.events.llm_events import (
    LLMResponseStartedEvent,
    LLMResponseStoppedEvent,
)
from application.conversation.domain.events.router import (
    EventRouter,
)
from application.conversation.domain.events.user_events import (
    UserDeltaSpeakingEvent,
    UserStartSpeakingEvent,
    UserStopSpeakingEvent,
)
from application.conversation.domain.signals.llm_signal import LLMSignal
from application.conversation.domain.signals.vad_signal import VADSignal
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

        self._event_router = EventRouter()
        self._event_router.register("vad", UserEventDeducer())
        self._event_router.register("llm_stream", LLMEventDeducer())

    def execute(self) -> None:
        self._loop.create_task(self._run())

    async def _run(self) -> None:
        self._event_router.on(UserStartSpeakingEvent, self._handle_start_event)
        self._event_router.on(UserStopSpeakingEvent, self._handle_end_event)
        self._event_router.on(UserDeltaSpeakingEvent, self._handle_intermediate_event)
        self._event_router.on(LLMResponseStartedEvent, self._handle_llm_response_start)
        self._event_router.on(LLMResponseStoppedEvent, self._handle_llm_response_stop)
        async for chunk in self._process():
            ndarray_chunk = np.frombuffer(chunk, dtype=np.float32).copy()
            vad_result = self._vad.process_audio_chunk(ndarray_chunk)

            data = VADSignal(
                payload=VADSignal.Payload(
                    vad_data=vad_result, audio_bytes=ndarray_chunk
                )
            )
            await self._event_router.process(data, ConversationContext())

    async def _process(self) -> AsyncIterator[bytes]:
        while True:
            chunk = await self._audio_in_buffer_queue.get()
            yield chunk

    def _handle_llm_response_start(self, event: LLMResponseStartedEvent) -> None:
        print(event.__class__)

    async def _handle_llm_response_stop(self, event: LLMResponseStoppedEvent) -> None:
        print(event.__class__)
        print(event.llm_response_text)
        audio = self._tts.generate_audio(event.llm_response_text)
        await self._audio_out_buffer_queue.put(audio)

    def _handle_intermediate_event(self, event: UserDeltaSpeakingEvent) -> None:
        print(event.__class__)
        if self._is_user_speaking:
            self._audio_to_text_buffer.append(event.audio_bytes)

    def _handle_start_event(self, event: UserStartSpeakingEvent) -> None:
        print(event.__class__)
        self._is_user_speaking = True

    async def _handle_end_event(self, event: UserStopSpeakingEvent) -> None:

        print(event.__class__)
        self._is_user_speaking = False
        built_up_audio = np.array(self._audio_to_text_buffer).flatten()
        self._audio_to_text_buffer.clear()

        user_text = self._stt.generate_text(built_up_audio)
        print(f"You said: {user_text}")
        if not user_text.strip():
            return

        await self._event_router.process(
            LLMSignal(
                payload=LLMSignal.Payload(user_input=user_text, llm_response_text=None)
            ),
            ConversationContext(),
        )

        llm_response = self._llm_provider.generate_response(user_text)
        if llm_response is None:
            return
        await self._event_router.process(
            LLMSignal(
                payload=LLMSignal.Payload(
                    user_input=None, llm_response_text=llm_response
                )
            ),
            ConversationContext(),
        )
