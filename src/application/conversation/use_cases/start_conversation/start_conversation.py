import asyncio
from collections.abc import AsyncIterator
from typing import TypeVar

import numpy as np

from application.conversation.domain.context.context import (
    ConversationContext,
)
from application.conversation.domain.deducers.llm_event_deducer import (
    LLMEventDeducer,
)
from application.conversation.domain.deducers.stt_deducer import STTDeducer
from application.conversation.domain.deducers.tts_deducer import TTSDeducer
from application.conversation.domain.deducers.vad_event_deducer import (
    VADEventDeducer,
)
from application.conversation.domain.events.event import (
    ConversationEvent,
)
from application.conversation.domain.events.llm_events import (
    AssistantResponseStoppedEvent,
)
from application.conversation.domain.events.router import (
    EventHandler,
    EventRouter,
)
from application.conversation.domain.events.stt_events import (
    AssistantTranscriptionFinishedEvent,
)
from application.conversation.domain.events.vad_events import (
    UserDeltaSpeakingEvent,
    UserInterruptedAssistantEvent,
    UserStartSpeakingEvent,
    UserStopSpeakingEvent,
)
from application.conversation.domain.signals.llm_signal import LLMSignal
from application.conversation.domain.signals.stt_signal import STTSignal
from application.conversation.domain.signals.tts_signal import (
    TTSSignal,
)
from application.conversation.domain.signals.vad_signal import VADSignal
from application.conversation.ports.ILLMProvider import ILLMProvider
from application.conversation.ports.ISTT import ISTT
from application.conversation.ports.ITTS import ITTS
from application.conversation.ports.IVAD import IVAD

TEvent = TypeVar("TEvent", bound=ConversationEvent)


class StartConversation:
    def __init__(
        self,
        llm_provider: ILLMProvider,
        stt: ISTT,
        tts: ITTS,
        vad: IVAD,
        audio_chunks: AsyncIterator[bytes],
    ) -> None:
        self._llm_provider = llm_provider
        self._stt = stt
        self._tts = tts
        self._vad = vad

        self._loop = asyncio.get_running_loop()
        self._audio_chunks = audio_chunks
        self._audio_to_text_buffer: list[np.ndarray] = []
        self._active_turn: asyncio.Task[None] | None = None

        self._event_router = EventRouter()
        self._ctx = ConversationContext(is_user_speaking=False)

        self._event_router.register("vad", VADEventDeducer())
        self._event_router.register("llm_stream", LLMEventDeducer())
        self._event_router.register("tts", TTSDeducer())
        self._event_router.register("stt", STTDeducer())

    def on(self, event_type: type[TEvent], handler: EventHandler[TEvent]) -> None:
        self._event_router.on(event_type, handler)

    def execute(self) -> None:
        self._loop.create_task(self._run())

    async def _run(self) -> None:
        self._event_router.on(UserStartSpeakingEvent, self._handle_start_event)
        self._event_router.on(UserStopSpeakingEvent, self._handle_end_event)
        self._event_router.on(UserDeltaSpeakingEvent, self._handle_intermediate_event)
        self._event_router.on(
            UserInterruptedAssistantEvent, self._handle_interruption_event
        )
        self._event_router.on(
            AssistantResponseStoppedEvent, self._handle_llm_response_stop
        )
        self._event_router.on(
            AssistantTranscriptionFinishedEvent, self._handle_llm_transcription_finished
        )
        async for chunk in self._audio_chunks:
            ndarray_chunk = np.frombuffer(chunk, dtype=np.float32).copy()
            vad_result = self._vad.process_audio_chunk(ndarray_chunk)

            data = VADSignal.from_vad_result(vad_result, ndarray_chunk)
            if data is not None:
                await self._event_router.process(data, self._ctx)

    async def _handle_llm_response_stop(
        self, event: AssistantResponseStoppedEvent
    ) -> None:
        if not event.llm_response_text.strip():
            return

        await self._event_router.process(
            TTSSignal(payload=TTSSignal.StartedPayload()), self._ctx
        )
        audio = self._tts.generate_audio(event.llm_response_text)
        await self._event_router.process(
            TTSSignal(payload=TTSSignal.FinishedPayload(audio_bytes=audio)), self._ctx
        )

    async def _handle_interruption_event(
        self, event: UserInterruptedAssistantEvent
    ) -> None:
        if self._active_turn is not None:
            self._active_turn.cancel()
            try:
                await self._active_turn
            except asyncio.CancelledError:
                pass
            self._active_turn = None

        await self._event_router.process(
            TTSSignal(payload=TTSSignal.CancelledPayload()), self._ctx
        )
        self._audio_to_text_buffer.clear()
        self._ctx.is_assistant_speaking = False

    async def _handle_llm_transcription_finished(
        self, event: AssistantTranscriptionFinishedEvent
    ) -> None:
        if not event.transcription.strip():
            return

        await self._event_router.process(
            LLMSignal(payload=LLMSignal.StartedPayload(user_input=event.transcription)),
            self._ctx,
        )

        llm_response = self._llm_provider.generate_response(event.transcription) or ""
        await self._event_router.process(
            LLMSignal(
                payload=LLMSignal.FinishedPayload(llm_response_text=llm_response)
            ),
            self._ctx,
        )

    def _handle_intermediate_event(self, event: UserDeltaSpeakingEvent) -> None:
        if self._ctx.is_user_speaking:
            self._audio_to_text_buffer.append(event.audio_bytes)

    def _handle_start_event(self, event: UserStartSpeakingEvent) -> None:
        self._ctx.is_user_speaking = True

    def _handle_end_event(self, event: UserStopSpeakingEvent) -> None:
        self._ctx.is_user_speaking = False
        built_up_audio = np.array(self._audio_to_text_buffer).flatten()
        self._audio_to_text_buffer.clear()

        self._active_turn = self._loop.create_task(
            self._run_response_pipeline(built_up_audio)
        )

    async def _run_response_pipeline(self, built_up_audio: np.ndarray) -> None:
        try:
            await self._event_router.process(
                STTSignal(payload=STTSignal.StartedPayload()), self._ctx
            )

            user_text = self._stt.generate_text(built_up_audio)
            await self._event_router.process(
                STTSignal(payload=STTSignal.FinishedPayload(transcription=user_text)),
                self._ctx,
            )
        finally:
            self._active_turn = None
