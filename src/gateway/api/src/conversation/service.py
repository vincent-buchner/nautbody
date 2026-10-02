from fastapi import WebSocket

from application.conversation.domain.events.event import ConversationEvent
from application.conversation.domain.events.llm_events import (
    AssistantResponseStartedEvent,
    AssistantResponseStoppedEvent,
)
from application.conversation.domain.events.stt_events import (
    AssistantTranscriptionFinishedEvent,
    AssistantTranscriptionStartedEvent,
)
from application.conversation.domain.events.tts_events import (
    AssistantSpeakingCancelledEvent,
    AssistantSpeakingFinishedEvent,
    AssistantSpeakingStartedEvent,
)
from application.conversation.domain.events.vad_events import (
    UserInterruptedAssistantEvent,
    UserStartSpeakingEvent,
    UserStopSpeakingEvent,
)
from application.conversation.use_cases.start_conversation.start_conversation import (
    StartConversation,
)
from gateway.api.src.conversation.schemas import EventMessage


class WebSocketConversationHandler:
    def __init__(
        self,
        ws: WebSocket,
    ) -> None:
        self._ws = ws

    async def send_event_name(self, event: ConversationEvent) -> None:
        res = EventMessage(event.__class__.__name__)
        await self._ws.send_json(res.to_dict())

    async def handle_assistant_response_stopped(
        self,
        event: AssistantResponseStoppedEvent,
    ) -> None:
        res = EventMessage(event.__class__.__name__, event.llm_response_text)
        await self._ws.send_json(res.to_dict())

    async def handle_assistant_response_finished(
        self,
        event: AssistantTranscriptionFinishedEvent,
    ) -> None:
        res = EventMessage(event.__class__.__name__, event.transcription)
        await self._ws.send_json(res.to_dict())

    async def handle_assistant_speaking_stopped(
        self,
        event: AssistantSpeakingFinishedEvent,
    ) -> None:
        await self._ws.send_bytes(event.audio_response)


def register_ws_handlers(
    ws: WebSocket,
    conversation: StartConversation,
) -> None:

    handlers = WebSocketConversationHandler(ws)

    conversation.on(UserStartSpeakingEvent, handlers.send_event_name)
    conversation.on(UserStopSpeakingEvent, handlers.send_event_name)
    conversation.on(UserInterruptedAssistantEvent, handlers.send_event_name)
    conversation.on(AssistantResponseStartedEvent, handlers.send_event_name)
    conversation.on(
        AssistantResponseStoppedEvent, handlers.handle_assistant_response_stopped
    )
    conversation.on(AssistantTranscriptionStartedEvent, handlers.send_event_name)
    conversation.on(
        AssistantTranscriptionFinishedEvent,
        handlers.handle_assistant_response_finished,
    )
    conversation.on(AssistantSpeakingStartedEvent, handlers.send_event_name)
    conversation.on(
        AssistantSpeakingFinishedEvent,
        handlers.handle_assistant_speaking_stopped,
    )
    conversation.on(
        AssistantSpeakingCancelledEvent,
        handlers.send_event_name,
    )
