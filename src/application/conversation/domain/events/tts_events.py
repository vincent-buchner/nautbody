from dataclasses import dataclass

from application.conversation.domain.events.event import (
    ConversationEvent,
)


class AssistantSpeakingStartedEvent(ConversationEvent): ...


@dataclass
class AssistantSpeakingFinishedEvent(ConversationEvent):
    audio_response: bytes
