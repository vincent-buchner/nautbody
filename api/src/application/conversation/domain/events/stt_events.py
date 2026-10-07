from dataclasses import dataclass

from application.conversation.domain.events.event import (
    ConversationEvent,
)


class AssistantTranscriptionStartedEvent(ConversationEvent): ...


@dataclass
class AssistantTranscriptionFinishedEvent(ConversationEvent):
    transcription: str
