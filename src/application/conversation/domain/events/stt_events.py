from dataclasses import dataclass

from application.conversation.domain.events.event import (
    ConversationEvent,
)


class LLMTranscriptionStarted(ConversationEvent): ...


@dataclass
class LLMTranscriptionFinished(ConversationEvent):
    transcription: str
