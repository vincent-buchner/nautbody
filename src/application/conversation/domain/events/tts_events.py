from dataclasses import dataclass

from application.conversation.domain.events.event import (
    ConversationEvent,
)


class LLMSpeakingStarted(ConversationEvent): ...


@dataclass
class LLMSpeakingFinished(ConversationEvent):
    audio_response: bytes
