from dataclasses import dataclass

from application.conversation.domain.events.event import (
    ConversationEvent,
)


class LLMResponseStartedEvent(ConversationEvent): ...


@dataclass
class LLMResponseStoppedEvent(ConversationEvent):
    llm_response_text: str
