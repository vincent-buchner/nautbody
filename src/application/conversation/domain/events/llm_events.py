from dataclasses import dataclass

from application.conversation.domain.events.event import (
    ConversationEvent,
)


class AssistantResponseStartedEvent(ConversationEvent): ...


@dataclass
class AssistantResponseStoppedEvent(ConversationEvent):
    llm_response_text: str
