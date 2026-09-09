from application.conversation.domain.context.context import (
    ConversationContext,
)
from application.conversation.domain.deducers.duducer import (
    Deducer,
)
from application.conversation.domain.events.event import (
    ConversationEvent,
)
from application.conversation.domain.events.llm_events import (
    LLMResponseStartedEvent,
    LLMResponseStoppedEvent,
)
from application.conversation.domain.signals.llm_signal import (
    LLMSignal,
)


class LLMEventDeducer(Deducer[LLMSignal]):
    source_type = "llm_stream"

    def __init__(self) -> None:
        pass

    def deduce(
        self, data: LLMSignal, ctx: ConversationContext
    ) -> ConversationEvent | None:
        payload = data.payload
        if payload.llm_response_text is not None:
            return LLMResponseStoppedEvent(payload.llm_response_text)
        if payload.user_input is not None:
            return LLMResponseStartedEvent()
        return None
