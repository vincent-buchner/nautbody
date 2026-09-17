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
    AssistantResponseStartedEvent,
    AssistantResponseStoppedEvent,
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
        match data.payload:
            case LLMSignal.StartedPayload():
                if not ctx.is_user_speaking:
                    return AssistantResponseStartedEvent()
            case LLMSignal.FinishedPayload(llm_response_text=llm_response_text):
                return AssistantResponseStoppedEvent(llm_response_text)
