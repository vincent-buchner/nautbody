from application.conversation.domain.context.context import (
    ConversationContext,
)
from application.conversation.domain.deducers.duducer import (
    Deducer,
)
from application.conversation.domain.events.event import (
    ConversationEvent,
)
from application.conversation.domain.events.stt_events import (
    AssistantTranscriptionFinishedEvent,
    AssistantTranscriptionStartedEvent,
)
from application.conversation.domain.signals.stt_signal import STTSignal


class STTDeducer(Deducer[STTSignal]):
    source_type = "stt"

    def __init__(self) -> None:
        pass

    def deduce(
        self, data: STTSignal, ctx: ConversationContext
    ) -> ConversationEvent | None:
        match data.payload:
            case STTSignal.StartedPayload():
                return AssistantTranscriptionStartedEvent()
            case STTSignal.FinishedPayload(transcription=transcription):
                return AssistantTranscriptionFinishedEvent(transcription)
