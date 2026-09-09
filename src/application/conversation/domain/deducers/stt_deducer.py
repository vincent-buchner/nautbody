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
    LLMTranscriptionFinished,
    LLMTranscriptionStarted,
)
from application.conversation.domain.signals.stt_signal import STTSignal


class STTDeducer(Deducer[STTSignal]):
    source_type = "stt"

    def __init__(self) -> None:
        pass

    def deduce(
        self, data: STTSignal, ctx: ConversationContext
    ) -> ConversationEvent | None:
        transcription = data.payload.transcription
        return (
            LLMTranscriptionStarted()
            if transcription is None
            else LLMTranscriptionFinished(transcription)
        )
