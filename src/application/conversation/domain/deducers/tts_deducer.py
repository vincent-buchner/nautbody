from application.conversation.domain.context.context import (
    ConversationContext,
)
from application.conversation.domain.deducers.duducer import (
    Deducer,
)
from application.conversation.domain.events.event import (
    ConversationEvent,
)
from application.conversation.domain.events.tts_events import (
    AssistantSpeakingFinishedEvent,
    AssistantSpeakingStartedEvent,
)
from application.conversation.domain.signals.tts_signal import TTSSignal


class TTSDeducer(Deducer[TTSSignal]):
    source_type = "tts"

    def __init__(self) -> None:
        pass

    def deduce(
        self, data: TTSSignal, ctx: ConversationContext
    ) -> ConversationEvent | None:
        match data.payload:
            case TTSSignal.StartedPayload():
                return AssistantSpeakingStartedEvent()
            case TTSSignal.FinishedPayload(audio_bytes=audio_bytes):
                return AssistantSpeakingFinishedEvent(audio_bytes)
