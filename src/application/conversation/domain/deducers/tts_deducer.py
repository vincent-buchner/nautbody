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
    LLMSpeakingFinished,
    LLMSpeakingStarted,
)
from application.conversation.domain.signals.tts_signal import TTSSignal


class TTSDeducer(Deducer[TTSSignal]):
    source_type = "tts"

    def __init__(self) -> None:
        pass

    def deduce(
        self, data: TTSSignal, ctx: ConversationContext
    ) -> ConversationEvent | None:
        audio_bytes = data.payload.audio_bytes
        return (
            LLMSpeakingStarted()
            if audio_bytes is None
            else LLMSpeakingFinished(audio_bytes)
        )
