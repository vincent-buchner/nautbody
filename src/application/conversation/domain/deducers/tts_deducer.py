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
    AssistantSpeakingCancelledEvent,
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
                ctx.is_assistant_speaking = True
                if not ctx.is_user_speaking:
                    return AssistantSpeakingStartedEvent()
            case TTSSignal.FinishedPayload(audio_bytes=audio_bytes):
                ctx.is_assistant_speaking = False
                return AssistantSpeakingFinishedEvent(audio_bytes)
            case TTSSignal.CancelledPayload():
                ctx.is_assistant_speaking = False
                return AssistantSpeakingCancelledEvent()
