from application.conversation.domain.context.context import (
    ConversationContext,
)
from application.conversation.domain.deducers.duducer import (
    Deducer,
)
from application.conversation.domain.events.event import (
    ConversationEvent,
)
from application.conversation.domain.events.vad_events import (
    UserDeltaSpeakingEvent,
    UserStartSpeakingEvent,
    UserStopSpeakingEvent,
)
from application.conversation.domain.signals.vad_signal import (
    VADSignal,
)


class VADEventDeducer(Deducer[VADSignal]):
    source_type = "vad"

    def __init__(self) -> None:
        pass

    def deduce(
        self, data: VADSignal, ctx: ConversationContext
    ) -> ConversationEvent | None:
        match data.payload:
            case VADSignal.StartedPayload():
                return UserStartSpeakingEvent()
            case VADSignal.DeltaPayload(audio_bytes=audio_bytes):
                return UserDeltaSpeakingEvent(audio_bytes)
            case VADSignal.StoppedPayload():
                return UserStopSpeakingEvent()
