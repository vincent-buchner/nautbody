from application.conversation.domain.context.context import (
    ConversationContext,
)
from application.conversation.domain.deducers.IDuducer import (
    Deducer,
)
from application.conversation.domain.events.events import (
    ConversationEvent,
    UserDeltaSpeakingEvent,
    UserStartSpeakingEvent,
    UserStopSpeakingEvent,
)
from application.conversation.domain.signals.vad_signal import (
    VADSignal,
)


class UserEventDeducer(Deducer[VADSignal]):
    source_type = "vad"

    def __init__(self) -> None:
        pass

    def deduce(
        self, data: VADSignal, ctx: ConversationContext
    ) -> ConversationEvent | None:
        vad_data = data.payload.vad_data
        if vad_data is None:
            return UserDeltaSpeakingEvent(data.payload.audio_bytes)
        if vad_data.get("start"):
            return UserStartSpeakingEvent()
        if vad_data.get("end"):
            return UserStopSpeakingEvent()
