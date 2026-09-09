from application.conversation.domain.context.context import (
    ConversationContext,
)
from application.conversation.domain.deducers.duducer import (
    Deducer,
)
from application.conversation.domain.events.event import (
    ConversationEvent,
)
from application.conversation.domain.events.user_events import (
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
