from application.conversation.use_cases.start_conversation.context import (
    ConversationContext,
)
from application.conversation.use_cases.start_conversation.deducers.IDuducer import (
    Deducer,
)
from application.conversation.use_cases.start_conversation.events import (
    ConversationEvent,
    UserDeltaSpeakingEvent,
    UserStartSpeakingEvent,
    UserStopSpeakingEvent,
)
from application.conversation.use_cases.start_conversation.raw_data import (
    VADData,
)


class UserEventDeducer(Deducer[VADData]):
    source_type = "vad"

    def __init__(self) -> None:
        pass

    def deduce(
        self, data: VADData, ctx: ConversationContext
    ) -> ConversationEvent | None:
        vad_data = data.payload.vad_data
        if vad_data is None:
            return UserDeltaSpeakingEvent(data.payload.audio_bytes)
        if vad_data.get("start"):
            return UserStartSpeakingEvent()
        if vad_data.get("end"):
            return UserStopSpeakingEvent()
