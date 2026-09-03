from typing import Protocol

from application.conversation.use_cases.start_conversation.context import (
    ConversationContext,
)
from application.conversation.use_cases.start_conversation.events import (
    ConversationEvent,
)
from application.conversation.use_cases.start_conversation.raw_data import (
    SourceType,
    TRawData,
)


class Deducer(Protocol[TRawData]):
    source_type: SourceType

    def deduce(
        self, data: TRawData, ctx: ConversationContext
    ) -> ConversationEvent | None: ...
