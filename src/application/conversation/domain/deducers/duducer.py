from typing import Protocol

from application.conversation.domain.context.context import (
    ConversationContext,
)
from application.conversation.domain.events.event import (
    ConversationEvent,
)
from application.conversation.domain.signals.signal import TSignal


class Deducer(Protocol[TSignal]):
    source_type: str

    def deduce(
        self, data: TSignal, ctx: ConversationContext
    ) -> ConversationEvent | None: ...
