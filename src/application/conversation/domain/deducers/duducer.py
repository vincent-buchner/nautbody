from typing import Protocol

from application.conversation.domain.context.context import (
    ConversationContext,
)
from application.conversation.domain.events.events import (
    ConversationEvent,
)
from application.conversation.domain.signals.signals import (
    SourceType,
    TSignal,
)


class Deducer(Protocol[TSignal]):
    source_type: SourceType

    def deduce(
        self, data: TSignal, ctx: ConversationContext
    ) -> ConversationEvent | None: ...
