import inspect
from collections.abc import Awaitable, Callable
from typing import TypeVar, cast

from application.conversation.domain.context.context import (
    ConversationContext,
)
from application.conversation.domain.deducers.duducer import (
    Deducer,
)
from application.conversation.domain.events.event import (
    ConversationEvent,
)
from application.conversation.domain.signals.signals import Signal

TEvent = TypeVar("TEvent", bound=ConversationEvent)

EventHandler = Callable[[TEvent], None | Awaitable[None]]


class EventRouter:
    def __init__(self) -> None:
        self._deducers: dict[str, list[Deducer]] = {}
        self._listeners: dict[
            type[ConversationEvent], list[EventHandler[ConversationEvent]]
        ] = {}

    def register(self, source: str, deducer: Deducer) -> None:
        self._deducers.setdefault(source, []).append(deducer)

    def on(self, event_type: type[TEvent], handler: EventHandler[TEvent]) -> None:
        # Callable is contravariant in its param, so EventHandler[TEvent]
        # isn't an EventHandler[ConversationEvent]. Cast needed to store it.
        # Pro: callers get precisely typed handlers, no manual narrowing.
        # Con: nothing enforces event_type actually matches handler's type --
        # a mismatch would only surface as a runtime error inside the handler.
        self._listeners.setdefault(event_type, []).append(
            cast(EventHandler[ConversationEvent], handler)
        )

    async def process(self, data: Signal, ctx: ConversationContext) -> None:
        deducers = self._deducers.get(data.source_type, [])
        events = (
            event
            for deducer in deducers
            if (event := deducer.deduce(data, ctx)) is not None
        )
        for event in events:
            for handler in self._listeners.get(type(event), []):
                result = handler(event)
                if inspect.isawaitable(result):
                    await result
