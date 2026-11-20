from dataclasses import dataclass
from typing import Literal

from application.conversation.domain.signals.signal import Signal


@dataclass(frozen=True)
class LLMSignal(Signal):
    @dataclass(frozen=True)
    class StartedPayload:
        user_input: str

    @dataclass(frozen=True)
    class FinishedPayload:
        llm_response_text: str

    Payload = StartedPayload | FinishedPayload

    payload: Payload
    source_type: Literal["llm_stream"] = "llm_stream"
