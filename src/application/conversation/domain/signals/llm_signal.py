from dataclasses import dataclass
from typing import Literal

# TODO: Placeholder until the real LLM streaming interface exists.
# This is expected to become a union of several signal variants
# (e.g. LLMStartedSignal, LLMStoppedSignal, LLMToolCallResultSignal),
# mirroring how VADSignal/ConversationEvent are structured.


@dataclass(frozen=True)
class LLMPlaceholderSignal:
    @dataclass(frozen=True)
    class Payload:
        is_true: bool

    payload: Payload
    source_type: Literal["llm_stream"] = "llm_stream"
