from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class LLMSignal:
    @dataclass(frozen=True)
    class Payload:
        user_input: str | None
        llm_response_text: str | None

    payload: Payload
    source_type: Literal["llm_stream"] = "llm_stream"
