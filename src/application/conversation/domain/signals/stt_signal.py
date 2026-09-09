from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class STTSignal:
    @dataclass(frozen=True)
    class Payload:
        transcription: str | None = None

    payload: Payload
    source_type: Literal["stt"] = "stt"
