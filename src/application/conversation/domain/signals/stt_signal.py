from dataclasses import dataclass
from typing import Literal

from application.conversation.domain.signals.signal import Signal


@dataclass(frozen=True)
class STTSignal(Signal):
    @dataclass(frozen=True)
    class StartedPayload:
        pass

    @dataclass(frozen=True)
    class FinishedPayload:
        transcription: str

    Payload = StartedPayload | FinishedPayload

    payload: Payload
    source_type: Literal["stt"] = "stt"
