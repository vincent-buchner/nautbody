from dataclasses import dataclass
from typing import Literal

from application.conversation.domain.signals.signal import Signal


@dataclass(frozen=True)
class TTSSignal(Signal):
    @dataclass(frozen=True)
    class StartedPayload:
        pass

    @dataclass(frozen=True)
    class FinishedPayload:
        audio_bytes: bytes

    Payload = StartedPayload | FinishedPayload

    payload: Payload
    source_type: Literal["tts"] = "tts"
