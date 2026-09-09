from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TTSSignal:
    @dataclass(frozen=True)
    class Payload:
        audio_bytes: bytes | None = None

    payload: Payload
    source_type: Literal["tts"] = "tts"
