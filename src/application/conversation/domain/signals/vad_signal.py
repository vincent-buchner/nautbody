from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from application.conversation.domain.signals.signal import Signal

# The VAD port's raw detection output -- shared with the IVAD port's
# return type so the port and the domain agree on one shape.
VADResult = dict[str, int | float]


@dataclass(frozen=True)
class VADSignal(Signal):
    @dataclass(frozen=True)
    class StartedPayload:
        pass

    @dataclass(frozen=True)
    class DeltaPayload:
        audio_bytes: np.ndarray

    @dataclass(frozen=True)
    class StoppedPayload:
        pass

    Payload = StartedPayload | DeltaPayload | StoppedPayload

    payload: Payload
    source_type: Literal["vad"] = "vad"

    @classmethod
    def from_vad_result(
        cls, vad_result: VADResult | None, audio_bytes: np.ndarray
    ) -> VADSignal | None:
        """
        Translates the VAD port's raw dict into a typed payload variant.
        Returns None when the raw result doesn't map to a known state
        (present, but with neither a start nor an end marker).
        """
        if vad_result is None:
            return cls(payload=cls.DeltaPayload(audio_bytes))
        if vad_result.get("start"):
            return cls(payload=cls.StartedPayload())
        if vad_result.get("end"):
            return cls(payload=cls.StoppedPayload())
        return None
