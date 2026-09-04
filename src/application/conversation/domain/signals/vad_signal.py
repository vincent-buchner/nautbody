from dataclasses import dataclass
from typing import Literal

import numpy as np

# The VAD port's raw detection output -- shared with the IVAD port's
# return type so the port and the domain agree on one shape.
VADResult = dict[str, int | float]


@dataclass(frozen=True)
class VADSignal:
    @dataclass(frozen=True)
    class Payload:
        vad_data: VADResult | None
        audio_bytes: np.ndarray

    payload: Payload
    source_type: Literal["vad"] = "vad"
