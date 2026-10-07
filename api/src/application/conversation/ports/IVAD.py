from typing import Protocol

import numpy as np

from application.conversation.domain.signals.vad_signal import VADResult


class IVAD(Protocol):
    def process_audio_chunk(self, audio_bytes: np.ndarray) -> VADResult | None: ...

    def reset(self) -> None: ...
