from typing import Protocol

import numpy as np


class IVAD(Protocol):
    def process_audio_chunk(
        self, audio_bytes: np.ndarray
    ) -> dict[str, float] | None: ...

    def reset(self) -> None: ...
