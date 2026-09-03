from typing import Protocol

import numpy as np


class IVAD(Protocol):
    def process_audio_chunk(
        self,
        audio_bytes: np.ndarray,
        # TODO: Complex types / Return types should be broken out
    ) -> dict[str, float] | None: ...

    def reset(self) -> None: ...
