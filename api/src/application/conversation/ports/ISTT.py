from typing import Protocol

import numpy as np


class ISTT(Protocol):
    def generate_text(self, audio_bytes: np.ndarray) -> str: ...
