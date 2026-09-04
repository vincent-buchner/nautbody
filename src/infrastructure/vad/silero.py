import numpy as np
from silero_vad import VADIterator, load_silero_vad

from application.conversation.domain.signals.vad_signal import VADResult


class SileroVAD:
    def __init__(self, sample_rate: int) -> None:
        self._model = load_silero_vad()
        # TODO: Add better vad settings for padding speech
        self._vad = VADIterator(self._model, sampling_rate=sample_rate)

    def process_audio_chunk(self, audio_bytes: np.ndarray) -> VADResult | None:
        event = self._vad(audio_bytes, return_seconds=True)
        return event

    def reset(self) -> None:
        self._vad.reset_states()
