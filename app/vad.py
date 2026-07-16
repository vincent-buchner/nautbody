from silero_vad import VADIterator, load_silero_vad
import numpy as np


class VAD:
    def __init__(self, sample_rate: int) -> None:
        self._model = load_silero_vad()
        self._vad = VADIterator(self._model, sampling_rate=sample_rate)

    def build_events(self, audio_bytes: np.ndarray):
        event = self._vad(audio_bytes, return_seconds=True)
        return event

    def reset_vad(self):
        self._vad.reset_states()
