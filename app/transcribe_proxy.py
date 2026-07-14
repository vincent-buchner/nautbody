from faster_whisper import WhisperModel
from enum import Enum
import numpy as np


class TranscribeEvent(Enum):
    INPUT_FINISHED = "input_finished"
    INPUT_DELTA = "input_delta"
    INPUT_NAN = "no_input"
    INPUT_RESET = "input_reset"


class TranscribeProxy:
    def __init__(self) -> None:
        self._model = WhisperModel("base", device="cpu", compute_type="int16")
        self._current_input_text: list[str] = []

    def transcribe(self, audio_bytes: np.ndarray) -> TranscribeEvent:
        segments, info = self._model.transcribe(
            audio_bytes, beam_size=3, language="en", vad_filter=True
        )

        print("duration after vad: ", info.duration_after_vad)
        if info.duration_after_vad == 0 and len(self._current_input_text) > 0:
            return TranscribeEvent.INPUT_FINISHED

        if info.duration_after_vad == 0 and len(self._current_input_text) == 0:
            return TranscribeEvent.INPUT_NAN

        texts = []
        for seg in segments:
            texts.append(seg.text)

        self._current_input_text.append(" ".join(texts))
        return TranscribeEvent.INPUT_DELTA

    def clear_current_text(self) -> TranscribeEvent:
        self._current_input_text = []
        return TranscribeEvent.INPUT_RESET

    def get_current_text(self):
        return self._current_input_text
