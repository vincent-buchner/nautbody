from faster_whisper import WhisperModel
import numpy as np


class TranscribeProxy:
    def __init__(self) -> None:
        self._model = WhisperModel("base", device="cpu", compute_type="int16")
        self._current_input_text: list[str] = []

    def transcribe(self, audio_bytes: np.ndarray) -> str:
        segments, info = self._model.transcribe(
            audio_bytes, beam_size=3, language="en", vad_filter=True
        )

        textsArray = []
        for seg in segments:
            textsArray.append(seg.text)

        if len(textsArray) > 0:
            return " ".join(textsArray)
        return ""
