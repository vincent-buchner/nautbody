import numpy as np
from faster_whisper import WhisperModel


class FastWhisperTTS:
    def __init__(self, beam_size: int = 3, language: str = "en") -> None:
        self._model = WhisperModel("base", device="cpu", compute_type="int8")
        self._beam_size = beam_size
        self._language = language

    def generate_text(self, audio_bytes: np.ndarray) -> str:
        segments, _ = self._model.transcribe(
            audio_bytes,
            beam_size=self._beam_size,
            language=self._language,
            vad_filter=True,
        )

        textsArray = []
        for seg in segments:
            textsArray.append(seg.text)

        if len(textsArray) > 0:
            return " ".join(textsArray)

        return ""
