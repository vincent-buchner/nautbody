from pathlib import Path

import torch
from chatterbox.tts_turbo import ChatterboxTurboTTS as ChatterboxTTSModel


class ChatterboxTTS:
    def __init__(
        self, sample_audio_path: Path | str, verbose_generation: bool = False
    ) -> None:
        torch.set_num_threads(16)
        self._model = ChatterboxTTSModel.from_pretrained(device="cpu", nano=True)
        self._model.prepare_conditionals(sample_audio_path)
        self._verbose_generation = verbose_generation

    def generate_audio(self, text: str) -> bytes:
        generated_audio_tensor = self._model.generate(text)
        audio_bytes = generated_audio_tensor.numpy().tobytes()
        return audio_bytes
