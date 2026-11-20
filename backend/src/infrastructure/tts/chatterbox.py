from pathlib import Path

import numpy as np
import torch
from chatterbox.tts_turbo import ChatterboxTurboTTS as ChatterboxTTSModel


# NOTE: Current Issue Open for this float32 conversion: https://github.com/resemble-ai/chatterbox/issues/499
class _Float32ChatterboxTurboTTS(ChatterboxTTSModel):
    # NOTE: Disable Rule no type return and no typed params, missing from original as
    # well, this is monkey patch to get arounf `Double but found Float` issue
    def norm_loudness(self, wav, sr, target_lufs=-27):  # noqa: ANN001, ANN202
        out = super().norm_loudness(wav, sr, target_lufs)
        return out.astype(np.float32, copy=False)


class ChatterboxTTS:
    def __init__(
        self, sample_audio_path: Path | str, verbose_generation: bool = False
    ) -> None:
        torch.set_num_threads(16)
        self._model = _Float32ChatterboxTurboTTS.from_pretrained(
            device="cpu", nano=True
        )
        self._model.prepare_conditionals(sample_audio_path)
        self._verbose_generation = verbose_generation

    def generate_audio(self, text: str) -> bytes:
        generated_audio_tensor = self._model.generate(text)
        audio_bytes = generated_audio_tensor.numpy().tobytes()
        return audio_bytes
