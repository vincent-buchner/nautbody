from collections.abc import Generator
from pathlib import Path

from chatterbox.tts import ChatterboxTTS as ChatterboxTTSModel


class ChatterboxTTS:
    def __init__(
        self, sample_audio_path: Path | str, verbose_generation: bool = False
    ) -> None:
        self._model = ChatterboxTTSModel.from_pretrained(device="cuda")
        self._model.prepare_conditionals(sample_audio_path)
        self._verbose_generation = verbose_generation

    def generate_audio(self, text: str) -> Generator[bytes, None, list[bytes]]:
        full_audio_collection: list[bytes] = []

        for audio_tensor_chunk, metrics in self._model.generate_stream(text):
            audio_bytes_chunk = audio_tensor_chunk.numpy().tobytes()
            full_audio_collection.append(audio_bytes_chunk)

            if self._verbose_generation:
                print(f"Audio chunk generated:\n{metrics}")

            yield audio_bytes_chunk

        return full_audio_collection
