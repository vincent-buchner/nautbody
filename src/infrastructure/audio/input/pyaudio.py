from collections.abc import Generator

import pyaudio


class PyAudioInput:
    def __init__(
        self, chunk_size: int = 1024, sample_rate: int = 16_000, channels: int = 1
    ) -> None:
        self._audio = pyaudio.PyAudio()
        self._stream = None

        self._chunk_size = chunk_size
        self._sample_rate = sample_rate
        self._channels = channels

    def start_microphone(self) -> Generator[bytes, None, None]:
        if self._stream is None:
            self._stream = self._audio.open(
                rate=self._sample_rate,
                format=pyaudio.paInt16,
                input=True,
                channels=self._channels,
                frames_per_buffer=self._chunk_size,
            )

        while True:
            audio_bytes = self._stream.read(self._chunk_size)
            yield audio_bytes

    def kill_microphone(self) -> None:
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()

    def close_audio(self):
        self._audio.terminate()
