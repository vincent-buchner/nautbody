import pyaudio


class PyAudioOutput:
    def __init__(self, sample_rate: int = 16_000, channels: int = 1) -> None:
        self._audio = pyaudio.PyAudio()
        self._stream = None

        self._sample_rate = sample_rate
        self._channels = channels

    def play_speaker(self, audio_bytes: bytes) -> None:
        if self._stream is None:
            self._stream = self._audio.open(
                format=pyaudio.paFloat32,
                channels=self._channels,
                rate=self._sample_rate,
                output=True,
                frames_per_buffer=2048,
            )
        self._stream.write(audio_bytes)

    def kill_speaker(self) -> None:
        if self._stream is None:
            return
        self._stream.stop_stream()
        self._stream.close()

    def close_audio(self) -> None:
        self._audio.terminate()
