import threading

import pyaudio

BYTES_PER_SAMPLE = 4  # paFloat32


class PyAudioOutput:
    def __init__(
        self,
        sample_rate: int = 16_000,
        channels: int = 1,
        frames_per_buffer: int = 2048,
    ) -> None:
        self._audio = pyaudio.PyAudio()
        self._stream: pyaudio.Stream | None = None

        self._sample_rate = sample_rate
        self._channels = channels
        self._frames_per_buffer = frames_per_buffer
        self._stop_event = threading.Event()

    def play_speaker(self, audio_bytes: bytes) -> None:
        if self._stream is None:
            self._stream = self._audio.open(
                format=pyaudio.paFloat32,
                channels=self._channels,
                rate=self._sample_rate,
                output=True,
                frames_per_buffer=self._frames_per_buffer,
            )

        self._stop_event.clear()
        chunk_size = self._frames_per_buffer * BYTES_PER_SAMPLE * self._channels
        for offset in range(0, len(audio_bytes), chunk_size):
            if self._stop_event.is_set():
                return
            self._stream.write(audio_bytes[offset : offset + chunk_size])

    def kill_speaker(self) -> None:
        # Only ever flips a flag the writer thread checks between chunks --
        # never touches self._stream, since that's owned exclusively by
        # whichever thread is inside play_speaker.
        self._stop_event.set()

    def close_audio(self) -> None:
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        self._audio.terminate()
