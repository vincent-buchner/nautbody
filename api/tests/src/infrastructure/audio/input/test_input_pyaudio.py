import itertools
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest

from src.infrastructure.audio.input.pyaudio import PyAudioInput


class MockStream:
    def __init__(self, pcm_bytes: bytes, chunk_size: int, bytes_per_frame: int) -> None:
        self._chunk_bytes = chunk_size * bytes_per_frame
        self._offset = 0
        self._pcm_bytes = pcm_bytes
        self.stop_stream = MagicMock()
        self.close = MagicMock()

    def read(self, _chunk_size: int) -> bytes:
        start = self._offset
        end = start + self._chunk_bytes
        self._offset = end
        return self._pcm_bytes[start:end]


@pytest.fixture
def mock_pyaudio(pcm_bytes: bytes) -> Iterator[tuple[MagicMock, MockStream]]:
    with patch("src.infrastructure.audio.input.pyaudio.pyaudio.PyAudio") as mock_cls:
        mock_audio = mock_cls.return_value
        mock_stream = MockStream(pcm_bytes, chunk_size=1024, bytes_per_frame=4)
        mock_audio.open.return_value = mock_stream
        yield mock_audio, mock_stream


def test_start_microphone_opens_stream_with_configured_params(
    mock_pyaudio: tuple[MagicMock, MockStream],
) -> None:
    mock_audio, _ = mock_pyaudio
    audio_input = PyAudioInput(chunk_size=1024, sample_rate=16_000, channels=1)

    next(audio_input.start_microphone())

    import pyaudio as pyaudio_module

    mock_audio.open.assert_called_once_with(
        rate=16_000,
        format=pyaudio_module.paFloat32,
        input=True,
        channels=1,
        frames_per_buffer=1024,
    )


def test_start_microphone_yields_chunks_from_audio_source(
    mock_pyaudio: tuple[MagicMock, MockStream], pcm_bytes: bytes
) -> None:
    audio_input = PyAudioInput(chunk_size=1024, sample_rate=16_000, channels=1)

    chunks = list(itertools.islice(audio_input.start_microphone(), 5))

    assert len(chunks) == 5
    for chunk in chunks[:-1]:
        assert len(chunk) == 1024 * 4
    assert b"".join(chunks) == pcm_bytes[: 5 * 1024 * 4]


def test_start_microphone_reuses_existing_stream(
    mock_pyaudio: tuple[MagicMock, MockStream],
) -> None:
    mock_audio, _mock_stream = mock_pyaudio
    audio_input = PyAudioInput(chunk_size=1024, sample_rate=16_000, channels=1)

    gen = audio_input.start_microphone()
    next(gen)
    next(gen)

    mock_audio.open.assert_called_once()


def test_kill_microphone_stops_and_closes_stream(
    mock_pyaudio: tuple[MagicMock, MockStream],
) -> None:
    _, mock_stream = mock_pyaudio
    audio_input = PyAudioInput(chunk_size=1024, sample_rate=16_000, channels=1)
    next(audio_input.start_microphone())

    audio_input.kill_microphone()

    mock_stream.stop_stream.assert_called_once()
    mock_stream.close.assert_called_once()


def test_kill_microphone_without_stream_is_a_noop(
    mock_pyaudio: tuple[MagicMock, MockStream],
) -> None:
    audio_input = PyAudioInput(chunk_size=1024, sample_rate=16_000, channels=1)

    audio_input.kill_microphone()


def test_close_audio_terminates_pyaudio(
    mock_pyaudio: tuple[MagicMock, MockStream],
) -> None:
    mock_audio, _ = mock_pyaudio
    audio_input = PyAudioInput(chunk_size=1024, sample_rate=16_000, channels=1)

    audio_input.close_audio()

    mock_audio.terminate.assert_called_once()
