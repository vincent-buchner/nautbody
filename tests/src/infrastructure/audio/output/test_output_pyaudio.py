from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest

from src.infrastructure.audio.output.pyaudio import PyAudioOutput


class MockStream:
    def __init__(self) -> None:
        self.write = MagicMock()
        self.stop_stream = MagicMock()
        self.close = MagicMock()


@pytest.fixture
def mock_pyaudio() -> Iterator[tuple[MagicMock, MockStream]]:
    with patch("src.infrastructure.audio.output.pyaudio.pyaudio.PyAudio") as mock_cls:
        mock_audio = mock_cls.return_value
        mock_stream = MockStream()
        mock_audio.open.return_value = mock_stream
        yield mock_audio, mock_stream


def test_play_speaker_opens_stream_with_configured_params(
    mock_pyaudio: tuple[MagicMock, MockStream], pcm_bytes: bytes
) -> None:
    mock_audio, _ = mock_pyaudio
    audio_output = PyAudioOutput(sample_rate=16_000, channels=1)

    audio_output.play_speaker(pcm_bytes)

    import pyaudio as pyaudio_module

    mock_audio.open.assert_called_once_with(
        format=pyaudio_module.paFloat32,
        channels=1,
        rate=16_000,
        output=True,
        frames_per_buffer=2048,
    )


def test_play_speaker_writes_audio_bytes_to_stream(
    mock_pyaudio: tuple[MagicMock, MockStream], pcm_bytes: bytes
) -> None:
    _, mock_stream = mock_pyaudio
    audio_output = PyAudioOutput(sample_rate=16_000, channels=1)

    audio_output.play_speaker(pcm_bytes)

    mock_stream.write.assert_called_once_with(pcm_bytes)


def test_play_speaker_reuses_existing_stream(
    mock_pyaudio: tuple[MagicMock, MockStream], pcm_bytes: bytes
) -> None:
    mock_audio, mock_stream = mock_pyaudio
    audio_output = PyAudioOutput(sample_rate=16_000, channels=1)

    audio_output.play_speaker(pcm_bytes[:1024])
    audio_output.play_speaker(pcm_bytes[1024:2048])

    mock_audio.open.assert_called_once()
    assert mock_stream.write.call_count == 2


def test_kill_speaker_stops_and_closes_stream(
    mock_pyaudio: tuple[MagicMock, MockStream], pcm_bytes: bytes
) -> None:
    _, mock_stream = mock_pyaudio
    audio_output = PyAudioOutput(sample_rate=16_000, channels=1)
    audio_output.play_speaker(pcm_bytes)

    audio_output.kill_speaker()

    mock_stream.stop_stream.assert_called_once()
    mock_stream.close.assert_called_once()


def test_kill_speaker_without_stream_is_a_noop(
    mock_pyaudio: tuple[MagicMock, MockStream],
) -> None:
    audio_output = PyAudioOutput(sample_rate=16_000, channels=1)

    audio_output.kill_speaker()


def test_close_audio_terminates_pyaudio(
    mock_pyaudio: tuple[MagicMock, MockStream],
) -> None:
    mock_audio, _ = mock_pyaudio
    audio_output = PyAudioOutput(sample_rate=16_000, channels=1)

    audio_output.close_audio()

    mock_audio.terminate.assert_called_once()
