from unittest.mock import patch

import numpy as np
import pytest

from src.infrastructure.vad.silero import SileroVAD


@pytest.fixture
def mock_silero():
    with (
        patch("src.infrastructure.vad.silero.load_silero_vad") as mock_load_model,
        patch("src.infrastructure.vad.silero.VADIterator") as mock_vad_iterator_cls,
    ):
        mock_model = mock_load_model.return_value
        mock_vad = mock_vad_iterator_cls.return_value
        yield mock_load_model, mock_vad_iterator_cls, mock_model, mock_vad


def test_init_loads_the_silero_model(mock_silero):
    mock_load_model, *_ = mock_silero

    SileroVAD(sample_rate=16_000)

    mock_load_model.assert_called_once_with()


def test_init_creates_vad_iterator_with_model_and_sample_rate(mock_silero):
    _, mock_vad_iterator_cls, mock_model, _ = mock_silero

    SileroVAD(sample_rate=16_000)

    mock_vad_iterator_cls.assert_called_once_with(mock_model, sampling_rate=16_000)


def test_process_audio_chunk_calls_vad_with_return_seconds(mock_silero):
    _, _, _, mock_vad = mock_silero
    mock_vad.return_value = {"start": 1.5}
    vad = SileroVAD(sample_rate=16_000)
    audio_chunk = np.zeros(512, dtype=np.float32)

    result = vad.process_audio_chunk(audio_chunk)

    mock_vad.assert_called_once_with(audio_chunk, return_seconds=True)
    assert result == {"start": 1.5}


def test_process_audio_chunk_returns_none_when_no_event(mock_silero):
    _, _, _, mock_vad = mock_silero
    mock_vad.return_value = None
    vad = SileroVAD(sample_rate=16_000)
    audio_chunk = np.zeros(512, dtype=np.float32)

    result = vad.process_audio_chunk(audio_chunk)

    assert result is None


def test_reset_resets_vad_states(mock_silero):
    _, _, _, mock_vad = mock_silero
    vad = SileroVAD(sample_rate=16_000)

    vad.reset()

    mock_vad.reset_states.assert_called_once_with()
