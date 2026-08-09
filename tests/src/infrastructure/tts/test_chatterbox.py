from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.infrastructure.tts.chatterbox import ChatterboxTTS


def _tensor_chunk(values: list[int]) -> MagicMock:
    tensor = MagicMock()
    tensor.numpy.return_value = np.array(values, dtype=np.int16)
    return tensor


def _drain(generator):
    chunks = []
    try:
        while True:
            chunks.append(next(generator))
    except StopIteration as stop:
        return chunks, stop.value


@pytest.fixture
def mock_model_cls():
    with patch("src.infrastructure.tts.chatterbox.ChatterboxTTSModel") as mock_cls:
        mock_model = mock_cls.from_pretrained.return_value
        mock_model.generate_stream.return_value = [
            (_tensor_chunk([1, 2]), "metrics-1"),
            (_tensor_chunk([3, 4]), "metrics-2"),
        ]
        yield mock_cls, mock_model


def test_init_loads_model_from_pretrained_on_cuda(mock_model_cls):
    mock_cls, _ = mock_model_cls

    ChatterboxTTS(sample_audio_path="voice.wav")

    mock_cls.from_pretrained.assert_called_once_with(device="cuda")


def test_init_prepares_conditionals_with_sample_audio_path(mock_model_cls):
    _, mock_model = mock_model_cls

    ChatterboxTTS(sample_audio_path="voice.wav")

    mock_model.prepare_conditionals.assert_called_once_with("voice.wav")


def test_generate_audio_calls_generate_stream_with_text(mock_model_cls):
    _, mock_model = mock_model_cls
    tts = ChatterboxTTS(sample_audio_path="voice.wav")

    list(tts.generate_audio("hello world"))

    mock_model.generate_stream.assert_called_once_with("hello world")


def test_generate_audio_yields_bytes_chunks(mock_model_cls):
    tts = ChatterboxTTS(sample_audio_path="voice.wav")

    chunks, _ = _drain(tts.generate_audio("hello"))

    assert chunks == [
        np.array([1, 2], dtype=np.int16).tobytes(),
        np.array([3, 4], dtype=np.int16).tobytes(),
    ]


def test_generate_audio_return_value_collects_all_chunks(mock_model_cls):
    tts = ChatterboxTTS(sample_audio_path="voice.wav")

    chunks, returned = _drain(tts.generate_audio("hello"))

    assert returned == chunks


def test_generate_audio_prints_metrics_when_verbose(mock_model_cls):
    tts = ChatterboxTTS(sample_audio_path="voice.wav", verbose_generation=True)

    with patch("builtins.print") as mock_print:
        list(tts.generate_audio("hello"))

    assert mock_print.call_count == 2


def test_generate_audio_does_not_print_when_not_verbose(mock_model_cls):
    tts = ChatterboxTTS(sample_audio_path="voice.wav", verbose_generation=False)

    with patch("builtins.print") as mock_print:
        list(tts.generate_audio("hello"))

    mock_print.assert_not_called()
