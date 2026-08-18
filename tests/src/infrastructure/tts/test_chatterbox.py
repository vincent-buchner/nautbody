from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.infrastructure.tts.chatterbox import ChatterboxTTS


def _tensor(values: list[int]) -> MagicMock:
    tensor = MagicMock()
    tensor.numpy.return_value = np.array(values, dtype=np.int16)
    return tensor


@pytest.fixture
def mock_model_cls() -> Iterator[tuple[MagicMock, MagicMock]]:
    with patch("src.infrastructure.tts.chatterbox.ChatterboxTTSModel") as mock_cls:
        mock_model = mock_cls.from_pretrained.return_value
        mock_model.generate.return_value = _tensor([1, 2, 3, 4])
        yield mock_cls, mock_model


def test_init_loads_model_from_pretrained_on_cpu_nano(
    mock_model_cls: tuple[MagicMock, MagicMock],
) -> None:
    mock_cls, _ = mock_model_cls

    ChatterboxTTS(sample_audio_path="voice.wav")

    mock_cls.from_pretrained.assert_called_once_with(device="cpu", nano=True)


def test_init_prepares_conditionals_with_sample_audio_path(
    mock_model_cls: tuple[MagicMock, MagicMock],
) -> None:
    _, mock_model = mock_model_cls

    ChatterboxTTS(sample_audio_path="voice.wav")

    mock_model.prepare_conditionals.assert_called_once_with("voice.wav")


def test_generate_audio_calls_generate_with_text(
    mock_model_cls: tuple[MagicMock, MagicMock],
) -> None:
    _, mock_model = mock_model_cls
    tts = ChatterboxTTS(sample_audio_path="voice.wav")

    tts.generate_audio("hello world")

    mock_model.generate.assert_called_once_with("hello world")


def test_generate_audio_returns_tensor_bytes(
    mock_model_cls: tuple[MagicMock, MagicMock],
) -> None:
    tts = ChatterboxTTS(sample_audio_path="voice.wav")

    result = tts.generate_audio("hello")

    assert result == np.array([1, 2, 3, 4], dtype=np.int16).tobytes()
