from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.infrastructure.stt.fast_whisper import FastWhisperTTS


@pytest.fixture
def mock_whisper_model():
    with patch("src.infrastructure.stt.fast_whisper.WhisperModel") as mock_cls:
        mock_model = mock_cls.return_value
        mock_model.transcribe.return_value = ([], MagicMock())
        yield mock_cls, mock_model


def _segments(*texts: str) -> list[SimpleNamespace]:
    return [SimpleNamespace(text=text) for text in texts]


def test_init_creates_whisper_model_with_expected_params(mock_whisper_model):
    mock_cls, _ = mock_whisper_model

    FastWhisperTTS()

    mock_cls.assert_called_once_with("base", device="cpu", compute_type="int8")


def test_generate_text_returns_joined_segment_texts(mock_whisper_model):
    _, mock_model = mock_whisper_model
    mock_model.transcribe.return_value = (_segments("hello", "world"), MagicMock())
    stt = FastWhisperTTS()

    result = stt.generate_text(np.zeros(16_000, dtype=np.float32))

    assert result == "hello world"


def test_generate_text_returns_empty_string_when_no_segments(mock_whisper_model):
    _, mock_model = mock_whisper_model
    mock_model.transcribe.return_value = (_segments(), MagicMock())
    stt = FastWhisperTTS()

    result = stt.generate_text(np.zeros(16_000, dtype=np.float32))

    assert result == ""


def test_generate_text_passes_audio_and_configured_params(mock_whisper_model):
    _, mock_model = mock_whisper_model
    stt = FastWhisperTTS(beam_size=5, language="fr")
    audio = np.zeros(16_000, dtype=np.float32)

    stt.generate_text(audio)

    mock_model.transcribe.assert_called_once_with(
        audio,
        beam_size=5,
        language="fr",
        vad_filter=True,
    )


def test_generate_text_uses_default_beam_size_and_language(mock_whisper_model):
    _, mock_model = mock_whisper_model
    stt = FastWhisperTTS()
    audio = np.zeros(16_000, dtype=np.float32)

    stt.generate_text(audio)

    mock_model.transcribe.assert_called_once_with(
        audio,
        beam_size=3,
        language="en",
        vad_filter=True,
    )
