from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest

from src.infrastructure.llm_provider.groq import GroqModelConfig, GroqProvider

MODEL_CONFIG = GroqModelConfig(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    max_completion_tokens=256,
    top_p=1,
    stop=None,
)


def _mock_completion(content: str | None) -> MagicMock:
    completion = MagicMock()
    completion.choices[0].message.content = content
    return completion


@pytest.fixture(autouse=True)
def groq_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-api-key")


@pytest.fixture
def mock_groq() -> Iterator[tuple[MagicMock, MagicMock]]:
    with patch("src.infrastructure.llm_provider.groq.Groq") as mock_cls:
        mock_client = mock_cls.return_value
        mock_client.chat.completions.create.return_value = _mock_completion(
            "mocked response"
        )
        yield mock_cls, mock_client


def test_init_creates_client_with_api_key_from_env(
    mock_groq: tuple[MagicMock, MagicMock],
) -> None:
    mock_cls, _ = mock_groq

    GroqProvider(system_prompt="you are a helpful assistant", model_config=MODEL_CONFIG)

    mock_cls.assert_called_once_with(api_key="test-api-key")


def test_generate_response_returns_message_content(
    mock_groq: tuple[MagicMock, MagicMock],
) -> None:
    _, mock_client = mock_groq
    mock_client.chat.completions.create.return_value = _mock_completion("hello there")
    provider = GroqProvider(
        system_prompt="you are a helpful assistant", model_config=MODEL_CONFIG
    )

    result = provider.generate_response("hi")

    assert result == "hello there"


def test_generate_response_returns_none_when_no_content(
    mock_groq: tuple[MagicMock, MagicMock],
) -> None:
    _, mock_client = mock_groq
    mock_client.chat.completions.create.return_value = _mock_completion(None)
    provider = GroqProvider(
        system_prompt="you are a helpful assistant", model_config=MODEL_CONFIG
    )

    result = provider.generate_response("hi")

    assert result is None


def test_generate_response_sends_system_and_user_messages(
    mock_groq: tuple[MagicMock, MagicMock],
) -> None:
    _, mock_client = mock_groq
    provider = GroqProvider(
        system_prompt="you are a helpful assistant", model_config=MODEL_CONFIG
    )

    provider.generate_response("hi")

    mock_client.chat.completions.create.assert_called_once_with(
        messages=[
            {"role": "system", "content": "you are a helpful assistant"},
            {"role": "user", "content": "hi"},
        ],
        model="llama-3.1-8b-instant",
        temperature=0.7,
        max_completion_tokens=256,
        top_p=1,
        stop=None,
    )


def test_generate_response_accumulates_conversation_history(
    mock_groq: tuple[MagicMock, MagicMock],
) -> None:
    _, mock_client = mock_groq
    mock_client.chat.completions.create.side_effect = [
        _mock_completion("first reply"),
        _mock_completion("second reply"),
    ]
    provider = GroqProvider(
        system_prompt="you are a helpful assistant", model_config=MODEL_CONFIG
    )

    provider.generate_response("first message")
    provider.generate_response("second message")

    second_call_messages = mock_client.chat.completions.create.call_args_list[1].kwargs[
        "messages"
    ]
    assert second_call_messages == [
        {"role": "system", "content": "you are a helpful assistant"},
        {"role": "user", "content": "first message"},
        {"role": "user", "content": "second message"},
    ]
