from typing import Literal, TypeVar

from application.conversation.domain.signals.llm_signal import (
    LLMSignal,
)
from application.conversation.domain.signals.tts_signal import TTSSignal
from application.conversation.domain.signals.vad_signal import VADSignal

SourceType = Literal["vad", "llm_stream", "tts"]

Signal = VADSignal | LLMSignal | TTSSignal
TSignal = TypeVar("TSignal", bound=Signal, contravariant=True)
