from typing import Literal, TypeVar

from application.conversation.domain.signals.llm_signal import (
    LLMSignal,
)
from application.conversation.domain.signals.stt_signal import STTSignal
from application.conversation.domain.signals.tts_signal import TTSSignal
from application.conversation.domain.signals.vad_signal import VADSignal

SourceType = Literal["vad", "llm_stream", "tts", "stt"]

Signal = VADSignal | LLMSignal | TTSSignal | STTSignal
TSignal = TypeVar("TSignal", bound=Signal, contravariant=True)
