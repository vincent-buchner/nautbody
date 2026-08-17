from pathlib import Path

from application.conversation.use_cases.start_conversation.start_conversation import (
    StartConversation,
)
from core.asynchronous.buffer_queue import AsyncBufferQueue
from infrastructure.llm_provider import GroqModelConfig, GroqProvider
from infrastructure.stt.fast_whisper import FastWhisperTTS
from infrastructure.tts.chatterbox import ChatterboxTTS
from infrastructure.vad.silero import SileroVAD

SAMPLE_AUDIO_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "tests/src/infrastructure/audio/test_audio_input.flac"
)

# NOTE: Must be 8_000 or 16_000 per SileroVAD
SAMPLE_RATE = 16_000
LLM_SYSTEM_PROMPT = "You are a helpful voice assistant. Keep responses brief."

LLM_SYSTEM_PROMPT = "You are a helpful voice assistant. Keep responses brief."
LLM_MODEL_CONFIG = GroqModelConfig(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    max_completion_tokens=256,
    top_p=1,
    stop=None,
)


def makeStartConversationUseCase(
    audio_in_buffer_queue: AsyncBufferQueue, audio_out_buffer_queue: AsyncBufferQueue
) -> StartConversation:

    conversation = StartConversation(
        llm_provider=GroqProvider(
            system_prompt=LLM_SYSTEM_PROMPT, model_config=LLM_MODEL_CONFIG
        ),
        stt=FastWhisperTTS(),
        tts=ChatterboxTTS(sample_audio_path=SAMPLE_AUDIO_PATH),
        vad=SileroVAD(sample_rate=SAMPLE_RATE),
        audio_in_buffer_queue=audio_in_buffer_queue,
        audio_out_buffer_queue=audio_out_buffer_queue,
    )

    return conversation
