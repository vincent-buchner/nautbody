"""Manual smoke test for StartConversation: mic -> buffer queue -> VAD -> STT -> TTS.

Requires GROQ_API_KEY to be set (e.g. in a .env file).

Run: python scripts/run_conversation.py
"""

import asyncio
import os
import sys
import threading
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, "src")

from application.conversation.useCases.start_conversation.start_conversation import (
    StartConversation,
)
from core.asynchronous.buffer_queue import AsyncBufferQueue
from infrastructure.audio.input.pyaudio import PyAudioInput
from infrastructure.audio.output.pyaudio import PyAudioOutput
from infrastructure.llm_provider import GroqModelConfig, GroqProvider
from infrastructure.stt.fast_whisper import FastWhisperTTS
from infrastructure.tts.chatterbox import ChatterboxTTS
from infrastructure.vad.silero import SileroVAD

load_dotenv()

SAMPLE_RATE = 24_000
OUTPUT_SAMPLE_RATE = 24_000  # ChatterboxTTS (S3Gen) generates audio at 24kHz
VAD_CHUNK_SIZE = 512  # Silero VAD requires exactly this many samples at 16kHz
SAMPLE_AUDIO_PATH = (
    Path(__file__).parent.parent
    / "tests/src/infrastructure/audio/test_audio_input.flac"
)
LLM_SYSTEM_PROMPT = "You are a helpful voice assistant. Keep responses brief."
LLM_MODEL_CONFIG = GroqModelConfig(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    max_completion_tokens=256,
    top_p=1,
    stop=None,
)


def feed_microphone(
    mic: PyAudioInput,
    queue: AsyncBufferQueue[bytes],
    loop: asyncio.AbstractEventLoop,
) -> None:
    for chunk in mic.start_microphone():
        queue.put_buffer(chunk)
        asyncio.run_coroutine_threadsafe(queue.release_buffer(), loop)


async def play_audio_output(
    speaker: PyAudioOutput,
    queue: AsyncBufferQueue[bytes],
    loop: asyncio.AbstractEventLoop,
) -> None:
    while True:
        audio_bytes = await queue.pull()
        await loop.run_in_executor(None, speaker.play_speaker, audio_bytes)


async def main() -> None:
    loop = asyncio.get_running_loop()

    mic = PyAudioInput(sample_rate=SAMPLE_RATE, chunk_size=VAD_CHUNK_SIZE)
    speaker = PyAudioOutput(sample_rate=OUTPUT_SAMPLE_RATE)
    audio_in_buffer_queue = AsyncBufferQueue[bytes]()
    audio_out_buffer_queue = AsyncBufferQueue[bytes]()

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
    conversation.execute()
    loop.create_task(play_audio_output(speaker, audio_out_buffer_queue, loop))

    threading.Thread(
        target=feed_microphone,
        args=(mic, audio_in_buffer_queue, loop),
        daemon=True,
    ).start()

    print("Listening... speak into the mic. Ctrl+C to stop.")
    await asyncio.Event().wait()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # feed_microphone's thread is blocked in a synchronous stream.read()
        # call; joining it or closing the stream from here can hang if it's
        # mid-read. os._exit skips all of that and kills the process outright.
        os._exit(0)
