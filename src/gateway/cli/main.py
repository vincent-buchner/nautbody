import asyncio
import os
import sys
import threading

from dotenv import load_dotenv

sys.path.insert(0, "src")

from gateway.cli.bootstrap import make_start_conversation_use_case
from infrastructure.audio.input.pyaudio import PyAudioInput
from infrastructure.audio.output.pyaudio import PyAudioOutput

load_dotenv()

SAMPLE_RATE = 16_000
OUTPUT_SAMPLE_RATE = 24_000
VAD_CHUNK_SIZE = 512


def feed_microphone(
    mic: PyAudioInput,
    queue: asyncio.Queue[bytes],
    loop: asyncio.AbstractEventLoop,
) -> None:
    for chunk in mic.start_microphone():
        asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)


async def play_audio_output(
    speaker: PyAudioOutput,
    queue: asyncio.Queue[bytes],
) -> None:
    while True:
        audio_bytes = await queue.get()
        await asyncio.to_thread(speaker.play_speaker, audio_bytes)


async def main() -> None:
    loop = asyncio.get_running_loop()

    mic = PyAudioInput(sample_rate=SAMPLE_RATE, chunk_size=VAD_CHUNK_SIZE)
    speaker = PyAudioOutput(sample_rate=OUTPUT_SAMPLE_RATE)
    audio_in_buffer_queue = asyncio.Queue[bytes]()
    audio_out_buffer_queue = asyncio.Queue[bytes]()

    conversation = make_start_conversation_use_case(
        audio_in_buffer_queue, audio_out_buffer_queue
    )
    conversation.execute()

    loop.create_task(play_audio_output(speaker, audio_out_buffer_queue))

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
