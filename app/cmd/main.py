from dataclasses import astuple
import sys
from pathlib import Path
from dotenv import load_dotenv
import asyncio
import sounddevice as sd
import numpy as np
from torch import flatten


ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from app.llm import LLMProxy  # noqa: E402
from app.transcribe_proxy import TranscribeProxy  # noqa: E402
from app.audio_buffer_queue import AudioBufferQueue  # noqa: E402
from app.vad import VAD  # noqa: E402
from app.tts import TTSService  # noqa: E402


load_dotenv()

# TODO: Move to toml config
sample_rate = 16000


# TODO: Move to toml config (seconds)
audio_queue = queue = AudioBufferQueue(sample_rate, 3)
transcription_proxy = TranscribeProxy()
vad = VAD(sample_rate)
llm = LLMProxy()
tts = TTSService("./static/test/audio/test_audio.flac")


async def audio_processor():
    ######################################
    # START OF AUDIO -> TRANSCRIBER
    ######################################
    isSpeaking = False

    # TODO: Use the AudioBufferQueue and rework
    audio_bytes_buffer = []

    debounce_timer: asyncio.Task | None = None

    # TODO: Switch this over to BufferQueue
    transcription_collection: list[str] = []

    # NOTE: While the agent's response is being played back through the
    # speakers, the mic InputStream is still capturing. Without this gate the
    # playback gets picked back up by the mic, VAD triggers on it, and it gets
    # transcribed as if the user said it -- producing garbled follow-up turns.
    agent_speaking = asyncio.Event()

    async def message_after_duration(seconds: float):
        # When transcription ends, wait `x` amount of time
        # If start (isSpeaking) is triggered before x time reaches, only add to buffer
        # If no start while time before x time reaches, add to buffer and send the flatten buffer to the LLM service.
        # Then the LLM responses.
        # Print response

        await asyncio.sleep(seconds)

        nonlocal transcription_collection

        ######################################
        # START OF TRANSCRIPTS -> LLM
        ######################################
        print("Making agent response")
        response = llm.generate(" ".join(transcription_collection))
        transcription_collection = []
        ######################################
        # END OF TRANSCRIPTS -> LLM
        ######################################

        if response is None:
            print("Didn't receive anything from LLM")
            return

        audio = tts.generate(response)

        audio_np = audio.squeeze(0).cpu().numpy()

        agent_speaking.set()
        try:
            await asyncio.get_running_loop().run_in_executor(
                None, lambda: (sd.play(audio_np, samplerate=tts.sample_rate), sd.wait())
            )
        finally:
            # Drop any VAD state built up from picking up our own playback,
            # then resume listening for the user.
            vad.reset_vad()
            agent_speaking.clear()

    while True:
        audio_bytes = await audio_queue.get()

        if agent_speaking.is_set():
            # Don't let the agent's own speech coming out of the speakers
            # get treated as user input.
            continue

        result = await asyncio.get_running_loop().run_in_executor(
            None, vad.build_events, audio_bytes
        )

        print(result if result is not None else "", end="", flush=True)

        startTimeInSeconds = result.get("start") if result is not None else None
        endTimeInSeconds = result.get("end") if result is not None else None

        if startTimeInSeconds is not None:
            isSpeaking = True

            # NOTE: This is to cancel the timer when the user starts speaking
            if debounce_timer is not None and not debounce_timer.done():
                debounce_timer.cancel()
                debounce_timer = None

        if isSpeaking:
            audio_bytes_buffer.append(audio_bytes)

        if endTimeInSeconds is not None:
            transcription = await asyncio.get_running_loop().run_in_executor(
                None,
                transcription_proxy.transcribe,
                np.array(audio_bytes_buffer).flatten(),
            )
            transcription_collection.append(transcription)
            audio_bytes_buffer = []
            isSpeaking = False

            debounce_timer = asyncio.create_task(message_after_duration(3))

    ######################################
    # END OF AUDIO -> TRANSCRIBER
    ######################################


async def main():
    print("Starting... Press Ctrl+C to kill")

    task = asyncio.create_task(audio_processor())
    global_loop = asyncio.get_running_loop()

    def callback(indata: np.ndarray, frames, time, status):
        if status:
            print(f"Status: {status}")

        try:
            asyncio.run_coroutine_threadsafe(
                audio_queue.put(indata.flatten()),
                global_loop,
            )
        except Exception as e:
            print(f"\nCallback error: {e}")

    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=512,
            callback=callback,
        ):
            await asyncio.sleep(np.inf)
    except asyncio.CancelledError:
        print("\nMain task cancelled")
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Graceful shutdown
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        print("Shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())

# ========== NOTES ==========
#
