from dotenv import load_dotenv
import asyncio
import sounddevice as sd
import numpy as np

from llm import LLMProxy
from transcribe_proxy import TranscribeEvent, TranscribeProxy
from audio_buffer_queue import AudioBufferQueue

load_dotenv()


# ============== KNOWN ISSUES ==============
# [ ] The chunking duration time might result in poorly parsed sentences, should
# look into a way to make it continuous
#   - What I think we can do is inttroduce a VAD before the transcribe_proxy for detecting pauses

sample_rate = 16000

audio_queue = queue = AudioBufferQueue(sample_rate, 3)
transcription_proxy = TranscribeProxy()
llm = LLMProxy()


async def audio_processor():
    while True:
        audio_bytes = await audio_queue.get()
        result = await asyncio.get_running_loop().run_in_executor(
            None, transcription_proxy.transcribe, audio_bytes
        )

        if result == TranscribeEvent.INPUT_FINISHED:
            user_input = " ".join(transcription_proxy.get_current_text())
            stream = llm.stream(user_input)
            for seg in stream:
                print(seg, flush=True, end="")
            transcription_proxy.clear_current_text()
            print("Finished!")


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
