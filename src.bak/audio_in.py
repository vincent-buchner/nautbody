import sounddevice as sd
import asyncio
import numpy as np


class AudioInput:
    def __init__(self, samplerate: int):
        self._global_loop = asyncio.get_running_loop()
        self._samplerate = samplerate

    async def start_mic(self, audio_queue: asyncio.Queue[np.ndarray]):
        def callback(indata: np.ndarray, frames, time, status):
            if status:
                print(f"Status: {status}")

            try:
                asyncio.run_coroutine_threadsafe(
                    audio_queue.put(indata.flatten()),
                    self._global_loop,
                )
            except Exception as e:
                print(f"\nCallback error: {e}")

        with sd.InputStream(
            samplerate=self._samplerate,
            channels=1,
            dtype="float32",
            blocksize=512,
            callback=callback,
        ):
            await asyncio.sleep(np.inf)

    def kill_mic(self):
        pass

    def file_stream(self, file_path: str):
        pass
