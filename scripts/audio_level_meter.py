"""Quick manual check: prints a live level bar for mic input.

Run: python scripts/audio_level_meter.py
"""

import audioop
import sys

sys.path.insert(0, "src")

from infrastructure.audio.input.pyaudio import PyAudioInput

BAR_WIDTH = 50
MAX_RMS = 8000  # rough ceiling for int16 mic input, tune to taste


def level_bar(rms: int) -> str:
    filled = min(BAR_WIDTH, int((rms / MAX_RMS) * BAR_WIDTH))
    return "#" * filled + "-" * (BAR_WIDTH - filled)


def main() -> None:
    mic = PyAudioInput()
    try:
        for chunk in mic.start_microphone():
            rms = audioop.rms(chunk, 2)  # 2 bytes per sample (paInt16)
            print(f"\r[{level_bar(rms)}] {rms:5d}", end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        mic.kill_microphone()
        mic.close_audio()
        print()


if __name__ == "__main__":
    main()
