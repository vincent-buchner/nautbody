from pathlib import Path

import av
import pytest

AUDIO_FIXTURE = Path(__file__).parent / "test_audio_input.flac"


def decode_pcm16(path: Path, sample_rate: int, channels: int) -> bytes:
    layout = "mono" if channels == 1 else "stereo"
    resampler = av.AudioResampler(format="s16", layout=layout, rate=sample_rate)

    pcm = bytearray()
    with av.open(str(path)) as container:
        for frame in container.decode(audio=0):
            for resampled in resampler.resample(frame):
                pcm += bytes(resampled.planes[0])
    return bytes(pcm)


@pytest.fixture
def pcm_bytes() -> bytes:
    return decode_pcm16(AUDIO_FIXTURE, sample_rate=16_000, channels=1)
