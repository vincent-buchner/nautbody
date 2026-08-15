from collections.abc import Generator
from typing import Protocol


class IAudioInput(Protocol):
    def start_microphone(self) -> Generator[bytes, None, None]: ...

    def kill_microphone(self) -> None: ...

    def close_audio(self) -> None: ...
