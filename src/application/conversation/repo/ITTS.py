from collections.abc import Generator
from typing import Protocol


class ITTS(Protocol):
    def generate_audio(self, text: str) -> Generator[bytes, None, list[bytes]]: ...
