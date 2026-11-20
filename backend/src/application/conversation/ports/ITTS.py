from typing import Protocol


class ITTS(Protocol):
    def generate_audio(self, text: str) -> bytes: ...
