from typing import Protocol


class ILLMProvider(Protocol):
    def generate_response(self, user_input: str) -> str | None: ...
