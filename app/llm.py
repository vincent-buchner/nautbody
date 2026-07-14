import os
from typing import Generator
from groq import Groq


class LLMProxy:
    def __init__(self) -> None:
        self._client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def stream(self, user_input: str) -> Generator[str | None]:
        stream = self._client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are Leon from Mr. Robot"},
                {"role": "user", "content": user_input},
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.5,
            max_completion_tokens=1024,
            top_p=1,
            stop=None,
            stream=True,
        )

        for delta in stream:
            yield delta.choices[0].delta.content
