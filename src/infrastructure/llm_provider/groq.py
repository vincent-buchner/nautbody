import os

from groq import Groq
from groq.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel


class GroqModelConfig(BaseModel):
    model: str
    temperature: float
    max_completion_tokens: int
    top_p: int
    stop: list[str] | None


class GroqProvider:
    def __init__(self, system_prompt: str, model_config: GroqModelConfig) -> None:
        self._client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self._system_prompt = system_prompt
        self._model_config = model_config.model_dump()
        self._messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": self._system_prompt},
        ]

    def generate_response(self, user_input: str) -> str | None:

        self._messages.append({"role": "user", "content": user_input})

        response = self._client.chat.completions.create(
            messages=self._messages,
            **self._model_config,
        )

        return response.choices[0].message.content
