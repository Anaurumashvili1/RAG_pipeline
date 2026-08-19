"""OpenAI-compatible chat client.

Works unchanged against your Open-WebUI instance, a local llama.cpp / vLLM /
Ollama server, or api.openai.com - only LLM_BASE_URL and LLM_API_KEY change.

The API key is read from .env. It is never written in code.
"""

from __future__ import annotations

import time

from openai import OpenAI

from .config import LLMCfg


class ChatClient:
    def __init__(self, cfg: LLMCfg, max_retries: int = 3):
        if not cfg.base_url:
            raise ValueError("LLM_BASE_URL is not set. Copy .env.example to .env.")
        if not cfg.api_key:
            raise ValueError("LLM_API_KEY is not set. Copy .env.example to .env.")

        self.cfg = cfg
        self.max_retries = max_retries
        self._client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)

    def complete(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                completion = self._client.chat.completions.create(
                    model=self.cfg.model,
                    messages=messages,
                    temperature=self.cfg.temperature if temperature is None else temperature,
                    max_tokens=max_tokens or self.cfg.max_tokens,
                    timeout=self.cfg.timeout,
                )
                return (completion.choices[0].message.content or "").strip()
            except Exception as e:  # noqa: BLE001 - surface the real error after retries
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)

        raise RuntimeError(
            f"LLM call failed after {self.max_retries} attempts: {last_error}"
        ) from last_error

    def list_models(self) -> list[str]:
        """Handy when the endpoint 404s on an unknown model name."""
        try:
            return [m.id for m in self._client.models.list().data]
        except Exception as e:  # noqa: BLE001
            return [f"(could not list models: {e})"]
