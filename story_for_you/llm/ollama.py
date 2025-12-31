from __future__ import annotations

import json
from typing import Iterator

import httpx

from story_for_you.llm.base import LLMProvider, LLMResponse


class OllamaProvider(LLMProvider):
    """Default LLM provider targeting the Ollama runtime."""

    def __init__(
        self,
        model: str = "qwen2.5:7b-instruct",
        base_url: str = "http://localhost:11434",
        *,
        timeout: float = 120.0,
        options: dict | None = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.options = options or {}
        self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def generate(self, prompt: str, system: str = "") -> LLMResponse:
        """Synchronously generate a response via Ollama."""
        payload = self._build_payload(prompt, system, stream=False)
        try:
            response = self._client.post("/api/generate", json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc
        data = response.json()
        content = data.get("response", "").strip()
        tokens_used = data.get("eval_count") or data.get("prompt_eval_count") or 0
        return LLMResponse(content=content, tokens_used=tokens_used)

    def generate_stream(self, prompt: str, system: str = "") -> Iterator[str]:
        """Stream a response via Ollama's API."""
        payload = self._build_payload(prompt, system, stream=True)
        with httpx.stream(
            "POST",
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                chunk = data.get("response")
                if chunk:
                    yield chunk

    def close(self) -> None:
        """Dispose of the shared HTTP client."""
        self._client.close()

    def _build_payload(self, prompt: str, system: str, stream: bool) -> dict:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system or None,
            "stream": stream,
        }
        if self.options:
            payload["options"] = self.options
        # Drop None values to keep payload compact.
        return {key: value for key, value in payload.items() if value is not None}
