from __future__ import annotations

import json
import re
from typing import Iterator

import httpx

from story_for_you.exceptions import (
    LLMConnectionError,
    LLMResponseError,
    LLMTimeoutError,
)
from story_for_you.llm.base import LLMProvider, LLMResponse


class OllamaProvider(LLMProvider):
    """Default LLM provider targeting the Ollama runtime."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        *,
        timeout: float | httpx.Timeout | None = 300.0,
        options: dict | None = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.options = options or {}
        self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def generate(self, prompt: str, system: str = "", options: dict | None = None) -> LLMResponse:
        """Synchronously generate a response via Ollama."""
        payload = self._build_payload(prompt, system, stream=False, call_options=options)
        try:
            response = self._client.post("/api/generate", json=payload)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"Ollama request timed out: {exc}") from exc
        except httpx.ConnectError as exc:
            raise LLMConnectionError(f"Failed to connect to Ollama at {self.base_url}: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            raise LLMResponseError(f"Ollama returned error status {exc.response.status_code}: {exc}") from exc
        except httpx.HTTPError as exc:
            raise LLMConnectionError(f"Ollama request failed: {exc}") from exc

        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise LLMResponseError(f"Invalid JSON response from Ollama: {exc}") from exc

        content = data.get("response", "").strip()
        content = self._strip_thinking(content)
        tokens_used = data.get("eval_count") or data.get("prompt_eval_count") or 0
        return LLMResponse(content=content, tokens_used=tokens_used)

    def generate_stream(self, prompt: str, system: str = "", options: dict | None = None) -> Iterator[str]:
        """Stream a response via Ollama's API."""
        payload = self._build_payload(prompt, system, stream=True, call_options=options)
        is_qwen = "qwen" in self.model.lower()
        try:
            with self._client.stream(
                "POST",
                "/api/generate",
                json=payload,
            ) as response:
                response.raise_for_status()
                buf = [] if is_qwen else None
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    chunk = data.get("response")
                    if chunk:
                        if buf is not None:
                            buf.append(chunk)
                        else:
                            yield chunk
                if buf is not None:
                    cleaned = self._strip_thinking("".join(buf))
                    if cleaned:
                        yield cleaned
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"Ollama stream timed out: {exc}") from exc
        except httpx.ConnectError as exc:
            raise LLMConnectionError(f"Failed to connect to Ollama at {self.base_url}: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            raise LLMResponseError(f"Ollama returned error status {exc.response.status_code}: {exc}") from exc
        except httpx.HTTPError as exc:
            raise LLMConnectionError(f"Ollama stream failed: {exc}") from exc

    def close(self) -> None:
        """Dispose of the shared HTTP client."""
        self._client.close()

    def _strip_thinking(self, text: str) -> str:
        """Remove ``<think>...</think>`` blocks emitted by Qwen models."""
        if "qwen" not in self.model.lower():
            return text
        return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()

    def _build_payload(self, prompt: str, system: str, stream: bool, call_options: dict | None = None) -> dict:
        merged_options = dict(self.options)
        if call_options:
            merged_options.update({key: value for key, value in call_options.items() if value is not None})

        # Pop no_think before sending to Ollama; append directive to prompt for Qwen models.
        no_think = merged_options.pop("no_think", False)
        if no_think and "qwen" in self.model.lower():
            prompt = prompt + "\n\n/no_think"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system or None,
            "stream": stream,
        }
        if merged_options:
            payload["options"] = merged_options
        # Drop None values to keep payload compact.
        return {key: value for key, value in payload.items() if value is not None}
