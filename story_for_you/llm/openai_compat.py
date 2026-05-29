from __future__ import annotations

import json
from typing import Iterator

import httpx

from story_for_you.exceptions import (
    LLMConnectionError,
    LLMResponseError,
    LLMTimeoutError,
)
from story_for_you.llm.base import LLMProvider, LLMResponse


class OpenAICompatibleProvider(LLMProvider):
    """LLM provider for any OpenAI-compatible Chat Completions API.

    Works with DeepSeek, OpenAI, Groq, Together, Mistral, and other services
    that implement the ``/v1/chat/completions`` endpoint.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        *,
        api_key: str = "",
        timeout: float | httpx.Timeout | None = 300.0,
        options: dict | None = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.options = options or {}

        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout,
            headers=headers,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(
        self, prompt: str, system: str = "", options: dict | None = None
    ) -> LLMResponse:
        payload = self._build_payload(prompt, system, stream=False, call_options=options)
        try:
            response = self._client.post(self._chat_completions_path(), json=payload)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"Request timed out: {exc}") from exc
        except httpx.ConnectError as exc:
            raise LLMConnectionError(
                f"Failed to connect to {self.base_url}: {exc}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise self._map_status_error(exc) from exc
        except httpx.HTTPError as exc:
            raise LLMConnectionError(f"Request failed: {exc}") from exc

        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise LLMResponseError(f"Invalid JSON response: {exc}") from exc

        try:
            content = data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMResponseError(f"Unexpected response structure: {exc}") from exc

        tokens_used = 0
        usage = data.get("usage")
        if usage:
            tokens_used = usage.get("total_tokens", 0)

        return LLMResponse(content=content, tokens_used=tokens_used)

    def generate_stream(
        self, prompt: str, system: str = "", options: dict | None = None
    ) -> Iterator[str]:
        payload = self._build_payload(prompt, system, stream=True, call_options=options)
        try:
            with self._client.stream(
                "POST",
                self._chat_completions_path(),
                json=payload,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    delta = data.get("choices", [{}])[0].get("delta", {})
                    chunk = delta.get("content")
                    if chunk:
                        yield chunk
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"Stream timed out: {exc}") from exc
        except httpx.ConnectError as exc:
            raise LLMConnectionError(
                f"Failed to connect to {self.base_url}: {exc}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise self._map_status_error(exc) from exc
        except httpx.HTTPError as exc:
            raise LLMConnectionError(f"Stream failed: {exc}") from exc

    def close(self) -> None:
        self._client.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_payload(
        self,
        prompt: str,
        system: str,
        stream: bool,
        call_options: dict | None = None,
    ) -> dict:
        merged = dict(self.options)
        if call_options:
            merged.update(
                {k: v for k, v in call_options.items() if v is not None}
            )

        # Pop no_think before building payload; translate it for providers that
        # expose an explicit thinking switch.
        no_think = merged.pop("no_think", False)
        if no_think and "qwen" in self.model.lower():
            prompt = prompt + "\n\n/no_think"
        elif no_think and self._is_deepseek_endpoint():
            merged.setdefault("thinking", {"type": "disabled"})

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
        }

        # Map well-known option keys to top-level API parameters.
        _TOP_LEVEL_KEYS = {
            "temperature",
            "max_tokens",
            "seed",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "thinking",
            "response_format",
        }
        for key in _TOP_LEVEL_KEYS:
            if key in merged:
                payload[key] = merged.pop(key)

        return payload

    def _is_deepseek_endpoint(self) -> bool:
        return "deepseek.com" in self.base_url.lower()

    def _chat_completions_path(self) -> str:
        if self._is_deepseek_endpoint():
            return "/chat/completions"
        return "/v1/chat/completions"

    @staticmethod
    def _map_status_error(exc: httpx.HTTPStatusError) -> Exception:
        status = exc.response.status_code
        if status == 401:
            return LLMConnectionError(f"Authentication failed (401): {exc}")
        if status == 429:
            return LLMTimeoutError(f"Rate limited (429): {exc}")
        return LLMResponseError(f"HTTP error {status}: {exc}")
