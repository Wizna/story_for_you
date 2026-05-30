from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Iterator

from story_for_you.llm.base import LLMProvider, LLMResponse
from story_for_you.utils.prompting import CacheablePrompt

_TELEMETRY_PREFIX = "_sfy_"


@dataclass(slots=True)
class LLMPriceCard:
    input_cache_hit: float
    input_cache_miss: float
    output: float
    currency: str = "USD"


_DEEPSEEK_PRICES: dict[str, LLMPriceCard] = {
    "deepseek-v4-pro": LLMPriceCard(
        input_cache_hit=0.003625,
        input_cache_miss=0.435,
        output=0.87,
    ),
    "deepseek-v4-flash": LLMPriceCard(
        input_cache_hit=0.0028,
        input_cache_miss=0.14,
        output=0.28,
    ),
}


@dataclass(slots=True)
class LLMTelemetryState:
    label: str = "LLM"
    total_expected: int | None = None
    request_index: int = 0
    request_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_prompt_tokens: int = 0
    estimated_cost: float = 0.0


class TelemetryLLMProvider(LLMProvider):
    """Wrap an LLM provider with user-visible progress and cost reporting."""

    def __init__(self, provider: LLMProvider, emit: Callable[[str], None]):
        self._provider = provider
        self._emit = emit
        self._state = LLMTelemetryState()
        self.model = getattr(provider, "model", "unknown")
        self.base_url = getattr(provider, "base_url", "")

    def set_plan(self, label: str, total_expected: int | None = None) -> None:
        self._state.label = label
        self._state.total_expected = total_expected
        self._state.request_index = 0
        self._state.request_count = 0
        self._state.prompt_tokens = 0
        self._state.completion_tokens = 0
        self._state.cached_prompt_tokens = 0
        self._state.estimated_cost = 0.0

    def generate(
        self,
        prompt: CacheablePrompt,
        system: str = "",
        options: dict | None = None,
    ) -> LLMResponse:
        metadata, call_options = self._split_options(options)
        self._state.request_index += 1
        attempt = metadata.get("attempt")
        max_attempts = metadata.get("max_attempts")
        phase = metadata.get("phase") or self._state.label
        step = metadata.get("step")
        note = metadata.get("note")
        prompt_head = self._summarize_prompt(prompt)

        request_label = self._format_request_label(self._state.request_index)
        suffix = self._format_attempt_suffix(attempt, max_attempts)
        self._emit(f"[LLM {request_label}] {phase}{step or ''}{suffix}")
        if note:
            self._emit(f"[LLM {request_label}] {note}")
        if prompt_head:
            self._emit(f"[LLM {request_label}] prompt: {prompt_head}")
        if system.strip():
            self._emit(f"[LLM {request_label}] system: {self._summarize_text(system)}")

        start = time.perf_counter()
        try:
            response = self._provider.generate(prompt=prompt, system=system, options=call_options)
        except Exception as exc:
            elapsed = time.perf_counter() - start
            self._emit(f"[LLM {request_label}] failed after {elapsed:.1f}s: {exc}")
            raise
        elapsed = time.perf_counter() - start

        cost = self._estimate_cost(response)
        self._state.request_count += 1
        self._state.prompt_tokens += response.prompt_tokens
        self._state.completion_tokens += response.completion_tokens
        self._state.cached_prompt_tokens += response.cache_hit_prompt_tokens
        self._state.estimated_cost += cost

        self._emit(
            f"[LLM {request_label}] done in {elapsed:.1f}s | "
            f"in={response.prompt_tokens} out={response.completion_tokens} "
            f"total={response.tokens_used} cache={response.cache_hit_prompt_tokens} "
            f"cache_rate={self._format_cache_rate(response)} "
            f"cost={self._format_cost(cost)} "
            f"cumulative={self._format_cost(self._state.estimated_cost)}"
        )
        if self._state.total_expected:
            remaining = max(self._state.total_expected - self._state.request_index, 0)
            self._emit(f"[LLM {request_label}] remaining~{remaining}")
        return response

    def generate_stream(
        self,
        prompt: CacheablePrompt,
        system: str = "",
        options: dict | None = None,
    ) -> Iterator[str]:
        metadata, call_options = self._split_options(options)
        self._state.request_index += 1
        request_label = self._format_request_label(self._state.request_index)
        phase = metadata.get("phase") or self._state.label
        step = metadata.get("step")
        self._emit(f"[LLM {request_label}] {phase}{step or ''} (stream)")
        return self._provider.generate_stream(prompt=prompt, system=system, options=call_options)

    def close(self) -> None:
        self._provider.close()

    def __getattr__(self, name: str):
        return getattr(self._provider, name)

    def _split_options(self, options: dict | None) -> tuple[dict[str, object], dict | None]:
        if not options:
            return {}, None
        metadata: dict[str, object] = {}
        call_options: dict[str, object] = {}
        for key, value in options.items():
            if key.startswith(_TELEMETRY_PREFIX):
                metadata[key.removeprefix(_TELEMETRY_PREFIX)] = value
                continue
            call_options[key] = value
        return metadata, call_options or None

    def _summarize_prompt(self, prompt: CacheablePrompt) -> str:
        task_head = self._summarize_text(prompt.task)
        if prompt.prefix:
            prefix_head = self._summarize_text(prompt.prefix)
            text = f"cache-prefix: {prefix_head} / task: {task_head}"
        else:
            text = task_head
        return text if len(text) <= 180 else text[:177] + "..."

    def _summarize_text(self, text: str) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return "(empty)"
        head = " / ".join(lines[:2])
        return head if len(head) <= 220 else head[:217] + "..."

    def _format_attempt_suffix(self, attempt: object, max_attempts: object) -> str:
        if attempt is None and max_attempts is None:
            return ""
        if attempt is None:
            return f" attempt=?/{max_attempts}"
        if max_attempts is None:
            return f" attempt={attempt}"
        return f" attempt={attempt}/{max_attempts}"

    def _format_request_label(self, index: int) -> str:
        if self._state.total_expected:
            return f"{index}/{self._state.total_expected}"
        return str(index)

    def _format_cost(self, value: float) -> str:
        if value <= 0:
            return "n/a"
        return f"${value:.6f}"

    def _format_cache_rate(self, response: LLMResponse) -> str:
        total = response.cache_hit_prompt_tokens + response.cache_miss_prompt_tokens
        if total <= 0 and response.prompt_tokens > 0:
            total = response.prompt_tokens
        if total <= 0:
            return "n/a"
        return f"{(response.cache_hit_prompt_tokens / total) * 100:.1f}%"

    def _estimate_cost(self, response: LLMResponse) -> float:
        model = getattr(self._provider, "model", "").lower()
        prices = _DEEPSEEK_PRICES.get(model)
        if not prices:
            return 0.0
        if response.cache_hit_prompt_tokens or response.cache_miss_prompt_tokens:
            prompt_cost = (
                response.cache_hit_prompt_tokens * prices.input_cache_hit
                + response.cache_miss_prompt_tokens * prices.input_cache_miss
            ) / 1_000_000
        else:
            prompt_cost = (response.prompt_tokens * prices.input_cache_miss) / 1_000_000
        output_cost = (response.completion_tokens * prices.output) / 1_000_000
        return prompt_cost + output_cost


def telemetry_options(options: dict | None = None, **metadata: object) -> dict:
    payload = dict(options or {})
    for key, value in metadata.items():
        if value is not None:
            payload[f"{_TELEMETRY_PREFIX}{key}"] = value
    return payload
