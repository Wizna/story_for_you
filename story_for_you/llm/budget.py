from __future__ import annotations

from collections.abc import Iterator

from story_for_you.exceptions import LLMResponseError
from story_for_you.llm.base import LLMProvider, LLMResponse
from story_for_you.utils.prompting import CacheablePrompt


class BudgetedLLMProvider(LLMProvider):
    """Use UTF-8 bytes as a conservative token upper bound when no tokenizer is available."""

    def __init__(
        self, provider: LLMProvider, context_window: int, max_output_tokens: int
    ):
        self.provider = provider
        self.context_window = context_window
        self.max_output_tokens = max_output_tokens
        if context_window <= 0 or max_output_tokens <= 0:
            raise ValueError("Context and output budgets must be positive.")

    def check(self, prompt: CacheablePrompt, system: str = "") -> None:
        input_upper_bound = len((system + prompt.render()).encode("utf-8")) + 256
        if input_upper_bound + self.max_output_tokens > self.context_window:
            raise LLMResponseError(
                "Continuation prompt exceeds the conservative context budget including output reserve "
                f"({input_upper_bound} + {self.max_output_tokens} > {self.context_window}). "
                "Use fewer continuation units or a larger context window; story evidence was not truncated."
            )

    def generate(
        self, prompt: CacheablePrompt, system: str = "", options: dict | None = None
    ) -> LLMResponse:
        self.check(prompt, system)
        call_options = {**(options or {}), "max_tokens": self.max_output_tokens}
        response = self.provider.generate(prompt, system, call_options)
        response.ensure_complete()
        return response

    def generate_stream(
        self, prompt: CacheablePrompt, system: str = "", options: dict | None = None
    ) -> Iterator[str]:
        self.check(prompt, system)
        call_options = {**(options or {}), "max_tokens": self.max_output_tokens}
        yield from self.provider.generate_stream(prompt, system, call_options)

    def close(self) -> None:
        self.provider.close()
