from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Self

from story_for_you.utils.prompting import CacheablePrompt


@dataclass
class LLMResponse:
    content: str
    tokens_used: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit_prompt_tokens: int = 0
    cache_miss_prompt_tokens: int = 0


class LLMProvider(ABC):
    """Abstract base provider for language model calls."""

    @abstractmethod
    def generate(
        self,
        prompt: CacheablePrompt,
        system: str = "",
        options: dict | None = None,
    ) -> LLMResponse:
        """Generate a response for the provided prompt."""
        raise NotImplementedError

    @abstractmethod
    def generate_stream(
        self,
        prompt: CacheablePrompt,
        system: str = "",
        options: dict | None = None,
    ) -> Iterator[str]:
        """Yield a streaming response iterator."""
        raise NotImplementedError

    def close(self) -> None:
        """Release any resources held by the provider.

        Subclasses should override this to clean up HTTP clients, connections, etc.
        """
        pass

    def __enter__(self) -> Self:
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context manager and release resources."""
        self.close()
