from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    content: str
    tokens_used: int


class LLMProvider(ABC):
    """Abstract base provider for language model calls."""

    @abstractmethod
    def generate(self, prompt: str, system: str = "") -> LLMResponse:
        """Generate a response for the provided prompt."""
        raise NotImplementedError

    @abstractmethod
    def generate_stream(self, prompt: str, system: str = ""):
        """Yield a streaming response iterator."""
        raise NotImplementedError
