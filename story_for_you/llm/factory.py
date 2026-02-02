"""Factory functions for creating LLM provider instances.

Provides a centralized way to construct LLM providers from configuration settings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict

if TYPE_CHECKING:
    from story_for_you.config.settings import Settings
    from story_for_you.llm.base import LLMProvider

__all__ = [
    "build_llm",
    "register_provider",
    "get_provider_names",
]

# Provider registry: name -> factory function
_PROVIDERS: Dict[str, Callable[[Settings], "LLMProvider"]] = {}


def _build_ollama_provider(settings: Settings) -> "LLMProvider":
    """Build an OllamaProvider from settings."""
    from story_for_you.llm.ollama import OllamaProvider

    options: Dict[str, Any] = {
        "temperature": settings.llm.temperature,
        "seed": settings.llm.seed,
    }
    if settings.llm.max_tokens and settings.llm.max_tokens > 0:
        options["num_ctx"] = settings.llm.max_tokens
    options = {key: value for key, value in options.items() if value is not None}
    return OllamaProvider(
        model=settings.llm.model,
        base_url=settings.llm.base_url,
        timeout=settings.llm.timeout,
        options=options,
    )


# Register default providers
_PROVIDERS["ollama"] = _build_ollama_provider


def register_provider(
    name: str, factory: Callable[["Settings"], "LLMProvider"]
) -> None:
    """Register a new LLM provider factory.

    Args:
        name: The provider name (e.g., "openai", "anthropic").
        factory: A callable that takes Settings and returns an LLMProvider.
    """
    _PROVIDERS[name] = factory


def get_provider_names() -> list[str]:
    """Return a list of registered provider names."""
    return list(_PROVIDERS.keys())


def build_llm(settings: Settings, provider: str | None = None) -> "LLMProvider":
    """Build an LLM provider from settings.

    Args:
        settings: Application settings containing LLM configuration.
        provider: Optional provider name. If not specified, defaults to "ollama".

    Returns:
        A configured LLMProvider instance.

    Raises:
        ValueError: If the specified provider is not registered.
    """
    provider_name = provider or "ollama"
    factory = _PROVIDERS.get(provider_name)
    if factory is None:
        available = ", ".join(sorted(_PROVIDERS.keys()))
        raise ValueError(
            f"Unknown LLM provider '{provider_name}'. Available: {available}"
        )
    return factory(settings)
