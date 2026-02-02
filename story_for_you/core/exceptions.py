"""Custom exceptions for the story_for_you package.

Provides a hierarchy of domain-specific exceptions for better error handling.
"""

from __future__ import annotations

__all__ = [
    "StoryForYouError",
    "LLMError",
    "LLMConnectionError",
    "LLMTimeoutError",
    "LLMResponseError",
    "GenerationError",
    "AnalysisError",
    "ConfigurationError",
    "TemplateError",
]


class StoryForYouError(Exception):
    """Base exception for all story_for_you errors."""

    pass


class LLMError(StoryForYouError):
    """Base exception for LLM-related errors."""

    pass


class LLMConnectionError(LLMError):
    """Raised when connection to LLM provider fails."""

    pass


class LLMTimeoutError(LLMError):
    """Raised when LLM request times out."""

    pass


class LLMResponseError(LLMError):
    """Raised when LLM returns invalid or unexpected response."""

    pass


class GenerationError(StoryForYouError):
    """Raised when content generation fails."""

    pass


class AnalysisError(StoryForYouError):
    """Raised when story analysis fails."""

    pass


class ConfigurationError(StoryForYouError):
    """Raised when configuration is invalid."""

    pass


class TemplateError(StoryForYouError):
    """Raised when template loading or rendering fails."""

    pass
