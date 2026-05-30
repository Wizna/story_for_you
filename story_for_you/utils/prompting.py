"""Shared prompting utilities for template loading and filling.

This module provides the core template handling functionality used by both
analysis/prompting.py and core/prompting.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

__all__ = [
    "CacheablePrompt",
    "TemplateLoader",
    "cache_prompt",
    "fill_template",
    "clamp_text_middle",
    "build_cacheable_prompt",
    "PLACEHOLDER_PATTERN",
    "SNIPPET_EXCERPT_LEN",
]

SNIPPET_EXCERPT_LEN = 280

PLACEHOLDER_PATTERN = re.compile(r"\{\{(\w+)\}\}")


@dataclass(frozen=True)
class CacheablePrompt:
    """Prompt split into a stable prefix and variable task suffix."""

    prefix: str
    task: str

    def render(self) -> str:
        if not self.prefix:
            return self.task
        if not self.task:
            return self.prefix
        return f"{self.prefix}\n\n{self.task}"


def cache_prompt(task: str, prefix: str = "") -> CacheablePrompt:
    """Build the only prompt shape accepted by LLM providers."""
    return CacheablePrompt(prefix=prefix.strip(), task=task.strip())


class TemplateLoader:
    """Cached template loader for a specific template directory."""

    def __init__(self, template_dir: Path) -> None:
        self._template_dir = template_dir
        self._cache: dict[str, str] = {}

    def load(self, name: str) -> str:
        """Load and cache a template by name."""
        if name in self._cache:
            return self._cache[name]
        path = self._template_dir / f"{name}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Prompt template '{name}' is missing at {path}")
        content = path.read_text(encoding="utf-8")
        self._cache[name] = content
        return content

    def clear_cache(self) -> None:
        """Clear the template cache."""
        self._cache.clear()


def fill_template(template: str, **placeholders: str) -> str:
    """Replace `{{token}}` placeholders with provided values."""

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        return str(placeholders.get(key, ""))

    return PLACEHOLDER_PATTERN.sub(_replace, template)


def build_cacheable_prompt(
    prefix: str,
    template: str,
    *,
    prefix_placeholder: str,
    prefix_reference: str = "见前一条 user 消息中的完整文本。",
    **placeholders: str,
) -> CacheablePrompt:
    """Render a prompt while moving a long stable placeholder into the prefix.

    DeepSeek's context cache matches complete request prefixes. Keeping the
    repeated long text as the first user message lets later task variants reuse
    that prefix without requiring all templates to share identical instructions.
    """
    normalized_prefix = prefix.strip()
    rendered = fill_template(
        template,
        **{
            **placeholders,
            prefix_placeholder: prefix_reference,
        },
    ).strip()
    return cache_prompt(prefix=normalized_prefix, task=rendered)


def clamp_text_middle(text: str, max_chars: int, head_ratio: float = 0.7) -> str:
    """Return text truncated to `max_chars`, preserving the head and tail."""
    stripped = text.strip()
    if max_chars <= 0 or len(stripped) <= max_chars:
        return stripped
    head = max(1, int(max_chars * head_ratio))
    tail = max(1, max_chars - head)
    head_slice = stripped[:head].rstrip()
    tail_slice = stripped[-tail:].lstrip()
    marker = "\n<<<TRUNCATED>>>\n"
    budget = max_chars - len(marker)
    if budget <= 0:
        return head_slice[:max_chars]
    head_budget = min(len(head_slice), int(budget * head_ratio))
    tail_budget = max(0, budget - head_budget)
    return head_slice[:head_budget] + marker + tail_slice[-tail_budget:]


# Legacy support: module-level cached loader factory
@lru_cache(maxsize=None)
def _get_loader(template_dir: str) -> TemplateLoader:
    """Get or create a cached TemplateLoader for a directory."""
    return TemplateLoader(Path(template_dir))


def load_template_from_dir(template_dir: Path, name: str) -> str:
    """Load a template from a specific directory with caching."""
    return _get_loader(str(template_dir)).load(name)
