"""Prompting utilities for analysis module.

Provides budget-aware prompt rendering with automatic text truncation.
"""

from __future__ import annotations

from pathlib import Path

from story_for_you.utils.prompting import (
    CacheablePrompt,
    TemplateLoader,
    build_cacheable_prompt,
    cache_prompt,
    fill_template,
    clamp_text_middle,
)

__all__ = [
    "CacheablePrompt",
    "load_template",
    "cache_prompt",
    "fill_template",
    "build_cacheable_prompt",
    "render_prompt_with_budget",
    "render_cacheable_prompt_with_budget",
    "clamp_text_middle",
]

_TEMPLATE_DIR = Path(__file__).with_name("prompt_templates")
_loader = TemplateLoader(_TEMPLATE_DIR)


def load_template(name: str) -> str:
    """Return the raw template text with simple caching."""
    return _loader.load(name)


def render_prompt_with_budget(
    template: str,
    *,
    budget: int | None,
    text_key: str,
    text_value: str,
    min_text: int = 256,
    head_ratio: float = 0.7,
    **placeholders: str,
) -> tuple[str, bool]:
    """Render a prompt and clamp the chapter text if the prompt exceeds `budget`."""
    placeholders = dict(placeholders)
    placeholders[text_key] = text_value
    prompt = fill_template(template, **placeholders)
    if budget is None or budget <= 0 or len(prompt) <= budget:
        return prompt, False
    overhead = len(prompt) - len(text_value)
    allowance = max(min_text, budget - overhead)
    if allowance >= len(text_value):
        return prompt, False
    trimmed_text = clamp_text_middle(text_value, allowance, head_ratio=head_ratio)
    placeholders[text_key] = trimmed_text
    prompt = fill_template(template, **placeholders)
    return prompt, True


def render_cacheable_prompt_with_budget(
    template: str,
    *,
    budget: int | None,
    prefix_key: str,
    prefix_value: str,
    min_text: int = 256,
    head_ratio: float = 0.7,
    **placeholders: str,
) -> tuple[CacheablePrompt, bool]:
    """Render a cache-friendly prompt and clamp the stable prefix if needed."""
    cacheable = build_cacheable_prompt(
        prefix_value,
        template,
        prefix_placeholder=prefix_key,
        **placeholders,
    )
    rendered = cacheable.render()
    if budget is None or budget <= 0 or len(rendered) <= budget:
        return cacheable, False

    overhead = len(rendered) - len(cacheable.prefix)
    allowance = max(min_text, budget - overhead)
    if allowance >= len(cacheable.prefix):
        return cacheable, False
    trimmed_prefix = clamp_text_middle(cacheable.prefix, allowance, head_ratio=head_ratio)
    return (
        build_cacheable_prompt(
            trimmed_prefix,
            template,
            prefix_placeholder=prefix_key,
            **placeholders,
        ),
        True,
    )
