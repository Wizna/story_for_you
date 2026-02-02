"""Prompting utilities for analysis module.

Provides budget-aware prompt rendering with automatic text truncation.
"""

from __future__ import annotations

from pathlib import Path

from story_for_you.utils.prompting import (
    TemplateLoader,
    fill_template,
    clamp_text_middle,
)

__all__ = ["load_template", "fill_template", "render_prompt_with_budget", "clamp_text_middle"]

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
