from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

__all__ = ["load_template", "fill_template", "render_prompt_with_budget", "clamp_text_middle"]

_TEMPLATE_DIR = Path(__file__).with_name("prompt_templates")
_PLACEHOLDER_PATTERN = re.compile(r"\{\{(\w+)\}\}")


@lru_cache(maxsize=None)
def _read_template(name: str) -> str:
    path = _TEMPLATE_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template '{name}' is missing at {path}")
    return path.read_text(encoding="utf-8")


def load_template(name: str) -> str:
    """Return the raw template text with simple caching."""
    return _read_template(name)


def fill_template(template: str, **placeholders: str) -> str:
    """Replace `{{token}}` placeholders with provided values."""

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        return str(placeholders.get(key, ""))

    return _PLACEHOLDER_PATTERN.sub(_replace, template)


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
