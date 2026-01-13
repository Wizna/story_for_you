from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Dict

__all__ = [
    "load_template",
    "fill_template",
    "format_context_sections",
    "format_style_guide",
    "format_style_samples",
]

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


def format_context_sections(sections: Dict[str, str]) -> str:
    """Turn context sections into a normalized markdown block."""
    chunks: list[str] = []
    for key, value in sections.items():
        if not value:
            continue
        title = key.replace("_", " ").title()
        chunks.append(f"## {title}\n{value.strip()}")
    return "\n\n".join(chunks)


def format_style_guide(style) -> str:
    """Format style summary for prompt injection.

    Args:
        style: WritingStyle object or None.

    Returns:
        Style summary string or a neutral placeholder.
    """
    if style is None:
        return "(无风格信息，请保持中性文学风格)"
    summary = getattr(style, "style_summary", "")
    if not summary:
        return "(无风格摘要，请保持中性文学风格)"
    return summary


def format_style_samples(style, max_samples: int = 2) -> str:
    """Format representative style samples for prompt injection.

    Args:
        style: WritingStyle object or None.
        max_samples: Maximum number of samples to include.

    Returns:
        Formatted sample snippets or a placeholder.
    """
    if style is None:
        return "(无示例片段)"
    samples = getattr(style, "representative_samples", [])
    if not samples:
        return "(无示例片段)"
    lines = []
    for sample in samples[:max_samples]:
        content = getattr(sample, "content", "")
        if content:
            lines.append(f"「{content}」")
    return "\n".join(lines) if lines else "(无示例片段)"
