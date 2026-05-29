"""Prompting utilities for core module.

Provides template loading and style formatting functions for content generation.
"""

from __future__ import annotations

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict

from story_for_you.utils.prompting import TemplateLoader, fill_template

if TYPE_CHECKING:
    from story_for_you.analysis.context import WritingStyle

__all__ = [
    "load_template",
    "fill_template",
    "format_context_sections",
    "format_style_guide",
    "format_style_samples",
    "format_style_constraints",
]

_TEMPLATE_DIR = Path(__file__).with_name("prompt_templates")
_loader = TemplateLoader(_TEMPLATE_DIR)


def load_template(name: str) -> str:
    """Return the raw template text with simple caching."""
    return _loader.load(name)


def format_context_sections(sections: Dict[str, str]) -> str:
    """Turn context sections into a normalized markdown block."""
    chunks: list[str] = []
    for key, value in sections.items():
        if not value:
            continue
        title = key.replace("_", " ").title()
        chunks.append(f"## {title}\n{value.strip()}")
    return "\n\n".join(chunks)


def format_style_guide(style: WritingStyle | None) -> str:
    """Format structured style guide for prompt injection.

    Args:
        style: WritingStyle object or None.

    Returns:
        Structured style guide string with all attributes, or a neutral placeholder.
    """
    if style is None:
        return "(无风格信息，请保持中性文学风格)"

    sections = []

    # 句式结构
    avg_len = getattr(style, "avg_sentence_length", 0)
    if avg_len:
        sections.append(f"- 平均句长: {avg_len}字")
    variety = getattr(style, "sentence_variety", "")
    if variety:
        sections.append(f"- 句式变化: {variety}")
    density = getattr(style, "paragraph_density", "")
    if density:
        sections.append(f"- 段落密度: {density}")

    # 用词风格
    register = getattr(style, "register", "")
    if register:
        sections.append(f"- 语体: {register}")
    char_words = getattr(style, "characteristic_words", [])
    if char_words:
        sections.append(f"- 特征词汇: {', '.join(char_words)}")
    idiom_freq = getattr(style, "idiom_frequency", "")
    if idiom_freq:
        sections.append(f"- 成语使用: {idiom_freq}")

    # 修辞手法
    metaphor = getattr(style, "metaphor_style", "")
    if metaphor:
        sections.append(f"- 比喻风格: {metaphor}")
    desc_focus = getattr(style, "description_focus", [])
    if desc_focus:
        sections.append(f"- 描写重点: {', '.join(desc_focus)}")
    parallelism = getattr(style, "parallelism_use", "")
    if parallelism:
        sections.append(f"- 排比使用: {parallelism}")

    # 叙事语气
    tone = getattr(style, "tone_markers", [])
    if tone:
        sections.append(f"- 语气词: {', '.join(tone)}")
    narrator = getattr(style, "narrator_style", "")
    if narrator:
        sections.append(f"- 叙述者风格: {narrator}")

    # 摘要
    summary = getattr(style, "style_summary", "")
    if summary:
        sections.append(f"\n风格总述: {summary}")

    return "\n".join(sections) if sections else "(无风格摘要，请保持中性文学风格)"


def format_style_samples(style: WritingStyle | None, max_samples: int = 3) -> str:
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


def format_style_constraints(style: WritingStyle | None) -> str:
    """Render style constraints from model-extracted style attributes.

    Args:
        style: WritingStyle object or None.

    Returns:
        Prompt text that asks the LLM to apply the style model semantically.
    """
    if style is None:
        return ""

    register = getattr(style, "register", "mixed")
    desc_focus = [item for item in getattr(style, "description_focus", []) if item]
    metaphor_style = getattr(style, "metaphor_style", "").strip()
    tone_markers = getattr(style, "tone_markers", [])
    style_summary = getattr(style, "style_summary", "").strip()

    lines = [
        "## 写作约束",
        f"- 语体判断：{register}",
        "- 由模型根据风格摘要和示例自行判断何为不合风格的句式、词汇、节奏与叙述姿态。",
        "- 续写必须像正文自然延伸，不要出现任务说明、审稿意见或自我解释。",
    ]
    if style_summary:
        lines.append(f"- 风格摘要：{style_summary}")
    if desc_focus:
        lines.append("- 描写重点：" + "、".join(desc_focus))
    if metaphor_style:
        lines.append(f"- 比喻风格：{metaphor_style}")
    if tone_markers:
        lines.append("- 语气词参考：" + "、".join(tone_markers[:4]))
    return "\n".join(lines)
