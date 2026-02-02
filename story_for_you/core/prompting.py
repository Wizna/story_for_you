"""Prompting utilities for core module.

Provides template loading and style formatting functions for content generation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from story_for_you.utils.prompting import TemplateLoader, fill_template

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


def format_style_guide(style) -> str:
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


def format_style_samples(style, max_samples: int = 3) -> str:
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


def format_style_constraints(style) -> str:
    """根据风格类型生成不同强度的写作约束。

    Args:
        style: WritingStyle object or None.

    Returns:
        风格感知的写作约束文本。literary/classical 风格使用严格约束，
        colloquial/mixed 风格使用宽松约束。
    """
    if style is None:
        return ""

    register = getattr(style, "register", "mixed").lower()
    desc_focus = {item.lower() for item in getattr(style, "description_focus", []) if item}
    metaphor_style = getattr(style, "metaphor_style", "").strip()
    tone_markers = getattr(style, "tone_markers", [])

    focus_lines: list[str] = []
    if "landscape" in desc_focus:
        focus_lines.append("- 每段必须嵌入具体景物/方位细节，用景象映射人物情绪")
    if "psychological" in desc_focus:
        focus_lines.append('- 角色心理需通过细小动作、触觉或静物暗示呈现，避免直接写"他很伤心"')
    if "action" in desc_focus:
        focus_lines.append("- 动作描写要有起因与结果，避免无意义的打斗或奔跑")
    if metaphor_style:
        focus_lines.append(f"- 比喻应遵循「{metaphor_style}」的习惯表达，勿加入现代化比喻")
    if tone_markers:
        whitelisted = "、".join(tone_markers[:4])
        focus_lines.append(f"- 语气词限于：{whitelisted}，并且只在必要时使用一次")

    if register in ("literary", "classical"):
        # 文学/古典风格：严格约束
        base = """## 写作约束（文学风格）
- 情绪必须**间接表达**：通过动作、景物、对话暗示，禁止直接陈述（如"心中满是xxx"、"眼中满是"）
- 禁止网文套路：避免"脸上洋溢着"、"仿佛一切都有了新的开始"、"留下一个xxx的背影"等俗套
- 禁止解释性叙述：如"这让她感到温暖"，改为通过场景展示
- 对话要符合时代背景，避免现代口语"""
    else:
        # 通俗/网文风格：宽松约束
        base = """## 写作约束（通俗风格）
- 情绪表达可以直接，但避免过度重复相同句式
- 保持节奏流畅，情节推进明快
- 避免同一段落内重复使用相同的表达方式"""

    if focus_lines:
        base = base + "\n" + "\n".join(focus_lines)
    return base
