"""Hint interpretation for ending writer.

Parses user hints into structured directives for story continuation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from story_for_you.analysis.context import StoryContext

__all__ = [
    "HintDirectives",
    "HintInterpreter",
]


@dataclass
class HintDirectives:
    """归一化后的读者意图，便于在不同阶段复用。"""

    normalized_text: str = "无特别要求"
    ending_direction: str | None = None
    emotional_tone: str | None = None
    focus_characters: list[str] = field(default_factory=list)

    def for_prompt(self) -> str:
        """将结构化指令转成短句，用于 prompt 占位。"""

        parts: list[str] = []
        base = self.normalized_text.strip()
        if base and base != "无特别要求":
            parts.append(base)
        if self.ending_direction:
            parts.append(f"结局方向：{self.ending_direction}")
        if self.emotional_tone:
            parts.append(f"情感氛围：{self.emotional_tone}")
        if self.focus_characters:
            parts.append("重点人物：" + ", ".join(self.focus_characters))
        return "；".join(parts) if parts else "无特别要求"


class HintInterpreter:
    """Interprets user hints into structured directives."""

    # Ending direction keywords
    DIRECTION_ALIASES = {
        "HE": {"HE", "HAPPY", "HAPPY END", "HAPPY ENDING", "圆满", "大团圆", "皆大欢喜", "治愈", "希望", "甜"},
        "BE": {"BE", "BAD", "BE结局", "悲", "刀", "虐", "悲剧", "凄凉"},
        "OE": {"OE", "OPEN", "留白", "开放", "未知", "OE结局"},
    }

    # Emotional tone keywords
    TONE_ALIASES = {
        "温暖": ["温暖", "治愈", "柔和", "明亮", "希望"],
        "苍凉": ["苍凉", "萧瑟", "凄冷", "孤独"],
        "热烈": ["热烈", "激昂", "燃", "热血"],
        "哀伤": ["哀伤", "忧伤", "悲怆", "痛"],
        "平静": ["平静", "克制", "淡然", "沉稳"],
    }

    def interpret(self, raw_hint: str, context: StoryContext) -> HintDirectives:
        """Parse简写、情绪词汇以及人物指向，生成统一指令。"""

        hint = (raw_hint or "").strip()
        directives = HintDirectives(normalized_text=hint or "无特别要求")
        if not hint:
            directives.focus_characters = []
            return directives

        lowered = hint.lower()
        upper_tokens = {token.upper() for token in re.findall(r"[A-Za-z]+", hint)}

        # Detect ending direction
        for label, keywords in self.DIRECTION_ALIASES.items():
            if label in upper_tokens:
                directives.ending_direction = label
                break
            if any(keyword.lower() in lowered for keyword in keywords if keyword.isascii()):
                directives.ending_direction = label
                break
            if any(keyword in hint for keyword in keywords if not keyword.isascii()):
                directives.ending_direction = label
                break

        # Detect emotional tone
        for tone, keywords in self.TONE_ALIASES.items():
            if any(keyword in hint for keyword in keywords):
                directives.emotional_tone = tone
                break

        # Detect focus characters
        focus: list[str] = []
        for name in context.characters.keys():
            if name and name in hint:
                focus.append(name)
        directives.focus_characters = focus[:4]

        return directives
