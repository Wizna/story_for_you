"""Reusable prompt text for ending writer processing."""

from __future__ import annotations

__all__ = [
    "BANNED_EXPRESSIONS_PROMPT",
]

BANNED_EXPRESSIONS_PROMPT = """\
请主动判断哪些表达与原作风格、时代语感、人物处境或文学质地不符，并在生成时避免。
情绪、转折和主题收束应通过可感知的动作、景物、对话和叙事节奏呈现，不要用任务说明或审稿评语代替正文。"""
