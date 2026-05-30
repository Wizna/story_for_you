"""Reusable prompt text for ending writer processing."""

from __future__ import annotations

__all__ = [
    "BANNED_EXPRESSIONS_PROMPT",
]

BANNED_EXPRESSIONS_PROMPT = """\
请主动判断哪些表达与原作风格、时代语感、人物处境或文学质地不符，并在生成时避免。
情绪、转折和主题收束应通过可感知的动作、景物、对话和叙事节奏呈现，不要用任务说明或审稿评语代替正文。
非开放式结局也必须避免说明性复盘：不要让人物或旁白逐条解释所有误会、动机和归宿。
不要引入无铺垫的后代、陌生少年、多年后偶遇或临终独白来收束主线。
如果必须澄清误会，应通过当下场景中的短对白、行动和关系选择完成。
"""
