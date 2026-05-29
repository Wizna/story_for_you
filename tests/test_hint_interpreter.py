from __future__ import annotations

import json

from story_for_you.analysis.context import StoryContext
from story_for_you.core.ending.hint_interpreter import HintInterpreter
from story_for_you.llm.base import LLMProvider, LLMResponse


class _DirectiveLLM(LLMProvider):
    def generate(self, prompt: str, system: str = "", options: dict | None = None) -> LLMResponse:
        payload = {
            "normalized_text": "写成明确收束的非开放式结局",
            "closure": "closed",
            "ending_direction": None,
            "emotional_tone": "克制",
            "focus_characters": ["翠翠"],
            "required_outcomes": ["明确交代翠翠归宿"],
            "forbidden_outcomes": ["留下傩送是否回来不明的悬念"],
            "required_resolutions": ["收束等待与婚约线索"],
            "style_constraints": [],
        }
        return LLMResponse(content=json.dumps(payload, ensure_ascii=False), tokens_used=0)

    def generate_stream(self, prompt: str, system: str = "", options: dict | None = None):
        yield from []


def test_hint_interpreter_uses_llm_directives_without_keyword_matching():
    directives = HintInterpreter(_DirectiveLLM()).interpret("写一个非开放式结局。", StoryContext())

    assert directives.closure == "closed"
    assert directives.ending_direction is None
    assert directives.focus_characters == ["翠翠"]
    assert "非开放式" in directives.normalized_text
