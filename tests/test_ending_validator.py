from __future__ import annotations

import json
import pytest

from story_for_you.core.ending.hint_interpreter import HintDirectives
from story_for_you.core.ending.validator import EndingValidator
from story_for_you.core.exceptions import LLMResponseError
from story_for_you.llm.base import LLMProvider, LLMResponse


class _ReviewLLM(LLMProvider):
    def generate(self, prompt: str, system: str = "", options: dict | None = None) -> LLMResponse:
        payload = {
            "passed": False,
            "issues": ["人物结局前后矛盾"],
            "repair_instructions": ["统一傩送的最终状态"],
        }
        return LLMResponse(content=json.dumps(payload, ensure_ascii=False), tokens_used=0)

    def generate_stream(self, prompt: str, system: str = "", options: dict | None = None):
        yield from []


class _IncompleteReviewLLM(LLMProvider):
    def generate(self, prompt: str, system: str = "", options: dict | None = None) -> LLMResponse:
        return LLMResponse(content=json.dumps({"passed": True}, ensure_ascii=False), tokens_used=0)

    def generate_stream(self, prompt: str, system: str = "", options: dict | None = None):
        yield from []


def test_ending_validator_delegates_semantic_review_to_llm():
    directives = HintDirectives(closure="closed")

    result = EndingValidator(_ReviewLLM()).validate(
        "傩送死了。傩送又写信回来。",
        directives,
        context_block="人物：傩送",
    )

    assert not result.passed
    assert result.issues == ["人物结局前后矛盾"]
    assert result.repair_instructions == ["统一傩送的最终状态"]


def test_ending_validator_requires_complete_schema():
    with pytest.raises(LLMResponseError, match="issues"):
        EndingValidator(_IncompleteReviewLLM()).validate(
            "正文",
            HintDirectives(),
            context_block="上下文",
        )
