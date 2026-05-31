from __future__ import annotations

import json
import pytest

from story_for_you.analysis.context import CharacterState, StoryContext
from story_for_you.core.ending import StyleEnforcer
from story_for_you.core.exceptions import GenerationError
from story_for_you.core.ending_writer import EndingWriter
from story_for_you.indexer.segment import Segment, SegmentIndex
from story_for_you.llm.base import LLMProvider, LLMResponse
from story_for_you.utils.prompting import CacheablePrompt


class _FakeLLM(LLMProvider):
    def generate(self, prompt: CacheablePrompt, system: str = "", options: dict | None = None) -> LLMResponse:
        return LLMResponse(content="", tokens_used=0)

    def generate_stream(self, prompt: CacheablePrompt, system: str = "", options: dict | None = None):
        yield from []


class _BlockedResolutionLLM(LLMProvider):
    def generate(self, prompt: CacheablePrompt, system: str = "", options: dict | None = None) -> LLMResponse:
        payload = {
            "status": "blocked",
            "missing_threads": ["傩送归宿未交代"],
            "bridges": [],
            "notes": "无法自然补写",
        }
        return LLMResponse(content=json.dumps(payload, ensure_ascii=False), tokens_used=0)

    def generate_stream(self, prompt: CacheablePrompt, system: str = "", options: dict | None = None):
        yield from []


class _RepairLLM(LLMProvider):
    def __init__(self):
        self.prompts: list[CacheablePrompt] = []

    def generate(self, prompt: CacheablePrompt, system: str = "", options: dict | None = None) -> LLMResponse:
        self.prompts.append(prompt)
        return LLMResponse(content="修复后的正文", tokens_used=0)

    def generate_stream(self, prompt: CacheablePrompt, system: str = "", options: dict | None = None):
        yield from []


class _ValidationResult:
    def __init__(self, passed: bool):
        self.passed = passed
        self.issues = [] if passed else ["傩送归宿前后矛盾"]
        self.repair_instructions = [] if passed else ["删除少年过渡情节"]


class _FailThenPassValidator:
    def __init__(self):
        self.calls = 0

    def validate(self, text: str, directives, *, context_block: str):
        self.calls += 1
        return _ValidationResult(self.calls > 1)


def test_recent_segment_digest_uses_tail_excerpt():
    context = StoryContext(
        characters={"翠翠": CharacterState(name="翠翠", role="main")},
    )
    segment = Segment(
        segment_id=1,
        content="开头版权信息。" + ("中段。" * 100) + "结尾处翠翠守着渡口，杨马兵陪在旁边。",
        chapter=1,
        characters=["翠翠"],
    )
    index = SegmentIndex(
        segments=[segment],
        char_index={"翠翠": [1]},
        chapter_index={1: [1]},
        gap_map={},
    )
    writer = EndingWriter(_FakeLLM(), index)

    digest = writer._recent_segment_digest(context)

    assert "结尾处翠翠守着渡口" in digest
    assert "开头版权信息" not in digest


def test_resolution_blocked_status_raises():
    context = StoryContext(
        characters={"翠翠": CharacterState(name="翠翠", role="main", unresolved=["等待傩送"])},
    )
    writer = EndingWriter(
        _BlockedResolutionLLM(),
        SegmentIndex(segments=[], char_index={}, chapter_index={}, gap_map={}),
    )

    with pytest.raises(GenerationError, match="傩送归宿未交代"):
        writer._phase_resolution_review("正文", context, "上下文", None, "{}")


def test_final_validation_failure_repairs_once():
    llm = _RepairLLM()
    validator = _FailThenPassValidator()
    writer = EndingWriter(
        llm,
        SegmentIndex(segments=[], char_index={}, chapter_index={}, gap_map={}),
    )
    writer._ending_validator = validator

    repaired = writer._validate_or_repair_final(
        "有矛盾的正文",
        directives={},
        context_block="上下文",
        style=None,
        hint="续写一个非开放式结局。",
        enforcer=StyleEnforcer(None),
    )

    assert repaired == "修复后的正文"
    assert validator.calls == 2
    assert llm.prompts[0].prefix == "上下文"
    assert "删除少年过渡情节" in llm.prompts[0].render()
    assert "见前一条 user 消息中的完整文本" in llm.prompts[0].task
