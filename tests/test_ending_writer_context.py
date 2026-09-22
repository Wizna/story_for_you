from __future__ import annotations

import json
import pytest

from story_for_you.analysis.context import CharacterState, EventImpact, PlotEvent, StoryContext
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


class _PlannedEndingLLM(LLMProvider):
    def __init__(self):
        self.calls = 0

    def generate(self, prompt: CacheablePrompt, system: str = "", options: dict | None = None) -> LLMResponse:
        self.calls += 1
        if self.calls == 1:
            payload = {
                "normalized_text": "无特别要求", "closure": "closed", "ending_direction": None,
                "emotional_tone": None, "focus_characters": [], "required_outcomes": [],
                "forbidden_outcomes": [], "required_resolutions": [], "style_constraints": [],
            }
            return LLMResponse(json.dumps(payload, ensure_ascii=False), 0)
        if self.calls == 2:
            payload = {
                "chapter_count": 2, "rationale": "当前仍有两次因果推进",
                "core_theme": "选择与代价", "ending_direction": "OE",
                "chapters": [
                    {"title": "风暴将至", "purpose": "各方汇聚", "viewpoint": "甲", "setting": "当夜",
                     "beats": ["发现异常", "做出选择"], "focus_characters": ["甲"],
                     "resolutions": ["推进冲突"], "carry_forward": ["真相未明"]},
                    {"title": "天明之前", "purpose": "完成收束", "viewpoint": "乙", "setting": "翌日",
                     "beats": ["付出代价", "留下余波"], "focus_characters": ["乙"],
                     "resolutions": ["交代归宿"], "carry_forward": []},
                ],
            }
            return LLMResponse(json.dumps(payload, ensure_ascii=False), 0)
        if self.calls in (3, 5):
            return LLMResponse("初稿推进了一个完整场景。", 0)
        if self.calls in (4, 6):
            return LLMResponse("润色后的场景结果。", 0)
        return LLMResponse(json.dumps({"passed": True, "issues": [], "repair_instructions": []}), 0)

    def generate_stream(self, prompt: CacheablePrompt, system: str = "", options: dict | None = None):
        yield from []


class _GenericPlanLLM(LLMProvider):
    def generate(self, prompt: CacheablePrompt, system: str = "", options: dict | None = None) -> LLMResponse:
        payload = {
            "unit_count": 1,
            "rationale": "当前仍在展开阶段",
            "continuation_intent": "承接家族变化并保留余波",
            "narrative_horizon": "后四十回的前一结构段",
            "closure_level": "partial",
            "unit_type": "回",
            "chapters": [{
                "title": "灯影",
                "purpose": "推进关系变化",
                "viewpoint": "甲",
                "setting": "次日",
                "beats": ["察觉异常", "作出选择"],
                "focus_characters": ["甲"],
                "resolved_threads": [],
                "carry_forward": ["家族风波未止"],
                "end_state": "人物关系出现不可逆变化",
            }],
        }
        return LLMResponse(json.dumps(payload, ensure_ascii=False), 0)

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


def test_continuation_context_keeps_source_tail_and_hard_facts():
    event = PlotEvent(
        event_id="E1",
        chapter=3,
        type="setback",
        participants=["甲"],
        summary="甲战死，城门失守",
        impact=EventImpact(world_flags=["城门失守"]),
        is_irreversible=True,
    )
    context = StoryContext(
        characters={"甲": CharacterState(name="甲", role="support")},
        events=[event],
    )
    writer = EndingWriter(
        _FakeLLM(),
        SegmentIndex(segments=[], char_index={}, chapter_index={}, gap_map={}),
    )

    block = writer._build_continuation_context("原文尾部：皇帝抬眼问道。", context)

    assert "原文尾部：皇帝抬眼问道" in block
    assert "甲战死，城门失守" in block
    assert "Recent Source Scene" in block


def test_dead_support_character_is_not_mandatory_cast():
    event = PlotEvent(
        event_id="E1",
        chapter=1,
        type="setback",
        participants=["甲"],
        summary="甲已死亡",
        impact=EventImpact(),
        is_irreversible=True,
    )
    context = StoryContext(
        characters={"甲": CharacterState(name="甲", role="support")},
        events=[event],
    )
    writer = EndingWriter(_FakeLLM(), SegmentIndex([], {}, {}, {}))

    assert "甲" not in writer._format_main_characters(context)


def test_continue_story_uses_model_selected_multi_chapter_plan():
    llm = _PlannedEndingLLM()
    writer = EndingWriter(llm, SegmentIndex([], {}, {}, {}))

    result = writer.continue_story("原文末尾：众人已经汇聚。", StoryContext(), max_chapters=4)

    assert "第1章 风暴将至" in result
    assert "第2章 天明之前" in result
    assert llm.calls == 7


def test_plan_accepts_generic_continuation_schema():
    writer = EndingWriter(_GenericPlanLLM(), SegmentIndex([], {}, {}, {}))

    plan = writer._phase_plan("上下文", "后四十回", max_chapters=6)

    assert plan.unit_count == 1
    assert plan.continuation_intent == "承接家族变化并保留余波"
    assert plan.closure_level == "partial"
    assert plan.unit_type == "回"
    assert plan.chapters[0].end_state == "人物关系出现不可逆变化"


def test_ongoing_resolution_can_leave_threads_open():
    context = StoryContext(
        characters={"甲": CharacterState(name="甲", role="main", unresolved=["家族风波"])},
    )
    writer = EndingWriter(
        _BlockedResolutionLLM(),
        SegmentIndex(segments=[], char_index={}, chapter_index={}, gap_map={}),
    )

    assert writer._phase_resolution_review("正文", context, "上下文", None, "{}", "ongoing") == "正文"


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
