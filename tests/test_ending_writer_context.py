from __future__ import annotations

import json
import pytest

from story_for_you.analysis.context import CharacterState, EventImpact, PlotEvent, StoryContext
from story_for_you.core.ending import StyleEnforcer
from story_for_you.core.exceptions import GenerationError, LLMResponseError
from story_for_you.core.ending_writer import (
    AnchorAction,
    ContinuationPlan,
    EndingChapterPlan,
    EndingWriter,
    NarrativeSignal,
    SourceAnchor,
)
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
        return LLMResponse(content=json.dumps({"edits": [{
            "old_text": "有矛盾的正文", "new_text": "修复后的正文", "reason": "统一人物归宿",
        }]}), tokens_used=0)

    def generate_stream(self, prompt: CacheablePrompt, system: str = "", options: dict | None = None):
        yield from []


class _PlannedEndingLLM(LLMProvider):
    def __init__(self):
        self.calls = 0
        self.stage_calls = 0

    def generate(self, prompt: CacheablePrompt, system: str = "", options: dict | None = None) -> LLMResponse:
        self.calls += 1
        if "本单元场景因果规划" in prompt.task:
            return LLMResponse(json.dumps({"scenes": [{
                "viewpoint": "甲", "setting": "当夜", "goal": "探明异常", "obstacle": "线索不全",
                "action": "查验物件", "turn": "发现旧痕", "cost": "暴露行踪", "end_state": "决定追查",
                "knowledge_boundary": "只知道物件有异，不知道幕后人物",
            }]}), 0)
        if "单元承接审查与记忆更新" in prompt.task:
            return LLMResponse(json.dumps({
                "passed": True, "issues": [], "repair_instructions": [],
                "summary": "本单元推进了行动，形成当前场景结果。", "characters": [], "events": [], "story_state": None,
            }), 0)
        self.stage_calls += 1
        if self.stage_calls == 1:
            payload = {
                "normalized_text": "无特别要求", "closure": "closed", "ending_direction": None,
                "emotional_tone": None, "focus_characters": [], "required_outcomes": [],
                "forbidden_outcomes": [], "required_resolutions": [], "style_constraints": [],
            }
            return LLMResponse(json.dumps(payload, ensure_ascii=False), 0)
        if self.stage_calls == 2:
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
        if self.stage_calls in (3, 5):
            return LLMResponse("初稿推进了一个完整场景。", 0)
        if self.stage_calls in (4, 6):
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


def test_continuation_context_recalls_subtext_cue_paragraphs_without_interpreting_them():
    writer = EndingWriter(_FakeLLM(), SegmentIndex([], {}, {}, {}))
    block = writer._build_continuation_context(
        "老太太惊叫道：‘娘要被毒死了！’\n图上画着迦楼罗钳住飞龙。\n普通景物。",
        StoryContext(),
    )

    assert "Candidate Subtext Cues" in block
    assert "娘要被毒死了" in block
    assert "迦楼罗钳住飞龙" in block
    assert "不代表任何解释成立" in block


def test_continuation_context_lists_action_linked_anchor_candidates():
    writer = EndingWriter(_FakeLLM(), SegmentIndex([], {}, {}, {}))
    block = writer._build_continuation_context(
        "老者从怀里取出一道奏章，扔到了御榻上。\n普通景物。",
        StoryContext(),
    )

    assert "Candidate Anchor Inventory" in block
    assert "label=奏章" in block


def test_subtext_cues_keep_late_high_signal_ending_from_punctuation_flood():
    writer = EndingWriter(_FakeLLM(), SegmentIndex([], {}, {}, {}))
    noisy_middle = "\n".join(f"众人问道：‘第{i}句有何道理？’" for i in range(180))
    source = noisy_middle + "\n艳婷取出药瓶，老太太惊叫道：‘娘要被毒死了！’"

    cues = writer._extract_subtext_cues(source)

    assert "艳婷取出药瓶" in cues
    assert "娘要被毒死了" in cues


def test_single_subtext_cue_still_triggers_preanalysis():
    class SubtextLLM(_FakeLLM):
        def generate(self, prompt, system="", options=None):
            return LLMResponse(json.dumps({"signals": [{
                "signal_id": "s1", "kind": "foreshadowing", "label": "毒",
                "evidence": "她说：‘娘要被毒死了！’", "reading": "可能是死亡预告",
                "confidence": "medium", "candidate_consequences": ["查验药物"],
            }]}), 0)

    writer = EndingWriter(SubtextLLM(), SegmentIndex([], {}, {}, {}))
    signals = writer._phase_subtext(
        "Recent Source Scene\n她说：‘娘要被毒死了！’\nCandidate Subtext Cues",
        "承接", "她说：‘娘要被毒死了！’",
    )

    assert len(signals) == 1


def test_continuation_context_recalls_early_motif_for_ending_signal():
    writer = EndingWriter(
        _FakeLLM(),
        SegmentIndex(
            [
                Segment(1, "多年前宫中壁上已有迦楼罗钳龙的旧画。", chapter=8),
                Segment(2, "末章又画着迦楼罗钳住飞龙。", chapter=20),
            ],
            {}, {}, {},
        ),
    )
    signal = NarrativeSignal(
        "s-dragon", "metaphor", "钳龙图", "末章又画着迦楼罗钳住飞龙。",
        "可能对应权力关系", "medium", candidate_consequences=["身份或权力变化"],
    )

    block = writer._build_continuation_context(
        "末章又画着迦楼罗钳住飞龙。", StoryContext(),
        narrative_signals=[signal],
    )

    assert "多年前宫中壁上已有迦楼罗钳龙的旧画" in block
    assert "Long-range Foreshadowing Evidence" in block


def test_candidate_cue_history_recalls_early_match_before_subtext_interpretation():
    writer = EndingWriter(
        _FakeLLM(),
        SegmentIndex(
            [
                Segment(1, "早年曾有人提到玉玺藏在内库。", chapter=3),
                Segment(2, "末章又见玉玺入殿。", chapter=20),
                Segment(3, "近景。", chapter=21),
                Segment(4, "近景。", chapter=22),
                Segment(5, "近景。", chapter=23),
                Segment(6, "近景。", chapter=24),
            ],
            {}, {}, {},
        ),
    )

    history = writer._build_candidate_cue_history(
        StoryContext(), "- cue-1: 末章又见玉玺入殿。"
    )

    assert "早年曾有人提到玉玺藏在内库" in history
    assert "Candidate Cue Historical Matches" in history


def test_continuation_context_excludes_download_site_trailer():
    writer = EndingWriter(_FakeLLM(), SegmentIndex([], {}, {}, {}))
    text = (
        "第一章 正文\n故事完结。\n"
        "==========\n更多精校小说尽在：https://example.test\n==========\n"
    )

    block = writer._build_continuation_context(text, StoryContext())

    assert "故事完结" in block
    assert "example.test" not in block


def test_dead_support_character_is_not_mandatory_cast():
    event = PlotEvent(
        event_id="E1",
        chapter=1,
        type="setback",
        participants=["甲"],
        summary="甲已死亡",
        impact=EventImpact(),
        is_irreversible=True,
        deceased_characters=["甲"],
    )
    context = StoryContext(
        characters={"甲": CharacterState(name="甲", role="support")},
        events=[event],
    )
    writer = EndingWriter(_FakeLLM(), SegmentIndex([], {}, {}, {}))

    assert "甲" not in writer._format_main_characters(context)


def test_death_event_does_not_kill_witness_or_survivor():
    event = PlotEvent(
        "E1", 1, "setback", ["甲", "乙"], "乙战死，甲为乙收尸。",
        EventImpact(), True, deceased_characters=["乙"],
    )
    context = StoryContext(
        characters={"甲": CharacterState(name="甲", role="main", status="alive")},
        events=[event],
    )
    writer = EndingWriter(_FakeLLM(), SegmentIndex([], {}, {}, {}))
    assert not writer._is_dead_character("甲", context)
    assert writer._is_dead_character("乙", context)
    assert "甲" in writer._format_main_characters(context)


def test_continue_story_uses_model_selected_multi_chapter_plan():
    llm = _PlannedEndingLLM()
    writer = EndingWriter(llm, SegmentIndex([], {}, {}, {}))

    result = writer.continue_story("原文末尾：众人已经汇聚。", StoryContext(), max_chapters=4)

    assert "第1章 风暴将至" in result
    assert "第2章 天明之前" in result
    assert llm.calls == 11


def test_plan_accepts_generic_continuation_schema():
    writer = EndingWriter(_GenericPlanLLM(), SegmentIndex([], {}, {}, {}))

    plan = writer._phase_plan("上下文", "后四十回", max_chapters=6)

    assert plan.unit_count == 1
    assert plan.continuation_intent == "承接家族变化并保留余波"
    assert plan.closure_level == "partial"
    assert plan.unit_type == "回"
    assert plan.chapters[0].end_state == "人物关系出现不可逆变化"


def test_plan_parses_source_anchor_and_assigns_it_to_a_unit():
    class AnchorPlan(_GenericPlanLLM):
        def generate(self, prompt, system="", options=None):
            payload = json.loads(super().generate(prompt, system, options).content)
            payload["source_anchors"] = [{
                "anchor_id": "a-memorial",
                "kind": "message",
                "label": "奏章",
                "evidence": "从怀里取出了一道奏章，扔到了御榻上",
                "significance": "它应当带来新的政务信息或冲突",
                "handling": "advance",
            }]
            payload["chapters"][0]["anchor_actions"] = [{
                "anchor_id": "a-memorial",
                "handling": "advance",
                "action": "让皇帝拆阅奏章并作出有代价的决定",
            }]
            return LLMResponse(json.dumps(payload, ensure_ascii=False), 0)

    writer = EndingWriter(AnchorPlan(), SegmentIndex([], {}, {}, {}))
    plan = writer._phase_plan(
        "上下文", "后四十回", max_chapters=6,
        source_text="老者从怀里取出了一道奏章，扔到了御榻上。",
    )

    assert plan.source_anchors[0].label == "奏章"
    assert plan.chapters[0].anchor_actions[0].anchor_id == "a-memorial"


def test_source_anchor_evidence_must_quote_source_text():
    class InvalidAnchorPlan(_GenericPlanLLM):
        def generate(self, prompt, system="", options=None):
            payload = json.loads(super().generate(prompt, system, options).content)
            payload["source_anchors"] = [{
                "anchor_id": "a-memorial",
                "kind": "message",
                "label": "奏章",
                "evidence": "模型编造的奏章内容",
                "significance": "会改变后续",
                "handling": "hold",
            }]
            payload["chapters"][0]["anchor_actions"] = [{
                "anchor_id": "a-memorial",
                "handling": "hold",
                "action": "保留为下一单元线索",
            }]
            return LLMResponse(json.dumps(payload, ensure_ascii=False), 0)

    writer = EndingWriter(InvalidAnchorPlan(), SegmentIndex([], {}, {}, {}))
    with pytest.raises(LLMResponseError, match="must quote the supplied source scene"):
        writer._phase_plan(
            "上下文", "后四十回", max_chapters=6,
            source_text="老者从怀里取出了一道奏章，扔到了御榻上。",
        )


def test_plan_anchor_evidence_must_come_from_ending_window_not_earlier_source():
    class EarlierAnchorPlan(_GenericPlanLLM):
        def generate(self, prompt, system="", options=None):
            payload = json.loads(super().generate(prompt, system, options).content)
            payload["source_anchors"] = [{
                "anchor_id": "old-object",
                "kind": "object",
                "label": "旧匣",
                "evidence": "早年曾有人打开旧匣，里面藏着一封信。",
                "significance": "可能关联旧案",
                "handling": "hold",
            }]
            payload["chapters"][0]["anchor_actions"] = [{
                "anchor_id": "old-object", "handling": "hold", "action": "保留悬念",
            }]
            return LLMResponse(json.dumps(payload, ensure_ascii=False), 0)

    writer = EndingWriter(EarlierAnchorPlan(), SegmentIndex([], {}, {}, {}))
    with pytest.raises(LLMResponseError, match="must quote the supplied source scene"):
        writer._phase_plan(
            "上下文", "承接", max_chapters=1,
            source_text=(
                "早年曾有人打开旧匣，里面藏着一封信。\n"
                + ("普通景物。" * 7000)
                + "\n末章只剩皇帝抬眼。"
            ),
        )


def test_plan_keeps_subtext_as_evidence_backed_hypothesis_and_recommends_plot():
    class SignalPlan(_GenericPlanLLM):
        def generate(self, prompt, system="", options=None):
            payload = json.loads(super().generate(prompt, system, options).content)
            payload["narrative_signals"] = [{
                "signal_id": "signal-poison",
                "kind": "double_meaning",
                "label": "娘要被毒死了",
                "evidence": "娘要被毒死了！",
                "reading": "可能是真实投毒，也可能是人物对艳婷的不信任预告",
                "confidence": "medium",
                "candidate_consequences": ["药物被查验", "艳婷的动机被迫暴露"],
            }]
            payload["plot_recommendations"] = [{
                "recommendation_id": "plot-poison",
                "title": "药散引出的试探",
                "premise": "围绕不信任先查药，再让关系和立储局势发生变化",
                "based_on": ["signal-poison"],
                "beats": ["查验药散", "发现与奏章相关的线索"],
                "tradeoff": "暂不确认艳婷是否真要下毒",
                "recommended": True,
            }]
            return LLMResponse(json.dumps(payload, ensure_ascii=False), 0)

    writer = EndingWriter(SignalPlan(), SegmentIndex([], {}, {}, {}))
    plan = writer._phase_plan(
        "上下文", "承接", max_chapters=1,
        source_text="老太太惊叫道：‘娘要被毒死了！’",
    )

    assert plan.narrative_signals[0].confidence == "medium"
    assert plan.plot_recommendations[0].based_on == ["signal-poison"]
    assert "不信任" in writer._format_plan_readings(plan)


def test_plan_requires_high_confidence_signal_to_be_covered_by_recommendation():
    class UncoveredSignalPlan(_GenericPlanLLM):
        def generate(self, prompt, system="", options=None):
            payload = json.loads(super().generate(prompt, system, options).content)
            payload["narrative_signals"] = [{
                "signal_id": "signal-death", "kind": "foreshadowing", "label": "死亡预告",
                "evidence": "娘要被毒死了！", "reading": "可能预告某人死亡",
                "confidence": "high", "candidate_consequences": ["死亡"],
            }]
            payload["plot_recommendations"] = [{
                "recommendation_id": "plot-other", "title": "另一路",
                "premise": "推进别的冲突", "based_on": [], "beats": ["行动"],
                "tradeoff": "忽略死亡预告", "recommended": True,
            }]
            return LLMResponse(json.dumps(payload, ensure_ascii=False), 0)

    writer = EndingWriter(UncoveredSignalPlan(), SegmentIndex([], {}, {}, {}))
    with pytest.raises(LLMResponseError, match="each high/medium signal"):
        writer._phase_plan(
            "上下文", "承接", max_chapters=1,
            source_text="老太太说：‘娘要被毒死了！’",
        )


def test_signal_evidence_is_rejected_when_it_is_not_contiguous_source_text():
    class InvalidSignalPlan(_GenericPlanLLM):
        def generate(self, prompt, system="", options=None):
            payload = json.loads(super().generate(prompt, system, options).content)
            payload["narrative_signals"] = [{
                "signal_id": "signal-poison",
                "kind": "foreshadowing",
                "label": "毒",
                "evidence": "娘会真的被毒死",
                "reading": "确定死亡",
                "confidence": "high",
                "candidate_consequences": ["死亡"],
            }]
            payload["plot_recommendations"] = [{
                "recommendation_id": "plot-poison",
                "title": "毒发",
                "premise": "毒发",
                "based_on": ["signal-poison"],
                "beats": ["死亡"],
                "tradeoff": "失去悬念",
                "recommended": True,
            }]
            return LLMResponse(json.dumps(payload, ensure_ascii=False), 0)

    writer = EndingWriter(InvalidSignalPlan(), SegmentIndex([], {}, {}, {}))
    with pytest.raises(LLMResponseError, match=r"narrative_signals\[0\].evidence"):
        writer._phase_plan(
            "上下文", "承接", max_chapters=1,
            source_text="老太太惊叫道：‘娘要被毒死了！’",
        )


def test_signal_evidence_snaps_terminal_punctuation_to_source():
    writer = EndingWriter(_FakeLLM(), SegmentIndex([], {}, {}, {}))
    source = "皇上问道：‘你们说这句话……有没有道理啊……’"
    signal = {
        "signal_id": "signal-question",
        "kind": "double_meaning",
        "label": "皇上试探群臣",
        "evidence": "皇上问道：‘你们说这句话……有没有道理啊？’",
        "reading": "可能是对群臣的试探",
        "confidence": "medium",
        "candidate_consequences": ["群臣被迫表态"],
    }

    parsed = writer._parse_narrative_signals([signal], source)

    assert parsed[0].evidence == source


def test_anchor_actions_are_visible_to_chapter_and_draft_prompts():
    class CaptureLLM(_FakeLLM):
        def __init__(self):
            self.prompts = []

        def generate(self, prompt, system="", options=None):
            self.prompts.append(prompt)
            return LLMResponse("正文", 0)

    anchor = SourceAnchor(
        "a-memorial", "message", "奏章", "扔到了御榻上", "带来政务信息", "hold"
    )
    action = AnchorAction("a-memorial", "hold", "本单元留下皇帝尚未拆阅的悬置")
    chapter = EndingChapterPlan(1, "灯影", "推进", "甲", "当夜", anchor_actions=[action])
    plan = ContinuationPlan(1, "原因", "推进", "当夜", "open", "章", [chapter], [anchor])
    llm = CaptureLLM()
    writer = EndingWriter(llm, SegmentIndex([], {}, {}, {}))
    instruction = writer._chapter_instruction(plan, chapter, "open")
    writer._phase_draft(
        StoryContext(), writer._outline_from_chapter(plan, chapter), None,
        "上下文", "", "", instruction, source_anchors=writer._format_anchor_actions([anchor], [action]),
    )

    assert "奏章" in instruction
    assert "本单元留下皇帝尚未拆阅的悬置" in llm.prompts[-1].render()


def test_hold_anchor_is_explicitly_checked_by_resolution_review():
    class CaptureResolution(_FakeLLM):
        def __init__(self):
            self.content = json.dumps({
                "status": "ok", "missing_threads": [], "edits": [], "notes": "",
            }, ensure_ascii=False)
            self.prompt = None

        def generate(self, prompt, system="", options=None):
            self.prompt = prompt
            return LLMResponse(self.content, 0)

    anchor = SourceAnchor(
        "a-memorial", "message", "奏章", "扔到了御榻上", "尚未拆阅", "hold"
    )
    llm = CaptureResolution()
    writer = EndingWriter(llm, SegmentIndex([], {}, {}, {}))
    result = writer._phase_resolution_review(
        "正文", StoryContext(characters={"甲": CharacterState(name="甲", unresolved=["旧事"])}),
        "上下文", None, "", "open", source_anchors=[anchor],
    )

    assert result == "正文"
    assert llm.prompt is not None
    assert "奏章" in llm.prompt.render()
    assert "不能因相关台词已经出现就视为已处理" in llm.prompt.render()


def test_resolution_review_runs_for_subtext_even_without_open_threads():
    class SignalResolution(_FakeLLM):
        def __init__(self):
            self.calls = 0

        def generate(self, prompt, system="", options=None):
            self.calls += 1
            return LLMResponse(json.dumps({
                "status": "ok", "missing_threads": [], "edits": [], "notes": "",
            }, ensure_ascii=False), 0)

    signal = NarrativeSignal(
        "s1", "foreshadowing", "死亡预告", "娘要被毒死了！", "可能是投毒预告",
        "medium", candidate_consequences=["查验药物"],
    )
    llm = SignalResolution()
    writer = EndingWriter(llm, SegmentIndex([], {}, {}, {}))
    result = writer._phase_resolution_review(
        "正文", StoryContext(), "上下文", None, "", "open", narrative_signals=[signal],
    )

    assert result == "正文"
    assert llm.calls == 1


def test_validation_cast_includes_characters_that_return_in_generated_text():
    context = StoryContext(
        characters={
            "甲": CharacterState(name="甲", role="main"),
            "乙": CharacterState(name="乙", role="support", goal="追回玉玺"),
        }
    )
    writer = EndingWriter(_FakeLLM(), SegmentIndex([], {}, {}, {}))
    plan = ContinuationPlan(
        1, "推进", "推进关系", "当夜", "open", "场景",
        [EndingChapterPlan(1, "夜", "推进", "甲", "城门")],
    )

    block = writer._build_validation_context(
        "原文", context, plan, [], [], final_text="甲在城门遇见乙，乙仍要追回玉玺。",
    )

    assert "乙" in block
    assert "追回玉玺" in block


def test_plan_prompt_receives_actual_limit_independently_of_full_requested_scope():
    class CapturePlan(_GenericPlanLLM):
        def __init__(self):
            self.prompts = []

        def generate(self, prompt, system="", options=None):
            self.prompts.append(prompt.render())
            return super().generate(prompt, system, options)

    llm = CapturePlan()
    writer = EndingWriter(llm, SegmentIndex([], {}, {}, {}))
    writer._phase_plan("原作证据", "补写后四十回", 1)
    writer._phase_plan("原作证据", "补写后四十回", 40)
    assert "max_chapters = 1。" in llm.prompts[0]
    assert "max_chapters = 40。" in llm.prompts[1]
    assert all("补写后四十回" in prompt for prompt in llm.prompts)


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
