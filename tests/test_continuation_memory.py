import json

import pytest

from story_for_you.analysis.context import (
    CharacterState,
    EventImpact,
    PlotEvent,
    StoryContext,
)
from story_for_you.core.ending.memory import ContinuationMemory, UnitUpdate
from story_for_you.core.ending.validator import ContinuationValidationResult
from story_for_you.core.ending_writer import EndingWriter
from story_for_you.exceptions import LLMResponseError
from story_for_you.indexer.segment import Segment, SegmentIndex
from tests.test_review_fixes import _ResponseLLM


def test_generated_memory_retains_old_changes_and_never_mutates_source():
    source = StoryContext(characters={"甲": CharacterState(name="甲", location="城门")})
    memory = ContinuationMemory(source)
    memory.commit(
        UnitUpdate(
            ContinuationValidationResult(True),
            "甲交出宝剑，进入客栈。",
            [CharacterState(name="甲", location="客栈", knowledge=["乙持有宝剑"])],
            [
                PlotEvent(
                    "GEN-U1-E1",
                    0,
                    "progress",
                    ["甲", "乙"],
                    "宝剑归乙持有",
                    EventImpact(),
                )
            ],
        ),
        1,
    )
    for number in range(2, 15):
        memory.commit(
            UnitUpdate(ContinuationValidationResult(True), f"单元{number}的新行动"),
            number,
        )
    snapshot = memory.snapshot()
    assert source.characters["甲"].location == "城门"
    assert snapshot.characters["甲"].location == "客栈"
    assert snapshot.characters["甲"].knowledge == ["乙持有宝剑"]
    assert "宝剑归乙持有" in memory.render()
    assert "甲交出宝剑" in memory.render()
    with pytest.raises(LLMResponseError, match="unvalidated"):
        memory.commit(UnitUpdate(ContinuationValidationResult(False)), 15)
    assert len(memory.summaries) == 14


def test_unit_memory_rejects_invented_evidence():
    response = {
        "passed": True,
        "issues": [],
        "repair_instructions": [],
        "summary": "新场景",
        "characters": [{"evidence": "原文没有这句话"}],
        "events": [],
        "story_state": None,
    }
    with pytest.raises(LLMResponseError, match="evidence"):
        ContinuationMemory(StoryContext()).review(
            _ResponseLLM(json.dumps(response)),
            "甲仍在城门。",
            "已知状态",
            1,
        )


def test_focus_retrieval_keeps_seventh_character_and_early_irreversible_fact():
    characters = {
        name: CharacterState(name=name, role="main", location=f"LOC_{name}")
        for name in "abcdefg"
    }
    old = PlotEvent("old", 1, "reveal", ["g"], "早期身份已公开", EventImpact(), True)
    newer = [
        PlotEvent(
            f"e{i}", i + 2, "progress", ["a"], f"后来事件{i}", EventImpact(), True
        )
        for i in range(25)
    ]
    segment = Segment(
        7, "无关开头" * 200 + "g从故乡带来信物。" + "无关结尾" * 200, characters=["g"]
    )
    writer = EndingWriter(_ResponseLLM(""), SegmentIndex([segment], {"g": [7]}, {}, {}))
    block = writer._build_continuation_context(
        "最近原文",
        StoryContext(characters=characters, events=[old, *newer]),
        focus_characters=["g"],
    )
    assert "LOC_g" in block
    assert "早期身份已公开" in block
    assert "g从故乡带来信物" in block
