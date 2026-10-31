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
from tests.test_cacheable_prompting import _RecordingLLM


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
        "characters": [{"name": "甲", "aliases": [], "role": "main", "realm": None,
                        "personality": [], "unresolved": [], "evidence": "原文没有这句话"}],
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


def test_memory_repair_checks_previous_state_without_committing_a_failed_update():
    source = StoryContext(characters={"甲": CharacterState("甲", goal="守城")})
    memory = ContinuationMemory(source)
    text = "甲放弃守城，随众人撤离。"
    response = {"passed": True, "issues": [], "repair_instructions": [], "summary": "甲撤离",
                "events": [], "story_state": None, "characters": [{
                    "name": "甲", "role": "main", "aliases": [], "realm": None, "personality": [],
                    "unresolved": [], "evidence": text, "state_changes": [{
                        "field": "goal", "operation": "clear", "previous": "寻找乙", "value": None, "evidence": text,
                    }],
                }]}
    bad = json.dumps(response)
    response["characters"][0]["state_changes"][0]["previous"] = "守城"
    llm = _RecordingLLM([bad, json.dumps(response)])
    update = memory.review(llm, text, "甲当前目标：守城", 1)
    assert memory.snapshot().characters["甲"].goal == "守城"
    assert "Stale goal update" in llm.prompts[1].task
    assert text in llm.prompts[1].task
    memory.commit(update, 1)
    assert memory.snapshot().characters["甲"].goal is None
    assert source.characters["甲"].goal == "守城"


def test_memory_quote_error_reports_all_locations_for_one_repair():
    payload = {"characters": [{"evidence": "改写一", "state_changes": [{"evidence": "改写二"}]}],
               "events": [{"evidence": "改写三"}]}
    with pytest.raises(LLMResponseError) as error:
        ContinuationMemory(StoryContext())._validate_quotes(payload, "甲把钥匙还给了乙。")
    assert "characters[0].evidence" in str(error.value)
    assert "characters[0].state_changes[0].evidence" in str(error.value)
    assert "events[0].evidence" in str(error.value)


def test_memory_recovers_unknown_thread_id_as_new_open_observation():
    source = StoryContext(
        characters={"甲": CharacterState(name="甲", unresolved=["寻找妹妹"])},
    )
    text = "甲在城门看见妹妹留下的红绳。"
    response = {
        "passed": True,
        "issues": [],
        "repair_instructions": [],
        "summary": "甲发现红绳",
        "characters": [{
            "name": "甲", "aliases": [], "role": "main", "realm": None,
            "personality": [], "unresolved": [], "status": "unknown",
            "location": "城门", "goal": None, "knowledge": [],
            "thread_updates": [{
                "operation": "update", "thread_id": "T-invented",
                "summary": "红绳的来历未明", "evidence": text,
            }],
            "state_changes": [], "evidence": text,
        }],
        "events": [],
        "relationships": [],
        "story_state": None,
    }
    memory = ContinuationMemory(source)
    update = memory.review(_ResponseLLM(json.dumps(response)), text, "已知状态", 1)

    assert update.validation.passed
    assert update.characters[0].thread_updates[0].operation == "add"
    assert update.characters[0].thread_updates[0].thread_id == ""
    memory.commit(update, 1)
    snapshot = memory.snapshot().characters["甲"]
    assert "红绳的来历未明" in snapshot.unresolved
    assert "寻找妹妹" in snapshot.unresolved


def test_quote_repair_supplies_original_passage_but_still_rejects_near_match():
    text = "他说：“站到我身边来。这里有张椅子，你坐。”"
    payload = {"characters": [{"evidence": "他说：“站到我身边来。”"}], "events": []}
    with pytest.raises(LLMResponseError) as error:
        ContinuationMemory(StoryContext())._validate_quotes(payload, text)
    assert text in str(error.value)
    assert "characters[0].evidence" in str(error.value)


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
