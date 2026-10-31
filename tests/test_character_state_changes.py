import json

import pytest

from story_for_you.analysis.context import CharacterState, StoryContext
from story_for_you.analysis.layers.state_store import StateStore
from story_for_you.analysis.state_changes import StateChange
from story_for_you.core.ending.memory import ContinuationMemory
from story_for_you.exceptions import LLMResponseError
from tests.test_review_fixes import _ResponseLLM


def test_completed_goal_and_obsolete_knowledge_can_be_retired_with_history():
    store = StateStore()
    store.update([CharacterState("甲", goal="寻找乙", knowledge=["口令是落霞"])], [], [])
    store.update([CharacterState("甲")], [], [])
    assert store.characters_snapshot()["甲"].goal == "寻找乙"
    store.update([CharacterState("甲", state_changes=[
        StateChange("goal", "clear", "甲终于找到了乙。", previous="寻找乙"),
        StateChange("knowledge", "replace", "乙说口令已改成归舟。", "口令是归舟", "口令是落霞"),
    ])], [], [])
    restored = StateStore.from_dict(store.to_dict()).characters_snapshot()["甲"]
    assert restored.goal is None
    assert restored.knowledge == ["口令是归舟"]
    assert len(restored.state_history) == 2
    assert restored.state_changes == []


def test_state_change_rejects_stale_previous_value():
    store = StateStore()
    store.update([CharacterState("甲", goal="守城")], [], [])
    with pytest.raises(LLMResponseError, match="Stale"):
        store.update([CharacterState("甲", state_changes=[
            StateChange("goal", "clear", "甲已经找到了乙。", previous="寻找乙"),
        ])], [], [])
    assert store.characters_snapshot()["甲"].goal == "守城"


def test_generated_state_correction_requires_its_own_quote():
    payload = {
        "passed": True, "issues": [], "repair_instructions": [], "summary": "甲在等候",
        "characters": [{
            "name": "甲", "role": "main", "aliases": [], "realm": None,
            "personality": [], "unresolved": [], "evidence": "甲仍在城门等候。",
            "state_changes": [{"field": "goal", "operation": "clear", "previous": "寻找乙",
                               "value": None, "evidence": "甲找到了乙。"}],
        }], "events": [], "story_state": None,
    }
    with pytest.raises(LLMResponseError, match="evidence"):
        ContinuationMemory(StoryContext()).review(
            _ResponseLLM(json.dumps(payload)), "甲仍在城门等候。", "已知状态", 1,
        )
