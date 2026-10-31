import json

import pytest

from story_for_you.analysis.context import CharacterState
from story_for_you.analysis.extractors.characters import CharacterExtractor
from story_for_you.analysis.extractors.events import EventExtractor
from story_for_you.analysis.extractors.relationships import RelationshipMapper
from story_for_you.analysis.threads import StoryThread, ThreadEvidence
from story_for_you.exceptions import LLMResponseError
from tests.test_cacheable_prompting import _RecordingLLM


def _event(evidence):
    return {"event_id": "E1", "chapter": 1, "type": "progress", "participants": ["甲"],
            "summary": "甲交出铜钥匙", "impact": {"power_shifts": {}, "relation_changes": {}, "world_flags": []},
            "is_irreversible": False, "deceased_characters": [], "evidence": evidence}


def test_event_quote_repair_receives_original_source_and_preserves_valid_events():
    source = "甲放下铜钥匙，转身离开。"
    llm = _RecordingLLM([json.dumps([_event("甲交出了钥匙。")]), json.dumps([_event(source)])])
    events = EventExtractor(llm).extract(source, ["甲"], 1, "")
    assert events[0].evidence == source
    assert llm.prompts[1].prefix == source
    assert "deceased_characters" in llm.prompts[1].task
    assert "E1" in llm.prompts[1].task
    assert len(llm.prompts) == 2


@pytest.mark.parametrize("repaired_evidence", ["甲放下铁钥匙。", ""])
def test_event_quote_still_fails_after_one_invalid_repair(repaired_evidence):
    llm = _RecordingLLM([json.dumps([_event("改写的句子")]), json.dumps([_event(repaired_evidence)])])
    with pytest.raises(LLMResponseError, match="failed after repair"):
        EventExtractor(llm).extract("甲放下铜钥匙。", ["甲"], 1, "")
    assert len(llm.prompts) == 2


def test_character_quote_repair_preserves_explicit_state_changes():
    source = "甲终于找到了乙。"
    character = {"name": "甲", "role": "main", "aliases": [], "realm": None,
                 "personality": [], "unresolved": [], "state_changes": [
                     {"field": "goal", "operation": "clear", "previous": "寻找乙",
                      "value": None, "evidence": "甲找到了乙。"},
                 ]}
    bad = json.dumps([character])
    character["state_changes"][0]["evidence"] = source
    llm = _RecordingLLM([bad, json.dumps([character])])
    result = CharacterExtractor(llm).extract(source)
    assert result[0].state_changes[0].operation == "clear"
    assert llm.prompts[1].prefix == source
    assert "state_changes" in llm.prompts[1].task
    assert "甲.state_changes[0].evidence" in llm.prompts[1].task


def test_character_death_requires_explicit_source_evidence():
    dead = {"name": "甲", "role": "main", "aliases": [], "realm": None,
            "personality": [], "unresolved": [], "status": "dead"}
    llm = _RecordingLLM([json.dumps([dead]), json.dumps([dead])])
    with pytest.raises(LLMResponseError, match="failed after repair"):
        CharacterExtractor(llm).extract("甲只是受了重伤。")


def test_character_death_status_is_kept_when_source_evidence_is_explicit():
    source = "甲倒在血泊中，再也没有起来。"
    dead = {"name": "甲", "role": "main", "aliases": [], "realm": None,
            "personality": [], "unresolved": [], "status": "dead", "status_evidence": source}
    result = CharacterExtractor(_RecordingLLM([json.dumps([dead])])).extract(source)
    assert result[0].status == "dead"
    assert result[0].status_evidence == source


def test_character_extraction_keeps_ninth_participant_for_relationships():
    names = [f"人物{i}" for i in range(9)]
    llm = _RecordingLLM([json.dumps([
        {"name": name, "role": "support", "aliases": [], "realm": None,
         "personality": [], "unresolved": []} for name in names
    ])])
    characters = CharacterExtractor(llm).extract("、".join(names) + "走进会场。")
    assert [character.name for character in characters] == names


def test_long_novel_character_roster_is_bounded_and_mentions_are_kept():
    names = [CharacterState(name=f"人物{i}", role="minor") for i in range(120)]
    names[117] = CharacterState(name="关键人物", role="support")
    extractor = CharacterExtractor(_RecordingLLM(["[]"]), max_known_characters=12)

    prompt = extractor._build_prompt("关键人物走进厅中。", names)

    marker = "**已知人物与伏笔 ID**：`"
    roster = json.loads(prompt.task.split(marker, 1)[1].split("`", 1)[0])
    assert len(roster) == 12
    assert any(item["name"] == "关键人物" for item in roster)


def test_long_novel_roster_keeps_mentioned_one_character_name():
    names = [CharacterState(name=name, role="minor") for name in ["甲", "乙", "丙"]]
    extractor = CharacterExtractor(_RecordingLLM(["[]"]), max_known_characters=1)

    prompt = extractor._build_prompt("乙走进厅中。", names)

    marker = "**已知人物与伏笔 ID**：`"
    roster = json.loads(prompt.task.split(marker, 1)[1].split("`", 1)[0])
    assert [item["name"] for item in roster] == ["乙"]


def test_known_character_roster_does_not_replay_all_thread_evidence():
    character = CharacterState(
        name="甲",
        threads=[
            StoryThread(
                thread_id=f"T-{index}",
                summary=f"线索{index}",
                history=[ThreadEvidence(f"线索{index}", f"证据{index}", source_unit=index)],
            )
            for index in range(20)
        ],
    )
    extractor = CharacterExtractor(_RecordingLLM(["[]"]), max_known_characters=1)

    prompt = extractor._build_prompt("甲在城门停下。", [character])

    marker = "**已知人物与伏笔 ID**：`"
    roster = json.loads(prompt.task.split(marker, 1)[1].split("`", 1)[0])
    assert len(roster[0]["threads"]) == 8
    assert "history" not in roster[0]["threads"][0]


def test_relationship_retry_receives_rejected_name_and_source():
    relation = {"source": "甲", "targets": ["丙"], "relation_type": "盟友",
                "sentiment": "positive", "description": "结盟"}
    bad = json.dumps([relation])
    relation["targets"] = ["乙"]
    llm = _RecordingLLM([bad, json.dumps([relation])])
    relationships = RelationshipMapper(llm).map("甲和乙结为盟友。", ["甲", "乙"])
    assert relationships[0].targets == ["乙"]
    assert llm.prompts[1].prefix == "甲和乙结为盟友。"
    assert "must come from roster: 丙" in llm.prompts[1].task


def test_relationship_mapper_can_require_source_evidence_for_analysis():
    relation = {"source": "甲", "targets": ["乙"], "relation_type": "盟友",
                "sentiment": "positive", "description": "结盟", "evidence": "甲和乙结为盟友。"}
    llm = _RecordingLLM([json.dumps([relation])])
    relationships = RelationshipMapper(llm, require_evidence=True).map(
        "甲和乙结为盟友。", ["甲", "乙"]
    )
    assert relationships[0].evidence == "甲和乙结为盟友。"


def test_relationship_mapper_rejects_missing_source_evidence_when_strict():
    relation = {"source": "甲", "targets": ["乙"], "relation_type": "盟友",
                "sentiment": "positive", "description": "结盟"}
    llm = _RecordingLLM([json.dumps([relation]), json.dumps([relation])])
    with pytest.raises(LLMResponseError, match="failed after retry"):
        RelationshipMapper(llm, require_evidence=True).map(
            "甲和乙结为盟友。", ["甲", "乙"]
        )
