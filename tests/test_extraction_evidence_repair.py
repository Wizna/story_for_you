import json

import pytest

from story_for_you.analysis.extractors.characters import CharacterExtractor
from story_for_you.analysis.extractors.events import EventExtractor
from story_for_you.analysis.extractors.relationships import RelationshipMapper
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


def test_character_extraction_keeps_ninth_participant_for_relationships():
    names = [f"人物{i}" for i in range(9)]
    llm = _RecordingLLM([json.dumps([
        {"name": name, "role": "support", "aliases": [], "realm": None,
         "personality": [], "unresolved": []} for name in names
    ])])
    characters = CharacterExtractor(llm).extract("、".join(names) + "走进会场。")
    assert [character.name for character in characters] == names


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
