from __future__ import annotations

import pytest

from story_for_you.analysis.context import CharacterState, StoryContext
from story_for_you.analysis.extractors.chapters import ChapterSummarizer
from story_for_you.analysis.extractors.characters import CharacterExtractor
from story_for_you.analysis.extractors.events import EventExtractor
from story_for_you.analysis.extractors.relationships import RelationshipMapper
from story_for_you.analysis.extractors.style import StyleExtractor
from story_for_you.analysis.layers.state_store import StateStore
from story_for_you.core.ending.hint_interpreter import HintInterpreter
from story_for_you.core.exceptions import LLMResponseError
from story_for_you.llm.base import LLMProvider, LLMResponse


class _FakeLLM(LLMProvider):
    def __init__(self, content: str):
        self.content = content

    def generate(self, prompt: str, system: str = "", options: dict | None = None) -> LLMResponse:
        return LLMResponse(content=self.content, tokens_used=0)

    def generate_stream(self, prompt: str, system: str = "", options: dict | None = None):
        yield from []


class _SequencedLLM(LLMProvider):
    def __init__(self, contents: list[str]):
        self.contents = list(contents)

    def generate(self, prompt: str, system: str = "", options: dict | None = None) -> LLMResponse:
        if not self.contents:
            raise AssertionError("No more queued responses.")
        return LLMResponse(content=self.contents.pop(0), tokens_used=0)

    def generate_stream(self, prompt: str, system: str = "", options: dict | None = None):
        yield from []


def test_context_from_dict_rejects_defaulted_event_type():
    payload = {
        "metadata": {},
        "chapter_window": [],
        "events": [
            {
                "event_id": "E1",
                "chapter": 1,
                "participants": [],
                "summary": "事件",
                "impact": {"power_shifts": {}, "relation_changes": {}, "world_flags": []},
                "is_irreversible": False,
            }
        ],
        "characters": {},
        "story_state": None,
        "writing_style": None,
    }

    with pytest.raises(LLMResponseError, match="PlotEvent missing required field: type"):
        StoryContext.from_dict(payload)


def test_character_extractor_requires_list_fields():
    llm = _FakeLLM(
        '[{"name":"翠翠","aliases":"翠翠","role":"main","realm":null,'
        '"personality":[],"unresolved":[]}]'
    )

    with pytest.raises(LLMResponseError, match="JSON arrays"):
        CharacterExtractor(llm).extract("翠翠在渡口。")


def test_character_extractor_repairs_missing_required_field_once():
    llm = _SequencedLLM(
        [
            '[{"name":"翠翠","aliases":[],"role":"main","realm":null,"unresolved":[]}]',
            '[{"name":"翠翠","aliases":[],"role":"main","realm":null,"personality":[],"unresolved":[]}]',
        ]
    )

    characters = CharacterExtractor(llm).extract("翠翠在渡口。")

    assert [character.name for character in characters] == ["翠翠"]


def test_character_extractor_fails_after_repair_if_schema_still_invalid():
    llm = _SequencedLLM(
        [
            '[{"name":"翠翠","aliases":[],"role":"main","realm":null,"unresolved":[]}]',
            '[{"name":"翠翠","aliases":[],"role":"main","realm":null,"unresolved":[]}]',
        ]
    )

    with pytest.raises(LLMResponseError, match="failed after repair"):
        CharacterExtractor(llm).extract("翠翠在渡口。")


def test_event_extractor_requires_boolean_and_roster_names():
    llm = _FakeLLM(
        '[{"event_id":"CH001-E01","chapter":1,"type":"progress","participants":["陌生人"],'
        '"summary":"陌生人出现","impact":{"power_shifts":{},'
        '"relation_changes":{},"world_flags":[]},"is_irreversible":"false"}]'
    )

    with pytest.raises(LLMResponseError, match="must come from roster"):
        EventExtractor(llm).extract("正文", ["翠翠"], 1, "")


def test_event_extractor_accepts_explicit_aliases_from_roster():
    llm = _FakeLLM(
        '[{"event_id":"CH001-E01","chapter":1,"type":"progress","participants":["二老"],'
        '"summary":"傩送回到渡口","impact":{"power_shifts":{},'
        '"relation_changes":{},"world_flags":[]},"is_irreversible":false}]'
    )

    events = EventExtractor(llm).extract(
        "正文",
        [CharacterState(name="傩送", aliases=["二老"], role="main")],
        1,
        "",
    )

    assert events[0].participants == ["傩送"]


def test_relationship_mapper_rejects_names_outside_roster():
    llm = _FakeLLM(
        '[{"source":"翠翠","targets":["陌生人"],"relation_type":"误会",'
        '"sentiment":"negative","description":"短暂误会"}]'
    )

    with pytest.raises(LLMResponseError, match="must come from roster"):
        RelationshipMapper(llm).map("正文", ["翠翠", "傩送"])


def test_relationship_mapper_accepts_explicit_aliases_from_roster():
    llm = _FakeLLM(
        '[{"source":"二老","targets":["翠翠"],"relation_type":"恋慕",'
        '"sentiment":"positive","description":"二人心意相关"}]'
    )

    relationships = RelationshipMapper(llm).map(
        "正文",
        [
            CharacterState(name="傩送", aliases=["二老"], role="main"),
            CharacterState(name="翠翠", role="main"),
        ],
    )

    assert relationships[0].source == "傩送"
    assert relationships[0].targets == ["翠翠"]


def test_relationship_mapper_retries_transient_invalid_schema():
    llm = _SequencedLLM(
        [
            '[{"source":"翠翠","targets":[],"relation_type":"恋慕",'
            '"sentiment":"positive","description":"缺少目标"}]',
            '[{"source":"翠翠","targets":["傩送"],"relation_type":"恋慕",'
            '"sentiment":"positive","description":"关系明确"}]',
        ]
    )

    relationships = RelationshipMapper(llm).map("正文", ["翠翠", "傩送"])

    assert relationships[0].source == "翠翠"
    assert relationships[0].targets == ["傩送"]


def test_relationship_mapper_fails_after_retry_if_schema_remains_invalid():
    llm = _SequencedLLM(
        [
            '[{"source":"翠翠","targets":[],"relation_type":"恋慕",'
            '"sentiment":"positive","description":"缺少目标"}]',
            '[{"source":"翠翠","targets":[],"relation_type":"恋慕",'
            '"sentiment":"positive","description":"仍缺少目标"}]',
        ]
    )

    with pytest.raises(LLMResponseError, match="failed after retry"):
        RelationshipMapper(llm).map("正文", ["翠翠", "傩送"])


def test_chapter_summarizer_requires_typed_list_items():
    llm = _FakeLLM(
        '{"chapter":1,"title":"渡口","pov":"third-person","beats":["等待",42],'
        '"mood":"somber","synopsis":"翠翠等待。","irreversible_flags":[]}'
    )

    with pytest.raises(LLMResponseError, match="beats.*strings"):
        ChapterSummarizer(llm).summarize("正文", 1, "")


def test_chapter_summarizer_repairs_object_irreversible_flags():
    llm = _SequencedLLM(
        [
            '{"chapter":1,"title":"渡口","pov":"third-person","beats":["等待"],'
            '"mood":"somber","synopsis":"翠翠等待。",'
            '"irreversible_flags":[{"type":"death","reason":"爷爷已死"}]}',
            '{"chapter":1,"title":"渡口","pov":"third-person","beats":["等待"],'
            '"mood":"somber","synopsis":"翠翠等待。",'
            '"irreversible_flags":["death: 爷爷已死"]}',
        ]
    )

    summary = ChapterSummarizer(llm).summarize("正文", 1, "")

    assert summary.irreversible_flags == ["death: 爷爷已死"]


def test_chapter_summarizer_retries_after_failed_repair():
    llm = _SequencedLLM(
        [
            '{"chapter":1,"title":"渡口","pov":"third-person","beats":["等待"],'
            '"mood":"somber","synopsis":"翠翠等待。",'
            '"irreversible_flags":[{"type":"death","reason":"爷爷已死"}]}',
            '{"chapter":1,"title":"渡口","pov":"third-person","beats":["等待"],'
            '"mood":"somber","synopsis":"翠翠等待。",'
            '"irreversible_flags":[{"type":"death","reason":"爷爷已死"}]}',
            '{"chapter":1,"title":"渡口","pov":"third-person","beats":["等待"],'
            '"mood":"somber","synopsis":"翠翠等待。",'
            '"irreversible_flags":["death: 爷爷已死"]}',
        ]
    )

    summary = ChapterSummarizer(llm).summarize("正文", 1, "")

    assert summary.irreversible_flags == ["death: 爷爷已死"]


def test_style_extractor_requires_string_list_items():
    llm = _FakeLLM(
        '{"avg_sentence_length":18,"sentence_variety":"mixed","paragraph_density":"sparse",'
        '"register":"literary","characteristic_words":["渡口",7],"idiom_frequency":"sparse",'
        '"metaphor_style":"","description_focus":[],"parallelism_use":"rare",'
        '"tone_markers":[],"narrator_style":"detached","representative_samples":[],'
        '"style_summary":"清淡克制"}'
    )

    with pytest.raises(LLMResponseError, match="list items must be strings"):
        StyleExtractor(llm).extract(["正文"], [])


def test_hint_interpreter_requires_normalized_text_from_llm():
    llm = _FakeLLM(
        '{"normalized_text":"","closure":"closed","ending_direction":null,'
        '"emotional_tone":null,"focus_characters":[],"required_outcomes":[],'
        '"forbidden_outcomes":[],"required_resolutions":[],"style_constraints":[]}'
    )

    with pytest.raises(LLMResponseError, match="normalized_text"):
        HintInterpreter(llm).interpret("非开放式结局", StoryContext())


def test_character_merge_only_uses_explicit_aliases():
    extractor = CharacterExtractor(_FakeLLM("[]"))
    merged = extractor.merge_aliases(
        [
            CharacterState(name="傩送", role="main"),
            CharacterState(name="傩送二老", role="support"),
        ]
    )

    assert [character.name for character in merged] == ["傩送", "傩送二老"]


def test_state_store_from_dict_rejects_defaulted_story_state():
    with pytest.raises(LLMResponseError, match="StoryState missing required field"):
        StateStore.from_dict({"characters": {}, "story_state": {"world_tension": "low"}, "event_log": []})
