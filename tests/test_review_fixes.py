from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from story_for_you.analysis.context import CharacterState, Relationship, StoryContext
from story_for_you.analysis.extractors.style import StyleExtractor
from story_for_you.analysis.layers.state_store import StateStore
from story_for_you.cache.store import ContextStore
from story_for_you.cli.main import _chunks_to_segments, app
from story_for_you.config.settings import Settings
from story_for_you.core.character_filter import CharacterFilter
from story_for_you.core.character_remover import CharacterRemover
from story_for_you.core.exceptions import LLMResponseError
from story_for_you.indexer.retriever import SegmentRetriever
from story_for_you.indexer.segment import Segment, SegmentIndex
from story_for_you.indexer.service import SegmentIndexService
from story_for_you.llm.base import LLMProvider, LLMResponse
from story_for_you.parser.text_splitter import TextChunk
from story_for_you.utils.prompting import CacheablePrompt


class _ResponseLLM(LLMProvider):
    def __init__(self, content: str):
        self.content = content

    def generate(
        self,
        prompt: CacheablePrompt,
        system: str = "",
        options: dict | None = None,
    ) -> LLMResponse:
        return LLMResponse(content=self.content, tokens_used=0)

    def generate_stream(
        self,
        prompt: CacheablePrompt,
        system: str = "",
        options: dict | None = None,
    ):
        yield from ()


def _style_payload() -> dict:
    return {
        "avg_sentence_length": 20,
        "sentence_variety": "varied",
        "paragraph_density": "medium",
        "register": "literary",
        "characteristic_words": ["静"],
        "idiom_frequency": "sparse",
        "metaphor_style": "自然意象",
        "description_focus": ["psychological"],
        "parallelism_use": "rare",
        "tone_markers": ["呢"],
        "narrator_style": "intimate",
        "representative_samples": [
            {"source_chapter": 1, "content": "风从窗外来。", "style_notes": "短句克制"}
        ],
        "style_summary": "克制细腻，重视心理描写。",
    }


def test_chunks_to_segments_removes_source_overlap():
    segments = _chunks_to_segments(
        [
            TextChunk("abcdefghij", 0, 10),
            TextChunk("hijklmnopq", 7, 17),
        ]
    )

    assert [segment.content for segment in segments] == ["abcdefghij", "klmnopq"]


def test_state_store_resolves_aliases_for_relationships():
    store = StateStore()
    store.update(
        [
            CharacterState(name="天保", aliases=["大老"], role="support"),
            CharacterState(name="翠翠", aliases=["小翠"], role="main"),
        ],
        [],
        [],
    )
    store.update(
        [],
        [
            Relationship(
                source="大老",
                targets=["小翠"],
                relation_type="保护",
                sentiment="positive",
                description="兄长保护翠翠",
            )
        ],
        [],
    )

    relationship = store.characters_snapshot()["天保"].relationships[0]
    assert relationship.source == "天保"
    assert relationship.targets == ["翠翠"]


def test_state_store_merges_possessive_kinship_name_variants():
    store = StateStore()
    store.update(
        [
            CharacterState(name="翠翠的母亲", role="minor"),
            CharacterState(name="翠翠母亲", role="minor"),
        ],
        [],
        [],
    )

    assert list(store.characters_snapshot()) == ["翠翠的母亲"]


def test_state_store_does_not_merge_distinct_people_with_generic_title():
    store = StateStore()
    store.update(
        [
            CharacterState(name="李长老", aliases=["师父"], role="support"),
            CharacterState(name="王长老", aliases=["师父"], role="support"),
        ],
        [],
        [],
    )

    assert set(store.characters_snapshot()) == {"李长老", "王长老"}


def test_state_store_accepts_new_affiliation_observation():
    store = StateStore()
    store.update([CharacterState(name="林凡", realm="青云宗")], [], [])
    store.update([CharacterState(name="林凡", realm="魔宗")], [], [])

    assert store.characters_snapshot()["林凡"].realm == "魔宗"


def test_state_store_replaces_current_unresolved_observation():
    store = StateStore()
    store.update([CharacterState(name="林凡", unresolved=["寻找妹妹"])], [], [])
    store.update([CharacterState(name="林凡", unresolved=[])], [], [])

    assert store.characters_snapshot()["林凡"].unresolved == []


def test_state_store_keeps_relationships_for_normalized_canonical_name():
    store = StateStore()
    store.update(
        [
            CharacterState(name="翠翠的母亲", role="minor"),
            CharacterState(name="翠翠", role="main"),
        ],
        [],
        [],
    )
    store.update(
        [],
        [
            Relationship(
                source="翠翠的母亲",
                targets=["翠翠"],
                relation_type="母女",
                sentiment="positive",
                description="母亲牵挂女儿",
            )
        ],
        [],
    )

    relationship = store.characters_snapshot()["翠翠的母亲"].relationships[0]
    assert relationship.source == "翠翠的母亲"
    assert relationship.targets == ["翠翠"]


def test_context_prompt_includes_character_relationships():
    context = StoryContext(
        characters={
            "甲": CharacterState(
                name="甲",
                role="main",
                relationships=[
                    Relationship(
                        source="甲",
                        targets=["乙"],
                        relation_type="盟友",
                        sentiment="positive",
                        description="并肩作战",
                    )
                ],
            )
        }
    )

    rendered = context.for_prompt()["characters"]
    assert "relations" in rendered
    assert "乙: 盟友/positive" in rendered


def test_context_prompt_includes_dynamic_character_state():
    context = StoryContext(
        characters={
            "林凡": CharacterState(
                name="林凡",
                role="main",
                status="alive",
                location="城门",
                goal="阻止决战",
                knowledge=["敌军已入城"],
            )
        }
    )

    rendered = context.for_prompt()["characters"]

    assert "location=城门" in rendered
    assert "goal=阻止决战" in rendered
    assert "敌军已入城" in rendered


def test_cache_directory_changes_when_analysis_settings_change(tmp_path):
    source = tmp_path / "novel.txt"
    source.write_text("正文", encoding="utf-8")
    store = ContextStore(tmp_path / "cache")
    settings = Settings()

    first = store._get_cache_dir(source, settings)
    settings.analysis.target_unit_chars += 1
    second = store._get_cache_dir(source, settings)

    assert first != second


def _removal_index() -> SegmentIndex:
    segments = [
        Segment(1, "甲与乙相遇。", chapter=1, characters=["乙"]),
        Segment(2, "甲独自离开。", chapter=1, characters=["甲"]),
    ]
    return SegmentIndex(
        segments=segments,
        char_index={"甲": [2], "乙": [1]},
        chapter_index={1: [1, 2]},
        gap_map={},
    )


def test_hard_removal_rejects_rewrite_that_mentions_target():
    response = json.dumps({"action": "rewrite", "content": "乙仍然站在门口。", "reason": "保留"}, ensure_ascii=False)
    remover = CharacterRemover(
        _ResponseLLM(response),
        SegmentRetriever(_removal_index()),
    )
    context = StoryContext(characters={"乙": CharacterState(name="乙", role="support")})

    with pytest.raises(LLMResponseError, match="still mentions"):
        remover.remove("甲与乙相遇。\n\n甲独自离开。", ["乙"], context, mode="hard")


def test_removal_keeps_generated_rewrite_length_when_spans_overlap():
    segments = [
        Segment(1, "原文前段", chapter=1, characters=[], metadata={"start": 0, "end": 4}),
        Segment(2, "包含乙的原文", chapter=1, characters=["乙"], metadata={"start": 2, "end": 8}),
    ]
    index = SegmentIndex(
        segments=segments,
        char_index={"乙": [2]},
        chapter_index={1: [1, 2]},
        gap_map={},
    )
    response = json.dumps({"action": "rewrite", "content": "改写后的完整内容", "reason": "移除人物"}, ensure_ascii=False)
    remover = CharacterRemover(_ResponseLLM(response), SegmentRetriever(index))

    result = remover.remove("原文前段\n\n包含乙的原文", ["乙"], StoryContext(), mode="hard")

    assert "改写后的完整内容" in result.content


def test_segment_index_supports_alias_filtering_and_removal():
    context = StoryContext(characters={"天保": CharacterState(name="天保", aliases=["大老"], role="support")})
    segments = [Segment(1, "大老走进院子。", chapter=1)]
    index = SegmentIndexService().build(context, segments)

    assert SegmentRetriever(index).retrieve_by_characters(["大老"], mode="strict")
    assert not SegmentRetriever(index).retrieve_excluding(["大老"], mode="hard")


def test_filter_rejects_unknown_character_instead_of_writing_empty_file():
    filterer = CharacterFilter(_ResponseLLM("unused"), SegmentRetriever(_removal_index()))
    with pytest.raises(LLMResponseError, match="No story segments matched"):
        filterer.filter("正文", ["不存在"], StoryContext(), mode="strict")


def test_style_sample_drops_front_matter_before_first_chapter():
    extractor = StyleExtractor(_ResponseLLM("{}"))
    text = "版权信息\n作者简介\n\n第1章 初见\n" + ("正文叙事。" * 600)

    sample = extractor._select_samples([text, "第二章\n正文。" * 100])[0][1]

    assert "版权信息" not in sample
    assert "第1章" in sample


def test_style_front_matter_recognizes_chapter_title_suffix():
    extractor = StyleExtractor(_ResponseLLM("{}"))

    cleaned = extractor._strip_front_matter("版权信息\n\n第1章 初见\n正文")

    assert cleaned.startswith("第1章 初见")


def test_style_inject_requires_context_before_llm(tmp_path):
    input_file = tmp_path / "novel.txt"
    input_file.write_text("正文", encoding="utf-8")

    result = CliRunner().invoke(app, ["style", str(input_file), "--inject"])

    assert result.exit_code != 0
    assert "--inject requires --context" in result.output


def test_style_inject_updates_context_json(tmp_path, monkeypatch):
    input_file = tmp_path / "novel.txt"
    context_file = tmp_path / "context.json"
    input_file.write_text("正文", encoding="utf-8")
    context_file.write_text(json.dumps(StoryContext().to_dict()), encoding="utf-8")
    monkeypatch.setattr(
        "story_for_you.cli.main._build_cli_llm",
        lambda settings: _ResponseLLM(json.dumps(_style_payload(), ensure_ascii=False)),
    )

    result = CliRunner().invoke(
        app,
        ["style", str(input_file), "--context", str(context_file), "--inject"],
    )

    assert result.exit_code == 0, result.output
    saved = StoryContext.from_dict(json.loads(context_file.read_text(encoding="utf-8")))
    assert saved.writing_style is not None
    assert saved.writing_style.style_summary == "克制细腻，重视心理描写。"


def test_analyze_rejects_empty_input_before_building_llm(tmp_path):
    input_file = tmp_path / "empty.txt"
    input_file.write_text("\n  \n", encoding="utf-8")

    result = CliRunner().invoke(app, ["analyze", str(input_file), "--no-resume"])

    assert result.exit_code != 0
    assert "Input file is empty" in result.output
