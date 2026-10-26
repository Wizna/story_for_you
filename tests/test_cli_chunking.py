from __future__ import annotations

from story_for_you.cli.main import _analysis_progress_key, _split_analysis_text, _split_text
from story_for_you.config.settings import Settings


def test_split_text_uses_context_window_not_output_tokens():
    settings = Settings()
    settings.llm.max_tokens = 1024
    settings.llm.context_window = 20000
    settings.prompt.margin = 1000
    settings.parser.chunk_size = 15000
    settings.parser.overlap = 100

    text = "。".join(["段落"] * 6000)
    chunks = _split_text(text, settings)

    assert chunks
    assert len(chunks[0].content) > settings.llm.max_tokens
    assert len(chunks[0].content) <= settings.parser.chunk_size


def test_split_text_caps_chunk_size_to_context_budget():
    settings = Settings()
    settings.llm.context_window = 10000
    settings.prompt.margin = 2000
    settings.parser.chunk_size = 20000
    settings.parser.overlap = 100

    text = "。".join(["段落"] * 6000)
    chunks = _split_text(text, settings)

    assert chunks
    assert len(chunks[0].content) <= settings.llm.context_window - settings.prompt.margin


def test_analysis_split_uses_analysis_granularity_under_large_context():
    settings = Settings()
    settings.llm.context_window = 1_000_000
    settings.parser.chunk_size = 120000
    settings.parser.overlap = 2000
    settings.analysis.target_unit_chars = 8000
    settings.analysis.min_units = 8

    text = "。".join(["段落"] * 20000)
    chunks = _split_analysis_text(text, settings)

    assert len(chunks) > 1
    assert len(chunks[0].content) <= settings.analysis.target_unit_chars


def test_analysis_split_prefers_real_chapter_boundaries_when_available():
    settings = Settings()
    settings.analysis.target_unit_chars = 1000
    text = "版权信息\n\n一\n甲。\n\n二\n乙。\n\n三\n丙。"

    chunks = _split_analysis_text(text, settings)

    assert [chunk.chapter for chunk in chunks] == ["1", "2", "3"]
    assert [chunk.content for chunk in chunks] == ["一\n甲。", "二\n乙。", "三\n丙。"]


def test_analysis_progress_key_changes_with_settings_and_chunk_boundaries(tmp_path):
    source = tmp_path / "story.txt"
    source.write_text("第一章\n甲。\n第二章\n乙。", encoding="utf-8")
    settings = Settings()
    chunks = _split_analysis_text(source.read_text(encoding="utf-8"), settings)
    original = _analysis_progress_key(source, settings, chunks)

    settings.analysis.window_size += 1
    assert _analysis_progress_key(source, settings, chunks) != original
    settings.analysis.window_size -= 1
    settings.llm.model = "another-model"
    assert _analysis_progress_key(source, settings, chunks) != original
    settings.llm.model = Settings().llm.model
    chunks[0].end_pos += 1
    assert _analysis_progress_key(source, settings, chunks) != original
