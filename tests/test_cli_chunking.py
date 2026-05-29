from __future__ import annotations

from story_for_you.cli.main import _split_analysis_text, _split_text
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
