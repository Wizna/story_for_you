from __future__ import annotations

from story_for_you.cli.main import _split_text
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
