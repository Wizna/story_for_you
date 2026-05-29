from __future__ import annotations

from story_for_you.parser.text_splitter import TextSplitter


def test_splitter_keeps_configured_overlap():
    text = "abcdefghijklmnopqrstuvwxyz"
    splitter = TextSplitter(chunk_size=10, overlap=3)

    chunks = splitter.split(text)

    assert len(chunks) > 1
    assert chunks[0].content[-3:] == chunks[1].content[:3]
