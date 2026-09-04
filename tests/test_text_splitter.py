from __future__ import annotations

from story_for_you.parser.text_splitter import TextSplitter


def test_splitter_keeps_configured_overlap():
    text = "abcdefghijklmnopqrstuvwxyz"
    splitter = TextSplitter(chunk_size=10, overlap=3)

    chunks = splitter.split(text)

    assert len(chunks) > 1
    assert chunks[0].content[-3:] == chunks[1].content[:3]


def test_splitter_positions_match_trimmed_content():
    text = "  abcdefghij  klmnopqrst  "
    chunks = TextSplitter(chunk_size=12, overlap=3).split(text)

    for chunk in chunks:
        assert text[chunk.start_pos : chunk.end_pos] == chunk.content


def test_splitter_uses_chinese_numbered_chapters_and_skips_front_matter():
    text = (
        "版权信息\n出版社\n\n题记\n说明文字\n\n"
        "一\n第一章正文。\n\n二\n第二章正文。"
    )

    chunks = TextSplitter(
        chunk_size=100,
        overlap=20,
        preserve_chapter_boundaries=True,
    ).split(text)

    assert [chunk.chapter for chunk in chunks] == ["1", "2"]
    assert [chunk.content for chunk in chunks] == ["一\n第一章正文。", "二\n第二章正文。"]
    assert all("版权信息" not in chunk.content for chunk in chunks)


def test_splitter_does_not_overlap_long_recognized_chapters():
    text = "一\n" + ("甲" * 20) + "。\n二\n" + ("乙" * 20) + "。"

    chunks = TextSplitter(
        chunk_size=10,
        overlap=4,
        preserve_chapter_boundaries=True,
    ).split(text)

    for first, second in zip(chunks, chunks[1:]):
        if first.chapter == second.chapter:
            assert first.end_pos <= second.start_pos
