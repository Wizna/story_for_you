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


def test_splitter_uses_nested_volume_chapters_and_keeps_prologue():
    text = (
        "第一卷 开端\n楔子正文。\n"
        "第一卷 开端 第一章 相逢\n第一章正文。\n"
        "第二卷 续篇\n"
        "第二卷 续篇 第一章 重逢\n第二章正文。"
    )

    chunks = TextSplitter(
        chunk_size=100,
        preserve_chapter_boundaries=True,
    ).split(text)

    assert [chunk.chapter for chunk in chunks] == ["1", "2", "3"]
    assert [chunk.content for chunk in chunks] == [
        "第一卷 开端\n楔子正文。",
        "第一卷 开端 第一章 相逢\n第一章正文。",
        "第二卷 续篇\n第二卷 续篇 第一章 重逢\n第二章正文。",
    ]


def test_splitter_reuses_source_chapter_label_for_long_chapter_fragments():
    text = "第一章 长章\n" + ("正文。" * 20) + "\n第二章 短章\n结尾。"

    chunks = TextSplitter(
        chunk_size=20,
        preserve_chapter_boundaries=True,
    ).split(text)

    labels = [chunk.chapter for chunk in chunks]
    assert labels.count("1") > 1
    assert labels[-1] == "2"
    assert labels == sorted(labels, key=int)


def test_splitter_removes_explicit_download_promotion_from_final_chunk():
    text = (
        "第一章 正文\n故事完结。\n"
        "==========\n更多精校小说尽在：https://example.test\n==========\n"
    )

    chunks = TextSplitter(
        chunk_size=100,
        preserve_chapter_boundaries=True,
    ).split(text)

    assert [chunk.content for chunk in chunks] == ["第一章 正文\n故事完结。"]
