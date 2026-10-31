from story_for_you.analysis.context import CharacterState, EventImpact, PlotEvent, StoryContext
from story_for_you.core.ending.retrieval import EvidenceRetriever
from story_for_you.core.ending_writer import EndingWriter
from story_for_you.indexer.segment import Segment, SegmentIndex
from tests.test_review_fixes import _ResponseLLM


def test_object_clue_is_recalled_without_current_cast_participation():
    clue = PlotEvent("clue", 1, "reveal", ["乙"], "青铜匣的开匣暗号是雨歇", EventImpact(),
                     evidence="乙说打开青铜匣须念雨歇二字。")
    other = [PlotEvent(f"e{i}", i + 2, "progress", ["丙"], f"商队第{i}日赶路", EventImpact()) for i in range(8)]
    source = clue.evidence + "丙赶路。" * 1600 + "甲发现了青铜匣。"
    segments = [Segment(1, clue.evidence, characters=["乙"]), Segment(2, "甲发现了青铜匣。", characters=["甲"])]
    context = StoryContext(characters={"甲": CharacterState("甲", role="main")}, events=[clue, *other])
    writer = EndingWriter(_ResponseLLM(""), SegmentIndex(segments, {"甲": [2], "乙": [1]}, {}, {}))
    block = writer._build_continuation_context(source, context, focus_characters=["甲"])
    assert "雨歇" in block
    assert "不代表当前角色已知" in block
    assert context.characters["甲"].knowledge == []


def test_retrieval_budget_keeps_relevant_events_in_chronological_order():
    events = [PlotEvent(f"e{i}", i, "progress", ["甲"], f"第{i}次检查青铜匣", EventImpact()) for i in range(1, 40)]
    retriever = EvidenceRetriever(SegmentIndex([], {}, {}, {}))
    selected, passages = retriever.retrieve("青铜匣", events, ["甲"], max_events=3, max_chars=40)
    assert len(selected) == 3
    assert selected == sorted(selected, key=lambda event: event.chapter)
    assert sum(len(event.summary) for event in selected) <= 40
    assert not passages


def test_retrieve_passages_keeps_source_diversity():
    index = SegmentIndex(
        segments=[
            Segment(1, "早年有人提过迦楼罗与龙的旧画。" * 3, chapter=2),
            Segment(2, "中段只写了市集喧闹。" * 3, chapter=8),
            Segment(3, "末章又出现迦楼罗钳龙的图像。" * 3, chapter=20),
        ],
        char_index={}, chapter_index={}, gap_map={},
    )
    passages = EvidenceRetriever(index).retrieve_passages(
        "迦楼罗 钳龙 图像", max_passages=4, max_chars=5000, max_per_chapter=1,
    )

    assert [item.segment.segment_id for item in passages] == [1, 3]
