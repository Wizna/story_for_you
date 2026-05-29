from __future__ import annotations

from story_for_you.analysis.context import CharacterState, StoryContext
from story_for_you.core.ending_writer import EndingWriter
from story_for_you.indexer.segment import Segment, SegmentIndex
from story_for_you.llm.base import LLMProvider, LLMResponse


class _FakeLLM(LLMProvider):
    def generate(self, prompt: str, system: str = "", options: dict | None = None) -> LLMResponse:
        return LLMResponse(content="", tokens_used=0)

    def generate_stream(self, prompt: str, system: str = "", options: dict | None = None):
        yield from []


def test_recent_segment_digest_uses_tail_excerpt():
    context = StoryContext(
        characters={"翠翠": CharacterState(name="翠翠", role="main")},
    )
    segment = Segment(
        segment_id=1,
        content="开头版权信息。" + ("中段。" * 100) + "结尾处翠翠守着渡口，杨马兵陪在旁边。",
        chapter=1,
        characters=["翠翠"],
    )
    index = SegmentIndex(
        segments=[segment],
        char_index={"翠翠": [1]},
        chapter_index={1: [1]},
        gap_map={},
    )
    writer = EndingWriter(_FakeLLM(), index)

    digest = writer._recent_segment_digest(context)

    assert "结尾处翠翠守着渡口" in digest
    assert "开头版权信息" not in digest
