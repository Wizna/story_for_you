from __future__ import annotations

import json
import pytest

from story_for_you.analysis.context import CharacterState, StoryContext
from story_for_you.core.exceptions import GenerationError
from story_for_you.core.ending_writer import EndingWriter
from story_for_you.indexer.segment import Segment, SegmentIndex
from story_for_you.llm.base import LLMProvider, LLMResponse


class _FakeLLM(LLMProvider):
    def generate(self, prompt: str, system: str = "", options: dict | None = None) -> LLMResponse:
        return LLMResponse(content="", tokens_used=0)

    def generate_stream(self, prompt: str, system: str = "", options: dict | None = None):
        yield from []


class _BlockedResolutionLLM(LLMProvider):
    def generate(self, prompt: str, system: str = "", options: dict | None = None) -> LLMResponse:
        payload = {
            "status": "blocked",
            "missing_threads": ["傩送归宿未交代"],
            "bridges": [],
            "notes": "无法自然补写",
        }
        return LLMResponse(content=json.dumps(payload, ensure_ascii=False), tokens_used=0)

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


def test_resolution_blocked_status_raises():
    context = StoryContext(
        characters={"翠翠": CharacterState(name="翠翠", role="main", unresolved=["等待傩送"])},
    )
    writer = EndingWriter(
        _BlockedResolutionLLM(),
        SegmentIndex(segments=[], char_index={}, chapter_index={}, gap_map={}),
    )

    with pytest.raises(GenerationError, match="傩送归宿未交代"):
        writer._phase_resolution_review("正文", context, "上下文", None, "{}")
