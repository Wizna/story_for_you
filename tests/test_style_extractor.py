from __future__ import annotations

import json
import pytest

from story_for_you.analysis.extractors.style import StyleExtractor
from story_for_you.llm.base import LLMProvider, LLMResponse
from story_for_you.utils.prompting import CacheablePrompt
from story_for_you.analysis.context import StyleSample, WritingStyle
from story_for_you.core.prompting import format_style_samples
from story_for_you.exceptions import LLMResponseError
from tests.test_review_fixes import _ResponseLLM, _style_payload


class _FakeLLM(LLMProvider):
    def generate(self, prompt: CacheablePrompt, system: str = "", options: dict | None = None) -> LLMResponse:
        return LLMResponse(content="{}", tokens_used=0)

    def generate_stream(self, prompt: CacheablePrompt, system: str = "", options: dict | None = None):
        yield from []


def test_single_chunk_style_sample_uses_structural_middle_slice():
    extractor = StyleExtractor(_FakeLLM())
    text = "开头。" * 800 + "中段风格样本。" * 80 + "结尾。" * 800

    samples = extractor._select_samples([text])

    assert len(samples) == 1
    assert "中段风格样本" in samples[0][1]


def test_long_novel_style_includes_actual_latest_units():
    assert StyleExtractor(_FakeLLM())._sample_indices(100) == [0, 50, 98, 99]


def test_style_samples_prioritize_current_speaker_and_scene():
    style = WritingStyle(20, "mixed", "medium", "literary", representative_samples=[
        StyleSample(90, "乙的对白", "闲谈", "乙", "闲谈"),
        StyleSample(1, "甲在审讯中说话", "克制追问", "甲", "审讯"),
    ])
    result = format_style_samples(style, max_samples=1, focus_characters=["甲"], scene_query="审讯")
    assert "甲在审讯中说话" in result
    assert "乙的对白" not in result


def test_style_extraction_rejects_fabricated_representative_quotes():
    with pytest.raises(LLMResponseError, match="quote"):
        StyleExtractor(_ResponseLLM(json.dumps(_style_payload()))).extract_from_raw(["这里没有风。"])
