from __future__ import annotations

from story_for_you.analysis.extractors.style import StyleExtractor
from story_for_you.llm.base import LLMProvider, LLMResponse


class _FakeLLM(LLMProvider):
    def generate(self, prompt: str, system: str = "", options: dict | None = None) -> LLMResponse:
        return LLMResponse(content="{}", tokens_used=0)

    def generate_stream(self, prompt: str, system: str = "", options: dict | None = None):
        yield from []


def test_single_chunk_style_sample_uses_structural_middle_slice():
    extractor = StyleExtractor(_FakeLLM())
    text = "开头。" * 800 + "中段风格样本。" * 80 + "结尾。" * 800

    samples = extractor._select_samples([text])

    assert len(samples) == 1
    assert "中段风格样本" in samples[0][1]
