from __future__ import annotations

from story_for_you.analysis.extractors.chapters import ChapterSummarizer
from story_for_you.analysis.extractors.characters import CharacterExtractor
from story_for_you.analysis.extractors.events import EventExtractor
from story_for_you.analysis.extractors.relationships import RelationshipMapper
from story_for_you.llm.base import LLMProvider, LLMResponse
from story_for_you.utils.prompting import CacheablePrompt


class _RecordingLLM(LLMProvider):
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.prompts: list[CacheablePrompt] = []

    def generate(
        self,
        prompt: CacheablePrompt,
        system: str = "",
        options: dict | None = None,
    ) -> LLMResponse:
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("No more queued responses.")
        return LLMResponse(content=self.responses.pop(0), tokens_used=0)

    def generate_stream(
        self,
        prompt: CacheablePrompt,
        system: str = "",
        options: dict | None = None,
    ):
        yield from []


def test_chapter_analysis_extractors_share_the_same_cache_prefix():
    chapter_text = "翠翠在渡口等傩送。"
    llm = _RecordingLLM(
        [
            '[{"name":"翠翠","aliases":[],"role":"main","realm":null,"personality":[],"unresolved":[]}]',
            "[]",
            (
                '{"chapter":1,"title":"渡口","pov":"third-person","beats":["等待"],'
                '"mood":"somber","synopsis":"翠翠在渡口等待傩送。","irreversible_flags":[]}'
            ),
            "[]",
        ]
    )

    characters = CharacterExtractor(llm).extract(chapter_text)
    RelationshipMapper(llm).map(chapter_text, characters)
    ChapterSummarizer(llm).summarize(chapter_text, 1, "")
    EventExtractor(llm).extract(chapter_text, characters, 1, "")

    assert len(llm.prompts) == 4
    assert {prompt.prefix for prompt in llm.prompts} == {chapter_text}
    assert all(chapter_text not in prompt.task for prompt in llm.prompts)
