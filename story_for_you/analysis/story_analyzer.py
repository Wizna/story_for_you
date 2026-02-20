from __future__ import annotations

from typing import Iterable

from story_for_you.analysis.analyzer_mixin import AnalyzerMixin
from story_for_you.analysis.context import StoryContext
from story_for_you.analysis.extractors import (
    ChapterSummarizer,
    CharacterExtractor,
    EventExtractor,
    RelationshipMapper,
    StateSynthesizer,
)
from story_for_you.analysis.layers import ChapterSummaryWindow, EventLedger, StateStore
from story_for_you.llm.base import LLMProvider


class StoryAnalyzer(AnalyzerMixin):
    """Coordinates extractor components to build a StoryContext."""

    def __init__(
        self, llm: LLMProvider, window_size: int = 12, prompt_budget: int | None = None
    ):
        self.llm = llm
        self.prompt_budget = prompt_budget
        self.chapter_window = ChapterSummaryWindow(window_size)
        self.event_ledger = EventLedger()
        self.state_store = StateStore()
        self.character_extractor = CharacterExtractor(llm, prompt_budget=prompt_budget)
        self.relationship_mapper = RelationshipMapper(llm)
        self.chapter_summarizer = ChapterSummarizer(llm, prompt_budget=prompt_budget)
        self.event_extractor = EventExtractor(llm, prompt_budget=prompt_budget)
        self.state_synthesizer = StateSynthesizer(llm)

    def analyze(self, chapters: Iterable[str]) -> StoryContext:
        """Run the analyzer across a list of chapter-sized texts."""
        chapters = list(chapters)
        self.chapter_window.clear()
        self.event_ledger.clear()
        self.state_store.clear()
        story_state = None
        for chapter_no, chapter_text in enumerate(chapters, start=1):
            story_state, _summary = self._process_chapter(
                chapter_no, chapter_text, story_state
            )

        return self._assemble_context(chapters)
