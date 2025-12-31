from typing import Any, Iterable

from story_for_you.analysis.context import StoryContext
from story_for_you.analysis.layers import ChapterSummaryWindow, EventLedger, StateStore
from story_for_you.analysis.extractors import (
    ChapterSummarizer,
    CharacterExtractor,
    EventExtractor,
    RelationshipMapper,
    StateSynthesizer,
)
from story_for_you.llm.base import LLMProvider


class StoryAnalyzer:
    """Coordinates extractor components to build a StoryContext."""

    def __init__(self, llm: LLMProvider, window_size: int = 12):
        self.llm = llm
        self.chapter_window = ChapterSummaryWindow(window_size)
        self.event_ledger = EventLedger()
        self.state_store = StateStore()
        self.character_extractor = CharacterExtractor(llm)
        self.relationship_mapper = RelationshipMapper(llm)
        self.chapter_summarizer = ChapterSummarizer(llm)
        self.event_extractor = EventExtractor(llm)
        self.state_synthesizer = StateSynthesizer(llm)

    def analyze(self, chapters: Iterable[str]) -> StoryContext:
        """Run the analyzer across a list of chapter-sized texts."""
        raise NotImplementedError("Story analysis pipeline not implemented yet.")

    def _build_metadata(self) -> dict[str, Any]:
        """Assemble metadata describing the analysis session."""
        return {"window_size": len(self.chapter_window.dump())}
