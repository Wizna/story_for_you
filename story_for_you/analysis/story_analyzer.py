from __future__ import annotations

from typing import Any, Iterable

import uuid

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
        self.chapter_window.clear()
        self.event_ledger.clear()
        self.state_store.clear()
        story_state = None
        for chapter_no, chapter_text in enumerate(chapters, start=1):
            characters = self.character_extractor.extract(chapter_text)
            character_names = [character.name for character in characters]
            relationships = self.relationship_mapper.map(chapter_text, character_names)
            summary = self.chapter_summarizer.summarize(chapter_text, chapter_no)
            events = self.event_extractor.extract(chapter_text, character_names)
            for event in events:
                event.chapter = chapter_no
            story_state = self.state_synthesizer.update(story_state, events)
            self.chapter_window.append(summary)
            self.event_ledger.record(events)
            self.state_store.update(characters, relationships, events)
            if story_state:
                self.state_store.set_story_state(story_state)
        return StoryContext(
            metadata=self._build_metadata(),
            chapter_window=self.chapter_window.dump(),
            events=self.event_ledger.timeline(),
            characters=self.state_store.characters_snapshot(),
            story_state=self.state_store.story_snapshot(),
        )

    def _build_metadata(self) -> dict[str, Any]:
        """Assemble metadata describing the analysis session."""
        model_name = getattr(self.llm, "model", "unknown")
        return {
            "window_size": self.chapter_window.window_size,
            "model": model_name,
            "_version": "0.1.0",
            "_fingerprint": uuid.uuid4().hex,
        }
