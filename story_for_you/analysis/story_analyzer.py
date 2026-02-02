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
    StyleExtractor,
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
            characters = self.character_extractor.extract(chapter_text)
            character_names = [character.name for character in characters]
            relationships = self.relationship_mapper.map(chapter_text, character_names)
            recent_context = self._build_recent_context(chapter_no)
            chapter_meta = self._build_chapter_meta(
                chapter_no, chapter_text, story_state
            )
            summary = self.chapter_summarizer.summarize(
                chapter_text,
                chapter_no,
                recent_context,
                chapter_meta,
            )
            events = self.event_extractor.extract(
                chapter_text,
                character_names,
                chapter_no,
                recent_context,
            )
            for event in events:
                event.chapter = chapter_no
            story_state = self.state_synthesizer.update(
                story_state, events, recent_context
            )
            self.chapter_window.append(summary)
            self.event_ledger.record(events)
            self.state_store.update(characters, relationships, events)
            if story_state:
                self.state_store.set_story_state(story_state)

        style_extractor = StyleExtractor(self.llm, prompt_budget=self.prompt_budget)
        summaries = self.chapter_window.dump()
        writing_style = style_extractor.extract(chapters, summaries)

        context = StoryContext(
            metadata=self._build_metadata(),
            chapter_window=summaries,
            events=self.event_ledger.timeline(),
            characters=self.state_store.characters_snapshot(),
            story_state=self.state_store.story_snapshot(),
            writing_style=writing_style,
        )
        self._enrich_metadata(context)
        return context
