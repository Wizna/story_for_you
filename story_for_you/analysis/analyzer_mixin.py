"""Shared methods for story analyzers.

This module contains the common functionality used by both StoryAnalyzer
and ResumableStoryAnalyzer to avoid code duplication.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import uuid

if TYPE_CHECKING:
    from story_for_you.analysis.context import (
        ChapterSummary,
        StoryContext,
        StoryState,
    )
    from story_for_you.analysis.extractors import (
        ChapterSummarizer,
        CharacterExtractor,
        EventExtractor,
        RelationshipMapper,
        StateSynthesizer,
    )
    from story_for_you.analysis.layers import ChapterSummaryWindow, EventLedger, StateStore
    from story_for_you.llm.base import LLMProvider


class AnalyzerMixin:
    """Mixin providing shared analyzer methods.

    Requires the following attributes on the class:
    - llm: LLMProvider
    - prompt_budget: int | None
    - chapter_window: ChapterSummaryWindow
    - event_ledger: EventLedger
    - state_store: StateStore
    - character_extractor: CharacterExtractor
    - relationship_mapper: RelationshipMapper
    - chapter_summarizer: ChapterSummarizer
    - event_extractor: EventExtractor
    - state_synthesizer: StateSynthesizer
    """

    # These attributes are expected to be defined in the concrete class
    llm: LLMProvider
    prompt_budget: int | None
    chapter_window: ChapterSummaryWindow
    event_ledger: EventLedger
    state_store: StateStore
    character_extractor: CharacterExtractor
    relationship_mapper: RelationshipMapper
    chapter_summarizer: ChapterSummarizer
    event_extractor: EventExtractor
    state_synthesizer: StateSynthesizer

    def _build_metadata(self) -> dict[str, Any]:
        """Assemble metadata describing the analysis session."""
        model_name = getattr(self.llm, "model", "unknown")
        return {
            "window_size": self.chapter_window.window_size,
            "model": model_name,
            "_version": "0.1.0",
            "_fingerprint": uuid.uuid4().hex,
        }

    def _build_recent_context(self, chapter_no: int) -> str:
        """Render a lightweight context string for prompt consumption."""
        lines: list[str] = [f"Target chapter: {chapter_no}"]
        story_state = self.state_store.story_snapshot()
        if story_state:
            lines.append(
                f"Arc={story_state.current_arc} | Tension={story_state.world_tension}"
            )
            if story_state.major_conflicts:
                lines.append(
                    "Conflicts: " + "; ".join(story_state.major_conflicts[-3:])
                )
            if story_state.unresolved_events:
                lines.append(
                    "Unresolved: " + "; ".join(story_state.unresolved_events[-3:])
                )
        recent_chapters = self.chapter_window.to_prompt_lines()[-3:]
        if recent_chapters:
            lines.append("Recent chapters:")
            lines.extend(recent_chapters)
        recent_events = self.event_ledger.recent(3)
        if recent_events:
            lines.append("Recent events:")
            lines.extend(
                f"- {event.type}: {event.summary} ({', '.join(event.participants[:3])})"
                + (" [irreversible]" if event.is_irreversible else "")
                for event in recent_events
            )
        return "\n".join(lines).strip()

    def _build_chapter_meta(
        self, chapter_no: int, chapter_text: str, story_state: StoryState | None
    ) -> dict[str, Any]:
        """Prepare chapter metadata payload for summarization prompts."""
        first_line = next(
            (line.strip() for line in chapter_text.splitlines() if line.strip()), ""
        )
        arc_hint = getattr(story_state, "current_arc", None) or "setup"
        return {
            "chapter_no": chapter_no,
            "title_hint": first_line[:40] or f"Chapter {chapter_no}",
            "arc_hint": arc_hint,
        }

    def _process_chapter(
        self, chapter_no: int, chapter_text: str, story_state: StoryState | None
    ) -> tuple[StoryState | None, ChapterSummary]:
        """Run the full extraction pipeline for a single chapter.

        Returns the updated story state and the chapter summary.
        """
        characters = self.character_extractor.extract(chapter_text)
        relationships = self.relationship_mapper.map(chapter_text, characters)
        recent_context = self._build_recent_context(chapter_no)
        chapter_meta = self._build_chapter_meta(chapter_no, chapter_text, story_state)

        summary = self.chapter_summarizer.summarize(
            chapter_text, chapter_no, recent_context, chapter_meta
        )
        events = self.event_extractor.extract(
            chapter_text, characters, chapter_no, recent_context
        )
        for event in events:
            event.chapter = chapter_no

        story_state = self.state_synthesizer.update(story_state, events, recent_context)

        self.chapter_window.append(summary)
        self.event_ledger.record(events)
        self.state_store.update(characters, relationships, events)
        if story_state:
            self.state_store.set_story_state(story_state)

        return story_state, summary

    def _assemble_context(self, chapters: list[str]) -> StoryContext:
        """Extract writing style and assemble the final StoryContext."""
        from story_for_you.analysis.context import StoryContext
        from story_for_you.analysis.extractors import StyleExtractor

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

    def _enrich_metadata(self, context: StoryContext) -> None:
        """Add derived metadata to the context."""
        from story_for_you.analysis.utils import compute_primary_cast

        coverage = self.chapter_window.coverage()
        if coverage:
            context.add_metadata("chapter_coverage", coverage)
        primary_cast = compute_primary_cast(context.characters.values())
        if primary_cast:
            context.add_metadata("primary_cast", primary_cast)
