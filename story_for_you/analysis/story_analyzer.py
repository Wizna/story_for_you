from __future__ import annotations

from typing import Any, Iterable

import uuid

from story_for_you.analysis.context import StoryContext, StoryState
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
            recent_context = self._build_recent_context(chapter_no)
            chapter_meta = self._build_chapter_meta(chapter_no, chapter_text, story_state)
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
            story_state = self.state_synthesizer.update(story_state, events, recent_context)
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

    def _build_recent_context(self, chapter_no: int) -> str:
        """Render a lightweight context string for prompt consumption."""
        lines: list[str] = [f"Target chapter: {chapter_no}"]
        story_state = self.state_store.story_snapshot()
        if story_state:
            lines.append(f"Arc={story_state.current_arc} | Tension={story_state.world_tension}")
            if story_state.major_conflicts:
                lines.append("Conflicts: " + "; ".join(story_state.major_conflicts[-3:]))
            if story_state.unresolved_events:
                lines.append("Unresolved: " + "; ".join(story_state.unresolved_events[-3:]))
        recent_chapters = self.chapter_window.to_prompt_lines()[-3:]
        if recent_chapters:
            lines.append("Recent chapters:")
            lines.extend(recent_chapters)
        recent_events = self.event_ledger.timeline()[-3:]
        if recent_events:
            lines.append("Recent events:")
            lines.extend(
                f"- {event.type}: {event.summary} ({', '.join(event.participants[:3])})".strip()
                for event in recent_events
            )
        return "\n".join(lines).strip()

    def _build_chapter_meta(self, chapter_no: int, chapter_text: str, story_state: StoryState | None) -> dict[str, Any]:
        """Prepare chapter metadata payload for summarization prompts."""
        first_line = next((line.strip() for line in chapter_text.splitlines() if line.strip()), "")
        arc_hint = getattr(story_state, "current_arc", None) or "setup"
        return {
            "chapter_no": chapter_no,
            "title_hint": first_line[:40] or f"Chapter {chapter_no}",
            "arc_hint": arc_hint,
        }
