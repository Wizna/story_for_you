"""Shared methods for story analyzers.

This module contains the common functionality used by both StoryAnalyzer
and ResumableStoryAnalyzer to avoid code duplication.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import uuid

if TYPE_CHECKING:
    from story_for_you.analysis.context import StoryContext, StoryState
    from story_for_you.analysis.layers import ChapterSummaryWindow, EventLedger, StateStore


class AnalyzerMixin:
    """Mixin providing shared analyzer methods.

    Requires the following attributes on the class:
    - llm: LLMProvider
    - chapter_window: ChapterSummaryWindow
    - event_ledger: EventLedger
    - state_store: StateStore
    """

    # These attributes are expected to be defined in the concrete class
    chapter_window: ChapterSummaryWindow
    event_ledger: EventLedger
    state_store: StateStore

    def _build_metadata(self) -> dict[str, Any]:
        """Assemble metadata describing the analysis session."""
        model_name = getattr(self.llm, "model", "unknown")  # type: ignore[attr-defined]
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

    def _enrich_metadata(self, context: StoryContext) -> None:
        """Add derived metadata to the context."""
        from story_for_you.analysis.utils import compute_primary_cast

        coverage = self.chapter_window.coverage()
        if coverage:
            context.add_metadata("chapter_coverage", coverage)
        primary_cast = compute_primary_cast(context.characters.values())
        if primary_cast:
            context.add_metadata("primary_cast", primary_cast)
