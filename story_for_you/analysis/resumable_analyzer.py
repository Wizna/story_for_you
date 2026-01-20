from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import uuid

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
    StyleExtractor,
)
from story_for_you.analysis.layers import ChapterSummaryWindow, EventLedger, StateStore
from story_for_you.analysis.utils import compute_primary_cast
from story_for_you.cache.progress_store import AnalysisProgress, ProgressStore
from story_for_you.llm.base import LLMProvider


class ResumableStoryAnalyzer:
    """Analyzer that can resume from saved progress."""

    def __init__(
        self,
        llm: LLMProvider,
        progress_store: ProgressStore,
        window_size: int = 12,
        prompt_budget: int | None = None,
    ):
        self.llm = llm
        self.progress_store = progress_store
        self.window_size = window_size
        self.prompt_budget = prompt_budget
        self._init_components()

    def _init_components(self) -> None:
        """Initialize or reinitialize analysis components."""
        self.chapter_window = ChapterSummaryWindow(self.window_size)
        self.event_ledger = EventLedger()
        self.state_store = StateStore()
        self.character_extractor = CharacterExtractor(
            self.llm, prompt_budget=self.prompt_budget
        )
        self.relationship_mapper = RelationshipMapper(self.llm)
        self.chapter_summarizer = ChapterSummarizer(
            self.llm, prompt_budget=self.prompt_budget
        )
        self.event_extractor = EventExtractor(
            self.llm, prompt_budget=self.prompt_budget
        )
        self.state_synthesizer = StateSynthesizer(self.llm)

    def analyze(
        self,
        chapters: list[str],
        file_hash: str,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> StoryContext:
        """Run analysis with resume support."""
        total_chapters = len(chapters)
        progress = self.progress_store.get_progress(file_hash)

        if progress and progress.total_chapters == total_chapters:
            start_from = progress.completed_chapters
            self._restore_state(progress)
        else:
            start_from = 0
            self._reset_state()
            if progress:
                self.progress_store.clear_progress(file_hash)

        story_state = self.state_store.story_snapshot()

        for i in range(start_from, total_chapters):
            chapter_no = i + 1
            chapter_text = chapters[i]

            characters = self.character_extractor.extract(chapter_text)
            character_names = [c.name for c in characters]
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

            self._save_progress(file_hash, total_chapters, chapter_no, summary)

            if progress_callback:
                progress_callback(chapter_no, total_chapters)

        self.progress_store.clear_progress(file_hash)

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

    def _reset_state(self) -> None:
        """Reset all analysis state."""
        self.chapter_window.clear()
        self.event_ledger.clear()
        self.state_store.clear()
        self._chapter_results: list[dict[str, Any]] = []

    def _restore_state(self, progress: AnalysisProgress) -> None:
        """Restore state from saved progress."""
        memory = progress.memory_state
        self.chapter_window = ChapterSummaryWindow.from_dict(
            memory.get("chapter_window", {})
        )
        self.event_ledger = EventLedger.from_dict(memory.get("event_ledger", {}))
        self.state_store = StateStore.from_dict(memory.get("state_store", {}))
        self._chapter_results = list(progress.chapter_results)

    def _save_progress(
        self,
        file_hash: str,
        total_chapters: int,
        completed_chapters: int,
        latest_summary: ChapterSummary,
    ) -> None:
        """Save current progress to disk."""
        self._chapter_results.append(asdict(latest_summary))
        progress = AnalysisProgress(
            file_hash=file_hash,
            total_chapters=total_chapters,
            completed_chapters=completed_chapters,
            chapter_results=self._chapter_results,
            memory_state={
                "chapter_window": self.chapter_window.to_dict(),
                "event_ledger": self.event_ledger.to_dict(),
                "state_store": self.state_store.to_dict(),
            },
        )
        self.progress_store.save_progress(progress)

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

    def _enrich_metadata(self, context: StoryContext) -> None:
        coverage = self.chapter_window.coverage()
        if coverage:
            context.add_metadata("chapter_coverage", coverage)
        primary_cast = compute_primary_cast(context.characters.values())
        if primary_cast:
            context.add_metadata("primary_cast", primary_cast)
