from __future__ import annotations

from typing import Callable

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
from story_for_you.cache.progress_store import AnalysisProgress, ProgressStore
from story_for_you.llm.base import LLMProvider


class ResumableStoryAnalyzer(AnalyzerMixin):
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
        analysis_key: str,
        progress_callback: Callable[[int, int], None] | None = None,
        chapter_labels: list[int | None] | None = None,
    ) -> StoryContext:
        """Run analysis with resume support."""
        total_chapters = len(chapters)
        labels = chapter_labels or []
        if labels and len(labels) != total_chapters:
            raise ValueError("chapter_labels must have the same length as chapters")
        progress = self.progress_store.get_progress(analysis_key)

        if progress and progress.total_units == total_chapters:
            start_from = progress.completed_units
            self._restore_state(progress)
        else:
            start_from = 0
            self._reset_state()
            if progress:
                self.progress_store.clear_progress(analysis_key)

        story_state = self.state_store.story_snapshot()

        for i in range(start_from, total_chapters):
            chapter_no = (labels[i] if labels else None) or i + 1
            chapter_text = chapters[i]

            story_state, _ = self._process_chapter(
                chapter_no, chapter_text, story_state, source_unit=i + 1
            )

            self._save_progress(analysis_key, total_chapters, i + 1)

            if progress_callback:
                progress_callback(i + 1, total_chapters)

        context = self._assemble_context(chapters)
        self.progress_store.clear_progress(analysis_key)
        return context

    def _reset_state(self) -> None:
        """Reset all analysis state."""
        self.chapter_window.clear()
        self.event_ledger.clear()
        self.state_store.clear()

    def _restore_state(self, progress: AnalysisProgress) -> None:
        """Restore state from saved progress."""
        memory = progress.memory_state
        self.chapter_window = ChapterSummaryWindow.from_dict(
            memory.get("chapter_window", {})
        )
        self.event_ledger = EventLedger.from_dict(memory.get("event_ledger", {}))
        self.state_store = StateStore.from_dict(memory.get("state_store", {}))

    def _save_progress(
        self,
        analysis_key: str,
        total_units: int,
        completed_units: int,
    ) -> None:
        """Save current progress to disk."""
        progress = AnalysisProgress(
            analysis_key=analysis_key,
            total_units=total_units,
            completed_units=completed_units,
            memory_state={
                "chapter_window": self.chapter_window.to_dict(),
                "event_ledger": self.event_ledger.to_dict(),
                "state_store": self.state_store.to_dict(),
            },
        )
        self.progress_store.save_progress(progress)
