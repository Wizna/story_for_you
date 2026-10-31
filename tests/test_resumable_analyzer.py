from __future__ import annotations

import pytest

from story_for_you.analysis.context import ChapterSummary, StoryContext
from story_for_you.analysis.resumable_analyzer import ResumableStoryAnalyzer
from story_for_you.cache.progress_store import AnalysisProgress, ProgressStore


class _OfflineAnalyzer(ResumableStoryAnalyzer):
    def __init__(
        self, progress_store: ProgressStore, *, fail_unit: str = "", fail_assembly: bool = False
    ) -> None:
        super().__init__(llm=None, progress_store=progress_store)
        self.fail_unit = fail_unit
        self.fail_assembly = fail_assembly
        self.processed: list[str] = []

    def _process_chapter(self, chapter_no, chapter_text, story_state, *, source_unit=None):
        if chapter_text == self.fail_unit:
            raise RuntimeError("interrupted")
        self.processed.append(chapter_text)
        summary = ChapterSummary(
            chapter=chapter_no,
            title=chapter_text,
            pov="unknown",
            beats=[],
            mood="unknown",
            synopsis=chapter_text,
        )
        self.chapter_window.append(summary)
        return story_state, summary

    def _assemble_context(self, chapters):
        if self.fail_assembly:
            raise RuntimeError("style extraction failed")
        return StoryContext(chapter_window=self.chapter_window.dump())


def test_resume_uses_unit_count_when_source_chapter_spans_multiple_units(tmp_path):
    store = ProgressStore(tmp_path)
    first = _OfflineAnalyzer(store, fail_unit="C")
    with pytest.raises(RuntimeError, match="interrupted"):
        first.analyze(["A", "B", "C"], "book", chapter_labels=[1, 1, 2])

    assert store.get_progress("book").completed_units == 2
    second = _OfflineAnalyzer(store)
    result = second.analyze(["A", "B", "C"], "book", chapter_labels=[1, 1, 2])

    assert second.processed == ["C"]
    assert [summary.title for summary in result.chapter_window] == ["A", "B", "C"]
    assert store.get_progress("book") is None


def test_final_style_failure_keeps_completed_analysis_checkpoint(tmp_path):
    store = ProgressStore(tmp_path)
    with pytest.raises(RuntimeError, match="style extraction failed"):
        _OfflineAnalyzer(store, fail_assembly=True).analyze(["A"], "book")

    assert store.get_progress("book").completed_units == 1
    resumed = _OfflineAnalyzer(store)
    result = resumed.analyze(["A"], "book")

    assert resumed.processed == []
    assert [summary.title for summary in result.chapter_window] == ["A"]


def test_checkpoint_rejects_missing_or_incomplete_analysis_state():
    with pytest.raises(ValueError, match="Invalid analysis progress state"):
        AnalysisProgress.from_dict({
            "analysis_key": "book", "total_units": 2, "completed_units": 1,
            "memory_state": {"chapter_window": {"history": []}},
        })
