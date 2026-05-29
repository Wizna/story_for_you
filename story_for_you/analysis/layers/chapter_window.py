from collections import deque
from dataclasses import asdict
from typing import Any, Deque, Iterable, List, Sequence

from story_for_you.analysis.context import ChapterSummary
from story_for_you.core.exceptions import LLMResponseError


class ChapterSummaryWindow:
    """Maintains a bounded prompt window plus a full chapter history."""

    def __init__(self, window_size: int = 12):
        self.window_size = window_size
        self._window: Deque[ChapterSummary] = deque(maxlen=window_size)
        self._history: list[ChapterSummary] = []

    def append(self, summary: ChapterSummary) -> None:
        """Add a summary to the rolling window."""
        self._history.append(summary)
        self._window.append(summary)

    def dump(self) -> List[ChapterSummary]:
        """Return a sorted copy of every chapter summary seen so far."""
        return sorted(self._history, key=lambda item: item.chapter)

    def recent(self) -> List[ChapterSummary]:
        """Expose only the bounded window (mostly for tests)."""
        return list(self._window)

    def to_prompt_lines(self) -> list[str]:
        """Render the window into concise prompt-friendly lines."""
        return [f"Chapter {item.chapter}: {item.synopsis}" for item in self._window]

    def extend(self, summaries: Iterable[ChapterSummary]) -> None:
        """Bulk extend the window with prepared summaries."""
        for summary in summaries:
            self.append(summary)

    def clear(self) -> None:
        """Reset the rolling window."""
        self._window.clear()
        self._history.clear()

    def coverage(self) -> dict[str, int] | None:
        """Return chapter coverage stats for metadata."""
        if not self._history:
            return None
        chapters = [summary.chapter for summary in self._history]
        return {
            "start": min(chapters),
            "end": max(chapters),
            "count": len(self._history),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize the window state to a dictionary."""
        return {
            "window_size": self.window_size,
            "window": [asdict(s) for s in self._window],
            "history": [asdict(s) for s in self._history],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChapterSummaryWindow":
        """Restore window state from a dictionary."""
        if not isinstance(payload, dict):
            raise LLMResponseError("ChapterSummaryWindow payload must be a JSON object.")
        for field_name in ("window_size", "history", "window"):
            if field_name not in payload:
                raise LLMResponseError(f"ChapterSummaryWindow missing required field: {field_name}")
        window_size = payload.get("window_size")
        if not isinstance(window_size, int) or isinstance(window_size, bool):
            raise LLMResponseError("ChapterSummaryWindow.window_size must be an integer.")
        instance = cls(window_size=window_size)
        history_payload: Sequence[dict[str, Any]] = payload.get("history")
        if not isinstance(history_payload, list):
            raise LLMResponseError("ChapterSummaryWindow.history must be a list.")
        for item in history_payload:
            instance._history.append(ChapterSummary.from_dict(item))
        window_payload: Sequence[dict[str, Any]] = payload.get("window")
        if not isinstance(window_payload, list):
            raise LLMResponseError("ChapterSummaryWindow.window must be a list.")
        for item in window_payload:
            instance._window.append(ChapterSummary.from_dict(item))
        return instance
