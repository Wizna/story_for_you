from collections import deque
from dataclasses import asdict
from typing import Any, Deque, Iterable, List

from story_for_you.analysis.context import ChapterSummary


class ChapterSummaryWindow:
    """Maintains a bounded window of recent chapter summaries."""

    def __init__(self, window_size: int = 12):
        self.window_size = window_size
        self._window: Deque[ChapterSummary] = deque(maxlen=window_size)

    def append(self, summary: ChapterSummary) -> None:
        """Add a summary to the rolling window."""
        self._window.append(summary)

    def dump(self) -> List[ChapterSummary]:
        """Return a list copy of the stored summaries."""
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

    def to_dict(self) -> dict[str, Any]:
        """Serialize the window state to a dictionary."""
        return {
            "window_size": self.window_size,
            "summaries": [asdict(s) for s in self._window],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChapterSummaryWindow":
        """Restore window state from a dictionary."""
        window_size = payload.get("window_size", 12)
        instance = cls(window_size=window_size)
        for item in payload.get("summaries", []):
            instance.append(ChapterSummary(**item))
        return instance
