from __future__ import annotations

from collections import defaultdict
from typing import Iterable, List

from story_for_you.analysis.context import PlotEvent


class EventLedger:
    """Keeps a chronological ledger of plot-impacting events."""

    def __init__(self) -> None:
        self._events: list[PlotEvent] = []
        self._char_index: dict[str, list[PlotEvent]] = defaultdict(list)
        self._irreversible: list[PlotEvent] = []

    def record(self, events: Iterable[PlotEvent]) -> None:
        """Append events into the ledger."""
        for event in events:
            self._events.append(event)
            for participant in event.participants:
                if participant:
                    self._char_index[participant].append(event)
            if event.is_irreversible:
                self._irreversible.append(event)

    def timeline(self) -> List[PlotEvent]:
        """Return the recorded events ordered by insertion."""
        return list(self._events)

    def find_by_character(self, name: str) -> list[PlotEvent]:
        """Return events that involve the given character."""
        return list(self._char_index.get(name, []))

    def list_irreversible_since(self, chapter: int | None = None) -> list[PlotEvent]:
        """Return irreversible events optionally filtered by chapter number."""
        if chapter is None:
            return list(self._irreversible)
        return [event for event in self._irreversible if event.chapter >= chapter]

    def recent(self, limit: int = 5) -> list[PlotEvent]:
        """Return the most recent plot events."""
        if limit <= 0:
            return []
        return self._events[-limit:]

    def clear(self) -> None:
        """Remove all recorded events."""
        self._events.clear()
        self._char_index.clear()
        self._irreversible.clear()
