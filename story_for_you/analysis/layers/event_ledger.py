from typing import Iterable, List

from story_for_you.analysis.context import PlotEvent


class EventLedger:
    """Keeps a chronological ledger of plot-impacting events."""

    def __init__(self) -> None:
        self._events: list[PlotEvent] = []

    def record(self, events: Iterable[PlotEvent]) -> None:
        """Append events into the ledger."""
        for event in events:
            self._events.append(event)

    def timeline(self) -> List[PlotEvent]:
        """Return the recorded events ordered by insertion."""
        return list(self._events)

    def find_by_character(self, name: str) -> list[PlotEvent]:
        """Return events that involve the given character."""
        return [event for event in self._events if name in event.participants]

    def clear(self) -> None:
        """Remove all recorded events."""
        self._events.clear()
