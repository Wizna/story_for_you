from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from typing import Any, Iterable, List

from story_for_you.analysis.context import PlotEvent
from story_for_you.core.exceptions import LLMResponseError


class EventLedger:
    """Keeps a chronological ledger of plot-impacting events."""

    def __init__(self) -> None:
        self._events: list[PlotEvent] = []
        self._char_index: dict[str, list[PlotEvent]] = defaultdict(list)
        self._irreversible: list[PlotEvent] = []
        self._event_keys: set[tuple[int | None, int, str]] = set()

    def record(self, events: Iterable[PlotEvent]) -> list[PlotEvent]:
        """Append new events and return the events actually retained.

        Only replaying the same extraction occurrence is idempotent. Similar
        actions in another scene can change state again. Physical source units
        also disambiguate event IDs in chapters split across several chunks.
        """
        recorded: list[PlotEvent] = []
        for event in events:
            key = self._event_key(event)
            if key in self._event_keys:
                continue
            self._event_keys.add(key)
            self._events.append(event)
            recorded.append(event)
            for participant in event.participants:
                if participant:
                    self._char_index[participant].append(event)
            if event.is_irreversible:
                self._irreversible.append(event)
        return recorded

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
        self._event_keys.clear()

    def _event_key(self, event: PlotEvent) -> tuple[int | None, int, str]:
        return event.source_unit, event.chapter, event.event_id

    def to_dict(self) -> dict[str, Any]:
        """Serialize the ledger state to a dictionary."""
        return {
            "events": [asdict(e) for e in self._events],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EventLedger:
        """Restore ledger state from a dictionary."""
        if not isinstance(payload, dict):
            raise LLMResponseError("EventLedger payload must be a JSON object.")
        if "events" not in payload:
            raise LLMResponseError("EventLedger missing required field: events")
        events_payload = payload.get("events")
        if not isinstance(events_payload, list):
            raise LLMResponseError("EventLedger.events must be a list.")
        instance = cls()
        events = [PlotEvent.from_dict(item) for item in events_payload]
        instance.record(events)
        return instance
