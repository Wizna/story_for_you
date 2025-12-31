from __future__ import annotations

import itertools
import re

from story_for_you.analysis.context import EventImpact, PlotEvent
from story_for_you.llm.base import LLMProvider


class EventExtractor:
    """LLM-backed event extractor placeholder."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm
        self._counter = itertools.count(1)

    def extract(self, chapter_text: str, participants: list[str]) -> list[PlotEvent]:
        """Extract lasting events from the chapter text."""
        events: list[PlotEvent] = []
        sentences = re.split(r"(?<=[。.!?])\s+", chapter_text.strip())
        for sentence in sentences:
            event_type = self._classify(sentence)
            if not event_type:
                continue
            involved = [name for name in participants if name in sentence]
            if not involved:
                continue
            event_id = f"evt-{next(self._counter):05d}"
            impact = self._build_impact(involved, event_type)
            irreversible = any(keyword in sentence.lower() for keyword in ["death", "destroy", "牺牲"])
            events.append(
                PlotEvent(
                    event_id=event_id,
                    chapter=0,
                    type=event_type,
                    participants=involved,
                    summary=sentence.strip(),
                    impact=impact,
                    is_irreversible=irreversible,
                )
            )
        return events

    def _classify(self, sentence: str) -> str | None:
        lowered = sentence.lower()
        if any(keyword in lowered for keyword in ["battle", "fight", "袭击"]):
            return "conflict"
        if any(keyword in lowered for keyword in ["revealed", "秘密", "发现"]):
            return "reveal"
        if any(keyword in lowered for keyword in ["plan", "journey", "联盟"]):
            return "progress"
        if any(keyword in lowered for keyword in ["fail", "失去", "阻止"]):
            return "setback"
        return None

    def _build_impact(self, participants: list[str], event_type: str) -> EventImpact:
        power_shifts = {name: event_type for name in participants[:2]}
        relation_changes = {name: event_type for name in participants}
        world_flags = [event_type]
        return EventImpact(
            power_shifts=power_shifts,
            relation_changes=relation_changes,
            world_flags=world_flags,
        )
