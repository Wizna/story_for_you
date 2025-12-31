from __future__ import annotations

import itertools
import json
import logging
import re
from typing import Any

from story_for_you.analysis.context import EventImpact, PlotEvent
from story_for_you.analysis.prompting import load_template, render_prompt_with_budget
from story_for_you.llm.base import LLMProvider
from story_for_you.utils.json_utils import load_json_response

logger = logging.getLogger(__name__)


class EventExtractor:
    """LLM-backed event extractor using structured prompts."""

    def __init__(self, llm: LLMProvider, prompt_budget: int | None = None):
        self.llm = llm
        self._counter = itertools.count(1)
        self.template = load_template("event_extraction")
        self.prompt_budget = prompt_budget

    def extract(
        self,
        chapter_text: str,
        participants: list[str],
        chapter_no: int,
        recent_context: str,
    ) -> list[PlotEvent]:
        """Extract lasting events from the chapter text."""
        roster = [
            {"name": name, "aliases": []}
            for name in sorted({participant for participant in participants if participant})
        ]
        recent_context_text = recent_context.strip() or "暂无历史上下文。"
        chapter_body = chapter_text.strip()
        prompt, truncated = render_prompt_with_budget(
            self.template,
            budget=self.prompt_budget,
            text_key="chapter_text",
            text_value=chapter_body,
            chapter_no=str(chapter_no),
            character_roster=json.dumps(roster, ensure_ascii=False),
            recent_context=recent_context_text,
        )
        if truncated:
            logger.debug("Event extraction prompt truncated to %s chars", len(prompt))
        response = self.llm.generate(prompt=prompt)
        try:
            payload = load_json_response(response.content)
            if isinstance(payload, list):
                return [self._from_payload(item, chapter_no) for item in payload]
            raise ValueError("Event extractor response is not a list.")
        except (ValueError, TypeError) as exc:
            logger.warning("Failed to parse event extraction response: %s", exc)
            return self._fallback_extract(chapter_text, participants, chapter_no)

    def _from_payload(self, payload: Any, chapter_no: int) -> PlotEvent:
        event_id = payload.get("event_id") or f"CH{chapter_no:03d}-E{next(self._counter):02d}"
        impact_payload = payload.get("impact") or {}
        impact = EventImpact(
            power_shifts=dict(impact_payload.get("power_shifts") or {}),
            relation_changes=dict(impact_payload.get("relation_changes") or {}),
            world_flags=list(impact_payload.get("world_flags") or []),
        )
        participants = [str(name) for name in payload.get("participants", [])]
        summary = str(payload.get("summary", "")).strip()
        return PlotEvent(
            event_id=event_id,
            chapter=int(payload.get("chapter", chapter_no)),
            type=str(payload.get("type", "progress")),
            participants=participants,
            summary=summary,
            impact=impact,
            is_irreversible=bool(payload.get("is_irreversible", False)),
        )

    # Fallback heuristics ----------------------------------------------------------
    def _fallback_extract(self, chapter_text: str, participants: list[str], chapter_no: int) -> list[PlotEvent]:
        events: list[PlotEvent] = []
        sentences = re.split(r"(?<=[。.!?])\s+", chapter_text.strip())
        for sentence in sentences:
            event_type = self._classify(sentence)
            if not event_type:
                continue
            involved = [name for name in participants if name in sentence]
            if not involved:
                continue
            event_id = f"CH{chapter_no:03d}-F{next(self._counter):02d}"
            impact = self._build_impact(involved, event_type)
            irreversible = any(keyword in sentence.lower() for keyword in ["death", "destroy", "牺牲"])
            events.append(
                PlotEvent(
                    event_id=event_id,
                    chapter=chapter_no,
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
