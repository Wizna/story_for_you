from __future__ import annotations

import itertools
import json
import logging
from typing import Any

from story_for_you.analysis.context import EventImpact, PlotEvent
from story_for_you.analysis.prompting import (
    clamp_text_middle,
    fill_template,
    load_template,
    render_prompt_with_budget,
)
from story_for_you.core.exceptions import LLMResponseError
from story_for_you.llm.base import LLMProvider
from story_for_you.utils.json_utils import load_json_response

logger = logging.getLogger(__name__)

_REPAIR_SNIPPET_BUDGET = 4000


class EventExtractor:
    """LLM-backed event extractor using structured prompts."""

    def __init__(self, llm: LLMProvider, prompt_budget: int | None = None):
        self.llm = llm
        self._counter = itertools.count(1)
        self.template = load_template("event_extraction")
        self.repair_template = load_template("event_extraction_repair")
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
        response = self.llm.generate(prompt=prompt, options={"no_think": True})
        events, error = self._parse_response(response.content, chapter_no)
        if events is not None:
            return events

        if error:
            logger.debug("Event extraction parse failed (%s). Attempting repair.", error)
        repaired_content = self._attempt_repair(response.content, error)
        if repaired_content:
            events, repair_error = self._parse_response(repaired_content, chapter_no)
            if events is not None:
                return events
            error = repair_error or error

        raw_content = response.content[:1000] if response.content else "(empty)"
        logger.warning(
            "Failed to parse event extraction response after repair: %s\nRaw response (truncated): %s",
            error or "Unknown parse failure",
            raw_content,
        )
        raise LLMResponseError(f"Event extraction failed after repair: {error or 'unknown parse failure'}")

    def _from_payload(self, payload: Any, chapter_no: int) -> PlotEvent:
        for field_name in (
            "chapter",
            "type",
            "participants",
            "summary",
            "impact",
            "is_irreversible",
        ):
            if field_name not in payload:
                raise LLMResponseError(f"Event item missing required field: {field_name}")
        event_id = payload.get("event_id") or f"CH{chapter_no:03d}-E{next(self._counter):02d}"
        impact_payload = payload.get("impact") or {}
        if not isinstance(impact_payload, dict):
            raise LLMResponseError("Event impact must be an object.")
        for field_name in ("power_shifts", "relation_changes", "world_flags"):
            if field_name not in impact_payload:
                raise LLMResponseError(f"Event impact missing required field: {field_name}")
        event_type = str(payload.get("type", "")).strip()
        if event_type not in {"conflict", "reveal", "progress", "setback"}:
            raise LLMResponseError(f"Invalid event type: {event_type!r}")
        summary = str(payload.get("summary", "")).strip()
        if not summary:
            raise LLMResponseError("Event summary is required.")
        impact = EventImpact(
            power_shifts=dict(impact_payload.get("power_shifts") or {}),
            relation_changes=dict(impact_payload.get("relation_changes") or {}),
            world_flags=list(impact_payload.get("world_flags") or []),
        )
        participants = [str(name) for name in payload.get("participants", [])]
        return PlotEvent(
            event_id=event_id,
            chapter=int(payload.get("chapter", chapter_no)),
            type=event_type,
            participants=participants,
            summary=summary,
            impact=impact,
            is_irreversible=bool(payload.get("is_irreversible", False)),
        )

    def _parse_response(self, content: str | None, chapter_no: int) -> tuple[list[PlotEvent] | None, str | None]:
        """Attempt to parse structured events from the model output."""
        if not content:
            return None, "Empty response body."
        payload = load_json_response(content)
        if payload is None:
            return None, "Response body was not valid JSON."
        events_payload = self._coerce_event_payload(payload)
        if events_payload is None:
            payload_repr = repr(payload)[:500]
            return None, (
                "Event extractor response is not a list. "
                f"Got type={type(payload).__name__}, value={payload_repr}"
            )
        events: list[PlotEvent] = []
        for idx, item in enumerate(events_payload, start=1):
            if not isinstance(item, dict):
                return None, f"Event payload at index {idx} is not an object."
            try:
                events.append(self._from_payload(item, chapter_no))
            except LLMResponseError as exc:
                return None, str(exc)
        return events, None

    def _coerce_event_payload(self, payload: Any) -> list[Any] | None:
        """Normalize various payload shapes into a list of events."""
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("events", "data", "result"):
                candidate = payload.get(key)
                if isinstance(candidate, list):
                    return candidate
        return None

    def _attempt_repair(self, raw_response: str | None, error: str | None) -> str | None:
        """Ask the LLM to fix its malformed JSON response."""
        if not raw_response:
            return None
        snippet = clamp_text_middle(raw_response, _REPAIR_SNIPPET_BUDGET)
        prompt = fill_template(
            self.repair_template,
            error_message=error or "JSON parsing failed.",
            invalid_output=snippet,
        )
        repaired = self.llm.generate(prompt=prompt, options={"no_think": True})
        return repaired.content
