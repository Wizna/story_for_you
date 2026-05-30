from __future__ import annotations

import json
import logging
from typing import Any

from story_for_you.analysis.context import CharacterState, EventImpact, PlotEvent
from story_for_you.analysis.prompting import (
    clamp_text_middle,
    fill_template,
    load_template,
    render_prompt_with_budget,
)
from story_for_you.core.exceptions import LLMResponseError
from story_for_you.llm.base import LLMProvider
from story_for_you.llm.telemetry import telemetry_options
from story_for_you.utils.json_utils import load_json_response

logger = logging.getLogger(__name__)

_REPAIR_SNIPPET_BUDGET = 4000
_STRUCTURED_OPTIONS = {"no_think": True, "temperature": 0.1}


class EventExtractor:
    """LLM-backed event extractor using structured prompts."""

    def __init__(self, llm: LLMProvider, prompt_budget: int | None = None):
        self.llm = llm
        self.template = load_template("event_extraction")
        self.repair_template = load_template("event_extraction_repair")
        self.prompt_budget = prompt_budget

    def extract(
        self,
        chapter_text: str,
        participants: list[str] | list[CharacterState],
        chapter_no: int,
        recent_context: str,
    ) -> list[PlotEvent]:
        """Extract lasting events from the chapter text."""
        roster, alias_map = self._build_roster(participants)
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
        response = self.llm.generate(
            prompt=prompt,
            options=telemetry_options(
                _STRUCTURED_OPTIONS,
                phase=f"analyze chapter {chapter_no}",
                step=": extract plot events",
            ),
        )
        allowed_participants = set(alias_map)
        events, error = self._parse_response(response.content, chapter_no, alias_map, allowed_participants)
        if events is not None:
            return events

        if error:
            logger.debug("Event extraction parse failed (%s). Attempting repair.", error)
        repaired_content = self._attempt_repair(response.content, error, chapter_no)
        if repaired_content:
            events, repair_error = self._parse_response(
                repaired_content,
                chapter_no,
                alias_map,
                allowed_participants,
            )
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

    def _from_payload(
        self,
        payload: Any,
        chapter_no: int,
        alias_map: dict[str, str],
        allowed_participants: set[str],
    ) -> PlotEvent:
        for field_name in (
            "event_id",
            "chapter",
            "type",
            "participants",
            "summary",
            "impact",
            "is_irreversible",
        ):
            if field_name not in payload:
                raise LLMResponseError(f"Event item missing required field: {field_name}")
        event_id = self._required_str(payload.get("event_id"), "event_id")
        impact_payload = payload.get("impact")
        if not isinstance(impact_payload, dict):
            raise LLMResponseError("Event impact must be an object.")
        for field_name in ("power_shifts", "relation_changes", "world_flags"):
            if field_name not in impact_payload:
                raise LLMResponseError(f"Event impact missing required field: {field_name}")
        event_type = self._required_str(payload.get("type"), "type")
        if event_type not in {"conflict", "reveal", "progress", "setback"}:
            raise LLMResponseError(f"Invalid event type: {event_type!r}")
        summary = self._required_str(payload.get("summary"), "summary")
        if not summary:
            raise LLMResponseError("Event summary is required.")
        participants_payload = payload.get("participants")
        if not isinstance(participants_payload, list):
            raise LLMResponseError("Event participants must be a list.")
        participants: list[str] = []
        for name in participants_payload:
            if not isinstance(name, str):
                raise LLMResponseError("Event participant names must be strings.")
            stripped = name.strip()
            if stripped:
                participants.append(alias_map.get(stripped, stripped))
        unknown = sorted(set(participants) - allowed_participants)
        if unknown:
            raise LLMResponseError("Event participants must come from roster: " + ", ".join(unknown))
        is_irreversible = payload.get("is_irreversible")
        if not isinstance(is_irreversible, bool):
            raise LLMResponseError("Event is_irreversible must be a boolean.")
        chapter_value = payload.get("chapter")
        if not isinstance(chapter_value, int) or isinstance(chapter_value, bool):
            raise LLMResponseError("Event chapter must be an integer.")
        impact = EventImpact(
            power_shifts=self._str_dict(impact_payload.get("power_shifts"), "power_shifts"),
            relation_changes=self._str_dict(impact_payload.get("relation_changes"), "relation_changes"),
            world_flags=self._str_list(impact_payload.get("world_flags"), "world_flags"),
        )
        return PlotEvent(
            event_id=event_id,
            chapter=chapter_value,
            type=event_type,
            participants=participants,
            summary=summary,
            impact=impact,
            is_irreversible=is_irreversible,
        )

    def _parse_response(
        self,
        content: str | None,
        chapter_no: int,
        alias_map: dict[str, str],
        allowed_participants: set[str],
    ) -> tuple[list[PlotEvent] | None, str | None]:
        """Attempt to parse structured events from the model output."""
        if not content:
            return None, "Empty response body."
        payload = load_json_response(content)
        if payload is None:
            return None, "Response body was not valid JSON."
        if not isinstance(payload, list):
            payload_repr = repr(payload)[:500]
            return None, (
                "Event extractor response is not a list. "
                f"Got type={type(payload).__name__}, value={payload_repr}"
            )
        events: list[PlotEvent] = []
        for idx, item in enumerate(payload, start=1):
            if not isinstance(item, dict):
                return None, f"Event payload at index {idx} is not an object."
            try:
                events.append(self._from_payload(item, chapter_no, alias_map, allowed_participants))
            except LLMResponseError as exc:
                return None, str(exc)
        return events, None

    def _str_dict(self, value: Any, field_name: str) -> dict[str, str]:
        if not isinstance(value, dict):
            raise LLMResponseError(f"Event impact {field_name} must be an object.")
        result: dict[str, str] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not isinstance(item, str):
                raise LLMResponseError(f"Event impact {field_name} keys and values must be strings.")
            if key.strip() and item.strip():
                result[key.strip()] = item.strip()
        return result

    def _str_list(self, value: Any, field_name: str) -> list[str]:
        if not isinstance(value, list):
            raise LLMResponseError(f"Event impact {field_name} must be a list.")
        items: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise LLMResponseError(f"Event impact {field_name} items must be strings.")
            text = item.strip()
            if text:
                items.append(text)
        return items

    def _required_str(self, value: Any, field_name: str) -> str:
        if not isinstance(value, str):
            raise LLMResponseError(f"Event {field_name} must be a string.")
        text = value.strip()
        if not text:
            raise LLMResponseError(f"Event {field_name} must not be empty.")
        return text

    def _build_roster(
        self,
        participants: list[str] | list[CharacterState],
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        roster_by_name: dict[str, dict[str, Any]] = {}
        alias_map: dict[str, str] = {}
        for participant in participants:
            if isinstance(participant, CharacterState):
                name = participant.name.strip()
                aliases = [alias.strip() for alias in participant.aliases if alias.strip()]
            elif isinstance(participant, str):
                name = participant.strip()
                aliases = []
            else:
                raise LLMResponseError("Event roster entries must be strings or CharacterState objects.")
            if not name:
                continue
            roster_by_name[name] = {"name": name, "aliases": sorted(set(aliases))}
            alias_map[name] = name
            for alias in aliases:
                alias_map[alias] = name
        return [roster_by_name[name] for name in sorted(roster_by_name)], alias_map

    def _attempt_repair(self, raw_response: str | None, error: str | None, chapter_no: int) -> str | None:
        """Ask the LLM to fix its malformed JSON response."""
        if not raw_response:
            return None
        snippet = clamp_text_middle(raw_response, _REPAIR_SNIPPET_BUDGET)
        prompt = fill_template(
            self.repair_template,
            error_message=error or "JSON parsing failed.",
            invalid_output=snippet,
        )
        repaired = self.llm.generate(
            prompt=prompt,
            options=telemetry_options(
                _STRUCTURED_OPTIONS,
                phase=f"analyze chapter {chapter_no}",
                step=": repair event JSON",
            ),
        )
        return repaired.content
