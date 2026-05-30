from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from story_for_you.analysis.context import PlotEvent, StoryState
from story_for_you.analysis.prompting import fill_template, load_template
from story_for_you.core.exceptions import LLMResponseError
from story_for_you.llm.base import LLMProvider
from story_for_you.llm.telemetry import telemetry_options
from story_for_you.utils.json_utils import load_json_response


_VALID_ARCS = {"setup", "journey", "twist", "climax", "dark-night", "resolution"}
_VALID_TENSIONS = {"low", "medium", "high"}
_STRUCTURED_OPTIONS = {"no_think": True, "temperature": 0.1}


class StateSynthesizer:
    """Produces an updated StoryState from plot events."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm
        self.template = load_template("state_update")

    def update(
        self,
        story_state: StoryState | None,
        events: list[PlotEvent],
        recent_context: str,
    ) -> StoryState:
        """Synthesize the long-term story state via the LLM."""
        prior_state_payload: Any = asdict(story_state) if story_state else None
        events_payload = [asdict(event) for event in events]
        prompt = fill_template(
            self.template,
            prior_state=(
                "null" if prior_state_payload is None else json.dumps(prior_state_payload, ensure_ascii=False)
            ),
            events=json.dumps(events_payload, ensure_ascii=False),
            recent_context=recent_context.strip() or "暂无历史上下文。",
        )
        response = self.llm.generate(
            prompt=prompt,
            options=telemetry_options(
                _STRUCTURED_OPTIONS,
                phase="analyze chapter",
                step=": update story state",
            ),
        )
        data = load_json_response(response.content)
        if not isinstance(data, dict):
            raise LLMResponseError("Story state response is not a JSON object.")
        for field_name in (
            "current_arc",
            "world_tension",
            "major_conflicts",
            "time_constraints",
            "unresolved_events",
        ):
            if field_name not in data:
                raise LLMResponseError(f"Story state response missing required field: {field_name}")
        current_arc = self._required_str(data.get("current_arc"), "current_arc")
        world_tension = self._required_str(data.get("world_tension"), "world_tension")
        if current_arc not in _VALID_ARCS:
            raise LLMResponseError(f"Invalid story arc: {current_arc!r}")
        if world_tension not in _VALID_TENSIONS:
            raise LLMResponseError(f"Invalid world tension: {world_tension!r}")
        return StoryState(
            current_arc=current_arc,
            world_tension=world_tension,
            major_conflicts=self._required_str_list(data.get("major_conflicts"), "major_conflicts")[:5],
            time_constraints=self._required_str_list(data.get("time_constraints"), "time_constraints")[:3],
            unresolved_events=self._required_str_list(data.get("unresolved_events"), "unresolved_events")[:5],
        )

    def _required_str_list(self, value: Any, field_name: str) -> list[str]:
        if not isinstance(value, list):
            raise LLMResponseError(f"Story state field must be a list: {field_name}")
        items: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise LLMResponseError(f"Story state list items must be strings: {field_name}")
            text = item.strip()
            if text:
                items.append(text)
        return items

    def _required_str(self, value: Any, field_name: str) -> str:
        if not isinstance(value, str):
            raise LLMResponseError(f"Story state field must be a string: {field_name}")
        text = value.strip()
        if not text:
            raise LLMResponseError(f"Story state field must not be empty: {field_name}")
        return text
