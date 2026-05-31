"""LLM-backed user directive extraction for ending generation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from story_for_you.core.exceptions import LLMResponseError
from story_for_you.core.prompting import format_context_sections, load_template
from story_for_you.llm.base import LLMProvider
from story_for_you.llm.telemetry import telemetry_options
from story_for_you.utils.json_utils import load_json_response
from story_for_you.utils.prompting import build_cacheable_prompt

if TYPE_CHECKING:
    from story_for_you.analysis.context import StoryContext
    from story_for_you.config.settings import RenderingLimits

__all__ = [
    "HintDirectives",
    "HintInterpreter",
]

_VALID_CLOSURES = {"open", "closed", "unspecified"}
_VALID_DIRECTIONS = {"HE", "BE", "OE"}


@dataclass
class HintDirectives:
    """Structured reader intent extracted by the model."""

    normalized_text: str = "无特别要求"
    ending_direction: str | None = None
    emotional_tone: str | None = None
    focus_characters: list[str] = field(default_factory=list)
    closure: str = "unspecified"
    required_outcomes: list[str] = field(default_factory=list)
    forbidden_outcomes: list[str] = field(default_factory=list)
    required_resolutions: list[str] = field(default_factory=list)
    style_constraints: list[str] = field(default_factory=list)

    def for_prompt(self) -> str:
        """Render structured directives for downstream generation prompts."""

        payload = {
            "normalized_text": self.normalized_text,
            "closure": self.closure,
            "ending_direction": self.ending_direction,
            "emotional_tone": self.emotional_tone,
            "focus_characters": self.focus_characters,
            "required_outcomes": self.required_outcomes,
            "forbidden_outcomes": self.forbidden_outcomes,
            "required_resolutions": self.required_resolutions,
            "style_constraints": self.style_constraints,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)


class HintInterpreter:
    """Delegates semantic interpretation of user hints to the configured LLM."""

    def __init__(self, llm: LLMProvider, rendering_limits: RenderingLimits | None = None):
        self.llm = llm
        self._limits = rendering_limits
        self.template = load_template("ending_directive")

    def interpret(self, raw_hint: str, context: StoryContext) -> HintDirectives:
        """Extract structured directives without local keyword matching."""

        hint = (raw_hint or "").strip() or "无特别要求"
        context_block = format_context_sections(context.for_prompt(limits=self._limits))
        prompt = build_cacheable_prompt(
            context_block or "(无上下文)",
            self.template,
            prefix_placeholder="context_block",
            raw_hint=hint,
        )
        response = self.llm.generate(
            prompt=prompt,
            options=telemetry_options(
                {"no_think": True},
                phase="continue",
                step=": interpret user hint",
            ),
        )
        payload = load_json_response(response.content)
        if not isinstance(payload, dict):
            raise LLMResponseError("Ending directive extraction returned invalid JSON object.")
        return self._from_payload(payload, hint)

    def _from_payload(self, payload: dict[str, Any], raw_hint_text: str) -> HintDirectives:
        for field_name in (
            "normalized_text",
            "closure",
            "ending_direction",
            "emotional_tone",
            "focus_characters",
            "required_outcomes",
            "forbidden_outcomes",
            "required_resolutions",
            "style_constraints",
        ):
            if field_name not in payload:
                raise LLMResponseError(f"Ending directive missing required field: {field_name}")
        closure = self._required_str(payload.get("closure"), "closure").lower()
        if closure not in _VALID_CLOSURES:
            raise LLMResponseError(f"Invalid closure value from directive extractor: {closure!r}")

        ending_direction = payload.get("ending_direction")
        if ending_direction is not None:
            if not isinstance(ending_direction, str):
                raise LLMResponseError("Ending direction must be a string or null.")
            ending_direction = ending_direction.strip() or None
            if ending_direction and ending_direction not in _VALID_DIRECTIONS:
                raise LLMResponseError(f"Invalid ending direction: {ending_direction!r}")

        return HintDirectives(
            normalized_text=self._required_str(payload.get("normalized_text"), "normalized_text"),
            ending_direction=ending_direction,
            emotional_tone=self._optional_str(payload.get("emotional_tone")),
            focus_characters=self._str_list(payload.get("focus_characters")),
            closure=closure,
            required_outcomes=self._str_list(payload.get("required_outcomes")),
            forbidden_outcomes=self._str_list(payload.get("forbidden_outcomes")),
            required_resolutions=self._str_list(payload.get("required_resolutions")),
            style_constraints=self._str_list(payload.get("style_constraints")),
        )

    def _optional_str(self, value: Any) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise LLMResponseError("Ending directive optional string fields must be strings or null.")
        text = value.strip()
        return text or None

    def _required_str(self, value: Any, field_name: str) -> str:
        if not isinstance(value, str):
            raise LLMResponseError(f"Ending directive {field_name} must be a string.")
        text = value.strip()
        if not text:
            raise LLMResponseError(f"Ending directive {field_name} must not be empty.")
        return text

    def _str_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise LLMResponseError("Ending directive list fields must be JSON arrays.")
        items: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise LLMResponseError("Ending directive list items must be strings.")
            text = item.strip()
            if text:
                items.append(text)
        return items
