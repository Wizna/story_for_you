from __future__ import annotations

import json
from typing import Any

from story_for_you.analysis.context import Relationship
from story_for_you.analysis.prompting import fill_template, load_template
from story_for_you.core.exceptions import LLMResponseError
from story_for_you.llm.base import LLMProvider
from story_for_you.utils.json_utils import load_json_response

_VALID_SENTIMENTS = {"positive", "neutral", "negative"}


class RelationshipMapper:
    """Maps relationship deltas between characters via the configured LLM."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm
        self.template = load_template("relationship_extraction")

    def map(self, chapter_text: str, characters: list[str] | None = None) -> list[Relationship]:
        """Return relationship changes observed in the text."""
        roster = [name for name in (characters or []) if name]
        if not roster:
            return []
        prompt = fill_template(
            self.template,
            chapter_text=chapter_text.strip(),
            character_roster=json.dumps(roster, ensure_ascii=False),
        )
        response = self.llm.generate(prompt=prompt, options={"no_think": True})
        payload = load_json_response(response.content)
        if not isinstance(payload, list):
            raise LLMResponseError("Relationship mapper response is not a JSON list.")
        allowed = set(roster)
        return [self._from_payload(item, allowed) for item in payload]

    def _from_payload(self, payload: Any, allowed_characters: set[str]) -> Relationship:
        if not isinstance(payload, dict):
            raise LLMResponseError("Relationship item is not a JSON object.")
        for field_name in ("source", "targets", "relation_type", "sentiment", "description"):
            if field_name not in payload:
                raise LLMResponseError(f"Relationship item missing required field: {field_name}")
        source = self._required_str(payload.get("source"), "source")
        targets = self._str_list(payload.get("targets"))
        if not source or not targets:
            raise LLMResponseError("Relationship item must include source and targets.")
        unknown = sorted(({source} | set(targets)) - allowed_characters)
        if unknown:
            raise LLMResponseError("Relationship names must come from roster: " + ", ".join(unknown))
        sentiment = self._required_str(payload.get("sentiment"), "sentiment")
        if sentiment not in _VALID_SENTIMENTS:
            raise LLMResponseError(f"Invalid relationship sentiment: {sentiment!r}")
        return Relationship(
            targets=targets,
            relation_type=self._required_str(payload.get("relation_type"), "relation_type"),
            sentiment=sentiment,
            description=self._required_str(payload.get("description"), "description", allow_empty=True),
            source=source,
        )

    def _required_str(self, value: Any, field_name: str, *, allow_empty: bool = False) -> str:
        if not isinstance(value, str):
            raise LLMResponseError(f"Relationship {field_name} must be a string.")
        text = value.strip()
        if not allow_empty and not text:
            raise LLMResponseError(f"Relationship {field_name} must not be empty.")
        return text

    def _str_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            raise LLMResponseError("Relationship targets must be a list.")
        items: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise LLMResponseError("Relationship target names must be strings.")
            text = item.strip()
            if text:
                items.append(text)
        return items
