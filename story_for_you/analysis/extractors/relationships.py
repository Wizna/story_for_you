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
        return [self._from_payload(item) for item in payload]

    def _from_payload(self, payload: Any) -> Relationship:
        if not isinstance(payload, dict):
            raise LLMResponseError("Relationship item is not a JSON object.")
        for field_name in ("source", "targets", "relation_type", "sentiment", "description"):
            if field_name not in payload:
                raise LLMResponseError(f"Relationship item missing required field: {field_name}")
        source = str(payload.get("source", "")).strip()
        targets = self._str_list(payload.get("targets"))
        if not source or not targets:
            raise LLMResponseError("Relationship item must include source and targets.")
        sentiment = str(payload.get("sentiment")).strip()
        if sentiment not in _VALID_SENTIMENTS:
            raise LLMResponseError(f"Invalid relationship sentiment: {sentiment!r}")
        return Relationship(
            targets=targets,
            relation_type=str(payload.get("relation_type")).strip(),
            sentiment=sentiment,
            description=str(payload.get("description") or "").strip(),
            source=source,
        )

    def _str_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            raise LLMResponseError("Relationship targets must be a list.")
        return [str(item).strip() for item in value if str(item).strip()]
