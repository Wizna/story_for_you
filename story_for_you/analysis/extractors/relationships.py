from __future__ import annotations

import json
from typing import Any

from story_for_you.analysis.context import CharacterState, Relationship
from story_for_you.analysis.prompting import build_cacheable_prompt, load_template
from story_for_you.core.exceptions import LLMResponseError
from story_for_you.llm.base import LLMProvider
from story_for_you.llm.telemetry import telemetry_options
from story_for_you.utils.json_utils import load_json_response
from story_for_you.utils.prompting import cache_prompt

_VALID_SENTIMENTS = {"positive", "neutral", "negative"}
_MAX_RELATIONSHIP_ATTEMPTS = 2
_DEFAULT_MAX_ROSTER_CHARACTERS = 96
_STRUCTURED_OPTIONS = {"no_think": True, "temperature": 0.1}


class RelationshipMapper:
    """Maps relationship deltas between characters via the configured LLM."""

    def __init__(
        self,
        llm: LLMProvider,
        max_roster_characters: int = _DEFAULT_MAX_ROSTER_CHARACTERS,
        require_evidence: bool = False,
    ):
        if max_roster_characters <= 0:
            raise ValueError("max_roster_characters must be positive")
        self.llm = llm
        self.template = load_template("relationship_extraction")
        self.max_roster_characters = max_roster_characters
        self.require_evidence = require_evidence

    def map(self, chapter_text: str, characters: list[str] | list[CharacterState] | None = None) -> list[Relationship]:
        """Return relationship changes observed in the text."""
        roster, alias_map = self._build_roster(characters or [], chapter_text)
        if len(roster) < 2:
            return []
        prompt = build_cacheable_prompt(
            chapter_text.strip(),
            self.template,
            prefix_placeholder="chapter_text",
            character_roster=json.dumps(roster, ensure_ascii=False),
        )
        allowed = set(alias_map)
        last_error: str | None = None
        for attempt in range(1, _MAX_RELATIONSHIP_ATTEMPTS + 1):
            response = self.llm.generate(
                prompt=prompt,
                options=telemetry_options(
                    _STRUCTURED_OPTIONS,
                    phase="analyze chapter",
                    step=": map relationships",
                    attempt=attempt,
                    max_attempts=_MAX_RELATIONSHIP_ATTEMPTS,
                ),
            )
            relationships, error = self._parse_response(
                response.content, alias_map, allowed, chapter_text
            )
            if relationships is not None:
                return relationships
            last_error = error
            if attempt < _MAX_RELATIONSHIP_ATTEMPTS:
                prompt = cache_prompt(
                    prompt.task + "\n\n上次输出未通过校验：" + (error or "未知错误")
                    + "\n请对照原文和角色清单修正，保留有效关系；清单外人物不能擅自映射给另一人。严格输出完整 JSON 数组。"
                    + "\n上次输出：\n" + (response.content or "(空)"),
                    prefix=prompt.prefix,
                )
        raise LLMResponseError(f"Relationship extraction failed after retry: {last_error or 'unknown parse failure'}")

    def _parse_response(
        self,
        content: str | None,
        alias_map: dict[str, str],
        allowed_characters: set[str],
        source_text: str = "",
    ) -> tuple[list[Relationship] | None, str | None]:
        if not content:
            return None, "Empty response body."
        payload = load_json_response(content)
        if not isinstance(payload, list):
            return None, "Relationship mapper response is not a JSON list."
        relationships: list[Relationship] = []
        for idx, item in enumerate(payload, start=1):
            try:
                relationship = self._from_payload(item, alias_map, allowed_characters)
                if self.require_evidence:
                    evidence = relationship.evidence.strip()
                    if not evidence or "".join(evidence.split()) not in "".join(source_text.split()):
                        raise LLMResponseError(
                            "Relationship evidence must quote the supplied chapter text."
                        )
                relationships.append(relationship)
            except LLMResponseError as exc:
                return None, f"Relationship payload at index {idx}: {exc}"
        return relationships, None

    def _from_payload(
        self,
        payload: Any,
        alias_map: dict[str, str],
        allowed_characters: set[str],
    ) -> Relationship:
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
        canonical_source = alias_map[source]
        canonical_targets = [alias_map[target] for target in targets]
        sentiment = self._required_str(payload.get("sentiment"), "sentiment")
        if sentiment not in _VALID_SENTIMENTS:
            raise LLMResponseError(f"Invalid relationship sentiment: {sentiment!r}")
        return Relationship(
            targets=canonical_targets,
            relation_type=self._required_str(payload.get("relation_type"), "relation_type"),
            sentiment=sentiment,
            description=self._required_str(payload.get("description"), "description", allow_empty=True),
            source=canonical_source,
            confidence=self._confidence(payload.get("confidence", 0.0)),
            evidence=self._required_str(payload.get("evidence", ""), "evidence", allow_empty=True),
            chapter=payload.get("chapter") if isinstance(payload.get("chapter"), int) else None,
        )

    def _confidence(self, value: Any) -> float:
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0.0 <= value <= 1.0:
            raise LLMResponseError("Relationship confidence must be between 0 and 1.")
        return float(value)

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

    def _build_roster(
        self,
        characters: list[str] | list[CharacterState],
        chapter_text: str = "",
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        entries: list[tuple[str, list[str], str, int]] = []
        role_priority = {"main": 3, "support": 2, "minor": 1}
        text = chapter_text.casefold()
        alias_map: dict[str, str] = {}
        for character in characters:
            if isinstance(character, CharacterState):
                name = character.name.strip()
                aliases = [alias.strip() for alias in character.aliases if alias.strip()]
                role = character.role
            elif isinstance(character, str):
                name = character.strip()
                aliases = []
                role = "minor"
            else:
                raise LLMResponseError("Relationship roster entries must be strings or CharacterState objects.")
            if not name:
                continue
            entries.append((name, sorted(set(aliases)), role, len(entries)))
        if len(entries) > self.max_roster_characters:
            scored = []
            for name, aliases, role, order in entries:
                mentions = sum(
                    text.count(label.casefold())
                    for label in [name, *aliases]
                    if label and (label == name or len(label) >= 2)
                )
                scored.append((int(mentions > 0), mentions, role_priority.get(role, 0), -order, name, aliases))
            scored.sort(reverse=True)
            entries = [
                (name, aliases, "minor", order)
                for _, _, _, neg_order, name, aliases in scored[:self.max_roster_characters]
                for order in [-neg_order]
            ]
            entries.sort(key=lambda item: item[3])
        roster_by_name: dict[str, dict[str, Any]] = {}
        canonical_names = {name for name, _aliases, _role, _order in entries}
        alias_candidates: dict[str, set[str]] = {}
        for name, aliases, _role, _order in entries:
            roster_by_name[name] = {"name": name, "aliases": aliases}
            alias_map[name] = name
            for alias in aliases:
                alias_candidates.setdefault(alias, set()).add(name)
        for alias, owners in alias_candidates.items():
            if alias in canonical_names or len(owners) != 1:
                continue
            alias_map[alias] = next(iter(owners))
        return [roster_by_name[name] for name in sorted(roster_by_name)], alias_map
