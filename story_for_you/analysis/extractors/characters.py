from __future__ import annotations

from typing import Any, Iterable

import logging

from story_for_you.analysis.context import CharacterState
from story_for_you.analysis.names import normalize_character_label
from story_for_you.analysis.prompting import (
    CacheablePrompt,
    clamp_text_middle,
    fill_template,
    load_template,
    render_cacheable_prompt_with_budget,
)
from story_for_you.core.exceptions import LLMResponseError
from story_for_you.llm.base import LLMProvider
from story_for_you.llm.telemetry import telemetry_options
from story_for_you.utils.json_utils import load_json_response
from story_for_you.utils.prompting import cache_prompt

logger = logging.getLogger(__name__)

_MAX_LLM_CHARACTERS = 8
_REPAIR_SNIPPET_BUDGET = 4000
_STRUCTURED_OPTIONS = {"no_think": True, "temperature": 0.1}
_ROLE_PRIORITY: dict[str, int] = {"main": 3, "support": 2, "minor": 1}
_GENERIC_CHARACTER_TITLES = {
    "师父", "师傅", "老师", "师兄", "师姐", "师弟", "师妹", "大哥", "二哥",
    "三哥", "大嫂", "二嫂", "爹", "娘", "父亲", "母亲", "老爷", "夫人",
    "公子", "小姐", "掌门", "长老", "殿下", "陛下", "将军", "先生",
}


class CharacterExtractor:
    """Identifies characters present in the supplied text."""

    def __init__(self, llm: LLMProvider, prompt_budget: int | None = None):
        self.llm = llm
        self.personality_analyzer = PersonalityAnalyzer(llm)
        self.template = load_template("character_sheet")
        self.repair_template = load_template("character_sheet_repair")
        self.prompt_budget = prompt_budget

    def extract(self, text: str) -> list[CharacterState]:
        """Return structured character states for the text."""
        prompt = self._build_prompt(text)
        characters = self._prompt_characters(prompt)
        merged = self.merge_aliases(characters)
        return self.personality_analyzer.analyze(merged)

    def merge_aliases(self, characters: Iterable[CharacterState]) -> list[CharacterState]:
        """Merge duplicate characters based on aliases and name overlap."""
        roster: list[CharacterState] = []
        for character in characters:
            target = self._find_character(roster, character)
            if not target:
                roster.append(character)
                continue
            self._merge_into(target, character)
        return roster

    def _build_prompt(self, text: str) -> CacheablePrompt:
        chapter_text = text.strip()
        prompt, truncated = render_cacheable_prompt_with_budget(
            self.template,
            budget=self.prompt_budget,
            prefix_key="chapter_text",
            prefix_value=chapter_text,
        )
        if truncated:
            logger.debug("Character prompt truncated to %s chars", len(prompt.render()))
        return prompt

    # Internal helpers -------------------------------------------------
    def _prompt_characters(self, prompt: CacheablePrompt) -> list[CharacterState]:
        """Use the configured LLM to extract structured characters."""
        response = self.llm.generate(
            prompt=prompt,
            options=telemetry_options(
                _STRUCTURED_OPTIONS,
                phase="analyze chapter",
                step=": extract characters",
            ),
        )
        characters, error = self._parse_response(response.content)
        if characters is not None:
            return characters

        if error:
            logger.debug("Character extraction parse failed (%s). Attempting repair.", error)
        repaired_content = self._attempt_repair(response.content, error)
        if repaired_content:
            characters, repair_error = self._parse_response(repaired_content)
            if characters is not None:
                return characters
            error = repair_error or error

        raw_content = response.content[:1000] if response.content else "(empty)"
        logger.warning(
            "Failed to parse character extraction response after repair: %s\nRaw response (truncated): %s",
            error or "Unknown parse failure",
            raw_content,
        )
        raise LLMResponseError(f"Character extraction failed after repair: {error or 'unknown parse failure'}")

    def _parse_response(self, content: str | None) -> tuple[list[CharacterState] | None, str | None]:
        if not content:
            return None, "Empty response body."
        payload = load_json_response(content)
        if payload is None:
            return None, "Character extractor returned invalid JSON payload."
        if not isinstance(payload, list):
            return None, "Character extractor payload is not a list."
        characters: list[CharacterState] = []
        for idx, item in enumerate(payload[:_MAX_LLM_CHARACTERS], start=1):
            if not isinstance(item, dict):
                return None, f"Character payload at index {idx} is not an object."
            try:
                character = self._to_character(item)
            except LLMResponseError as exc:
                return None, str(exc)
            if character:
                characters.append(character)
        return characters, None

    def _to_character(self, data: dict) -> CharacterState | None:
        for field_name in ("name", "aliases", "role", "realm", "personality", "unresolved"):
            if field_name not in data:
                raise LLMResponseError(f"Character item missing required field: {field_name}")
        name = self._required_str(data.get("name"), "name")
        if not name:
            raise LLMResponseError("Character item requires a non-empty name.")
        aliases = self._normalize_str_list(data.get("aliases", []))
        unresolved = self._normalize_str_list(data.get("unresolved", []))
        personality = self._localize_personality(self._normalize_str_list(data.get("personality", [])))
        role = self._normalize_role(data.get("role"))
        realm_payload = data.get("realm")
        if realm_payload is not None and not isinstance(realm_payload, str):
            raise LLMResponseError("Character realm must be a string or null.")
        realm = realm_payload.strip() or None if isinstance(realm_payload, str) else None
        status = data.get("status", "unknown")
        if status not in {"alive", "dead", "unknown"}:
            raise LLMResponseError("Character status must be alive, dead, or unknown.")
        location = data.get("location")
        goal = data.get("goal")
        if location is not None and not isinstance(location, str):
            raise LLMResponseError("Character location must be a string or null.")
        if goal is not None and not isinstance(goal, str):
            raise LLMResponseError("Character goal must be a string or null.")
        knowledge = self._normalize_str_list(data.get("knowledge", []))
        return CharacterState(
            name=name,
            aliases=aliases,
            realm=realm,
            role=role,
            personality=personality,
            relationships=[],
            unresolved=unresolved,
            status=status,
            location=location.strip() or None if isinstance(location, str) else None,
            goal=goal.strip() or None if isinstance(goal, str) else None,
            knowledge=knowledge,
        )

    def _normalize_role(self, value: Any) -> str:
        if not isinstance(value, str):
            raise LLMResponseError("Character role must be a string.")
        lowered = value.strip().lower()
        if lowered in _ROLE_PRIORITY:
            return lowered
        raise LLMResponseError(f"Invalid character role: {value!r}")

    def _find_character(self, roster: list[CharacterState], candidate: CharacterState) -> CharacterState | None:
        # Canonical names are safe identity keys.  Alias-only matches are
        # accepted only when that alias belongs to one character in this
        # chapter; common titles such as “师父” must remain separate.
        candidate_name = normalize_character_label(candidate.name)
        for existing in roster:
            if candidate_name == normalize_character_label(existing.name):
                return existing

        owners: dict[str, list[CharacterState]] = {}
        for existing in roster:
            for key in self._alias_keys(existing):
                owners.setdefault(key, []).append(existing)
        for key in self._alias_keys(candidate):
            if key in _GENERIC_CHARACTER_TITLES:
                continue
            matches = owners.get(key, [])
            if len(matches) == 1:
                return matches[0]

        return None

    def _merge_into(self, target: CharacterState, incoming: CharacterState) -> None:
        merged_aliases = set(target.aliases)
        merged_aliases.add(incoming.name)
        merged_aliases.update(incoming.aliases)
        target.aliases = sorted(alias for alias in merged_aliases if alias and alias != target.name)
        target.personality = self._localize_personality(
            self._merge_list(target.personality, incoming.personality)
        )
        target.unresolved = self._merge_list(target.unresolved, incoming.unresolved)
        target.realm = target.realm or incoming.realm
        if incoming.status != "unknown":
            target.status = incoming.status
        target.location = incoming.location or target.location
        target.goal = incoming.goal or target.goal
        target.knowledge = self._merge_list(target.knowledge, incoming.knowledge)
        if _ROLE_PRIORITY[incoming.role] > _ROLE_PRIORITY[target.role]:
            target.role = incoming.role

    def _alias_keys(self, character: CharacterState) -> set[str]:
        return {
            normalize_character_label(name)
            for name in [character.name, *character.aliases]
            if name.strip()
        }

    def _merge_list(self, base: list[str], incoming: list[str]) -> list[str]:
        merged = list(dict.fromkeys(item for item in base + incoming if item))
        return merged

    def _normalize_str_list(self, value: Any) -> list[str]:
        """Ensure we always work with a list of trimmed strings."""
        if not isinstance(value, list):
            raise LLMResponseError("Character string-list fields must be JSON arrays.")
        items: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise LLMResponseError("Character string-list items must be strings.")
            text = item.strip()
            if text:
                items.append(text)
        return items

    def _required_str(self, value: Any, field_name: str) -> str:
        if not isinstance(value, str):
            raise LLMResponseError(f"Character {field_name} must be a string.")
        return value.strip()

    def _localize_personality(self, traits: Iterable[str]) -> list[str]:
        normalized_traits: list[str] = []
        for trait in traits:
            normalized = trait.strip()
            if not normalized:
                continue
            normalized_traits.append(normalized)
        return list(dict.fromkeys(normalized_traits))

    def _attempt_repair(self, raw_response: str | None, error: str | None) -> str | None:
        """Ask the LLM to repair malformed character JSON once."""
        if not raw_response:
            return None
        snippet = clamp_text_middle(raw_response, _REPAIR_SNIPPET_BUDGET)
        prompt = fill_template(
            self.repair_template,
            error_message=error or "JSON parsing failed.",
            invalid_output=snippet,
        )
        repaired = self.llm.generate(
            prompt=cache_prompt(prompt),
            options=telemetry_options(
                _STRUCTURED_OPTIONS,
                phase="analyze chapter",
                step=": repair character JSON",
            ),
        )
        return repaired.content


class PersonalityAnalyzer:
    """Derives personality anchors for the detected characters."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def analyze(self, characters: Iterable[CharacterState]) -> list[CharacterState]:
        return list(characters)
