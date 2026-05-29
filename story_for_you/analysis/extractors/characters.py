from __future__ import annotations

from typing import Any, Iterable

import logging

from story_for_you.analysis.context import CharacterState
from story_for_you.analysis.prompting import load_template, render_prompt_with_budget
from story_for_you.core.exceptions import LLMResponseError
from story_for_you.llm.base import LLMProvider
from story_for_you.utils.chinese_name_utils import (
    ROLE_PRIORITY,
    names_have_overlap,
    split_compound_chinese_name,
)
from story_for_you.utils.json_utils import load_json_response

logger = logging.getLogger(__name__)

_MAX_LLM_CHARACTERS = 8


class CharacterExtractor:
    """Identifies characters present in the supplied text."""

    def __init__(self, llm: LLMProvider, prompt_budget: int | None = None):
        self.llm = llm
        self.personality_analyzer = PersonalityAnalyzer(llm)
        self.template = load_template("character_sheet")
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

    def _build_prompt(self, text: str) -> str:
        chapter_text = text.strip()
        prompt, truncated = render_prompt_with_budget(
            self.template,
            budget=self.prompt_budget,
            text_key="chapter_text",
            text_value=chapter_text,
        )
        if truncated:
            logger.debug("Character prompt truncated to %s chars", len(prompt))
        return prompt

    # Internal helpers -------------------------------------------------
    def _prompt_characters(self, prompt: str) -> list[CharacterState]:
        """Use the configured LLM to extract structured characters."""
        response = self.llm.generate(prompt=prompt, options={"no_think": True})
        payload = load_json_response(response.content)
        if payload is None:
            raise LLMResponseError("Character extractor returned invalid JSON payload.")
        if not isinstance(payload, list):
            raise LLMResponseError("Character extractor payload is not a list.")
        characters: list[CharacterState] = []
        for item in payload[:_MAX_LLM_CHARACTERS]:
            if not isinstance(item, dict):
                raise LLMResponseError("Character extractor list item is not an object.")
            character = self._to_character(item)
            if character:
                characters.append(character)
        return characters

    def _to_character(self, data: dict) -> CharacterState | None:
        for field_name in ("name", "aliases", "role", "realm", "personality", "unresolved"):
            if field_name not in data:
                raise LLMResponseError(f"Character item missing required field: {field_name}")
        name = str(data.get("name", "")).strip()
        if not name:
            raise LLMResponseError("Character item requires a non-empty name.")
        aliases = self._normalize_str_list(data.get("aliases", []))
        unresolved = self._normalize_str_list(data.get("unresolved", []))
        personality = self._localize_personality(self._normalize_str_list(data.get("personality", [])))
        role = self._normalize_role(data.get("role"))
        realm = str(data.get("realm") or "").strip() or None
        return CharacterState(
            name=name,
            aliases=aliases,
            realm=realm,
            role=role,
            personality=personality,
            relationships=[],
            unresolved=unresolved,
        )

    def _normalize_role(self, value: str | None) -> str:
        lowered = (value or "").lower()
        if lowered in ROLE_PRIORITY:
            return lowered
        raise LLMResponseError(f"Invalid character role: {value!r}")

    def _find_character(self, roster: list[CharacterState], candidate: CharacterState) -> CharacterState | None:
        candidate_keys = self._alias_keys(candidate)
        for existing in roster:
            # 第一步：精确 token 匹配
            if candidate_keys.intersection(self._alias_keys(existing)):
                return existing

        # 第二步：子串匹配（针对中文名）
        candidate_names = [candidate.name] + candidate.aliases
        for existing in roster:
            existing_names = [existing.name] + existing.aliases
            if names_have_overlap(candidate_names, existing_names):
                return existing

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
        if ROLE_PRIORITY[incoming.role] > ROLE_PRIORITY[target.role]:
            target.role = incoming.role

    def _alias_keys(self, character: CharacterState) -> set[str]:
        keys: set[str] = set()
        all_names = [character.name] + character.aliases
        for name in all_names:
            # 添加原始名字
            keys.add(name.lower())
            # 添加拆分后的组成部分
            for part in split_compound_chinese_name(name):
                keys.add(part.lower())
        return keys

    def _merge_list(self, base: list[str], incoming: list[str]) -> list[str]:
        merged = list(dict.fromkeys(item for item in base + incoming if item))
        return merged

    def _normalize_str_list(self, value: Any) -> list[str]:
        """Ensure we always work with a list of trimmed strings."""
        if isinstance(value, str):
            raw = [value]
        elif isinstance(value, Iterable):
            raw = [item for item in value if isinstance(item, str)]
        else:
            raw = []
        return [item.strip() for item in raw if item and item.strip()]

    def _localize_personality(self, traits: Iterable[str]) -> list[str]:
        normalized_traits: list[str] = []
        for trait in traits:
            normalized = trait.strip()
            if not normalized:
                continue
            normalized_traits.append(normalized)
        return list(dict.fromkeys(normalized_traits))


class PersonalityAnalyzer:
    """Derives personality anchors for the detected characters."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def analyze(self, characters: Iterable[CharacterState]) -> list[CharacterState]:
        return list(characters)
