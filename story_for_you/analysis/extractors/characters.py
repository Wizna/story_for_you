from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

import logging
import re

from story_for_you.analysis.context import CharacterState
from story_for_you.analysis.prompting import load_template, render_prompt_with_budget
from story_for_you.llm.base import LLMProvider
from story_for_you.utils.chinese_name_utils import split_compound_chinese_name
from story_for_you.utils.json_utils import load_json_response

logger = logging.getLogger(__name__)


class CharacterExtractor:
    """Identifies characters present in the supplied text."""

    ROLE_PRIORITY = {"main": 3, "support": 2, "minor": 1}
    TRAIT_TRANSLATIONS = {
        "resolute": "坚毅",
        "protective": "守护欲强",
        "curious": "好奇",
        "empathetic": "善解人意",
        "calculating": "精明",
        "ambitious": "野心勃勃",
    }

    def __init__(self, llm: LLMProvider, prompt_budget: int | None = None):
        self.llm = llm
        self.personality_analyzer = PersonalityAnalyzer(llm)
        self.template = load_template("character_sheet")
        self.prompt_budget = prompt_budget

    def extract(self, text: str) -> list[CharacterState]:
        """Return structured character states for the text."""
        prompt = self._build_prompt(text)
        characters = self._prompt_characters(prompt)
        if not characters:
            characters = self._heuristic_extract(text)
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

    def _collect_candidates(self, text: str) -> Counter[str]:
        latin_names = re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", text)
        chinese_names = re.findall(r"[\u4e00-\u9fff]{2,3}", text)
        tokens = latin_names + chinese_names
        return Counter(tokens)

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
        try:
            response = self.llm.generate(prompt=prompt)
        except Exception as exc:  # pragma: no cover - defensive against provider issues
            logger.warning("Character extraction prompt failed: %s", exc)
            return []
        payload = load_json_response(response.content)
        if payload is None:
            logger.warning("Character extractor returned invalid JSON payload.")
            return []
        if not isinstance(payload, list):
            logger.warning("Character extractor payload is not a list.")
            return []
        characters: list[CharacterState] = []
        for item in payload[:8]:
            character = self._to_character(item)
            if character:
                characters.append(character)
        return characters

    def _to_character(self, data: dict) -> CharacterState | None:
        name = str(data.get("name", "")).strip()
        if not name:
            return None
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
        if lowered in self.ROLE_PRIORITY:
            return lowered
        return "minor"

    def _heuristic_extract(self, text: str) -> list[CharacterState]:
        candidates = self._collect_candidates(text)
        if not candidates:
            return []
        characters: list[CharacterState] = []
        for idx, (name, _) in enumerate(candidates.most_common(8)):
            role = "main" if idx < 2 else "support" if idx < 5 else "minor"
            characters.append(
                CharacterState(
                    name=name,
                    role=role,
                    aliases=[],
                    personality=[],
                    relationships=[],
                )
            )
        return characters

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
            if self._names_have_overlap(candidate_names, existing_names):
                return existing

        return None

    def _names_have_overlap(self, names1: list[str], names2: list[str]) -> bool:
        """检测两组名字是否有实质性重叠（子串匹配）。"""
        for n1 in names1:
            n1_clean = n1.strip().lower()
            if len(n1_clean) < 2:
                continue
            for n2 in names2:
                n2_clean = n2.strip().lower()
                if len(n2_clean) < 2:
                    continue
                shorter, longer = (n1_clean, n2_clean) if len(n1_clean) <= len(n2_clean) else (n2_clean, n1_clean)
                if len(shorter) >= 2 and shorter in longer:
                    if len(shorter) >= len(longer) * 0.5 or (len(shorter) >= 2 and len(longer) <= 4):
                        return True
        return False

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
        if self.ROLE_PRIORITY[incoming.role] > self.ROLE_PRIORITY[target.role]:
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
        localized: list[str] = []
        for trait in traits:
            normalized = trait.strip()
            if not normalized:
                continue
            translation = self.TRAIT_TRANSLATIONS.get(normalized.lower())
            localized.append(translation or normalized)
        return list(dict.fromkeys(localized))


class PersonalityAnalyzer:
    """Derives personality anchors for the detected characters."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def analyze(self, characters: Iterable[CharacterState]) -> list[CharacterState]:
        """Enrich characters with additional traits."""
        enriched: list[CharacterState] = []
        for character in characters:
            if not character.personality:
                traits = self._guess_traits(character.name)
                character.personality = traits
            enriched.append(character)
        return enriched

    def _guess_traits(self, name: str) -> list[str]:
        bucket = sum(ord(ch) for ch in name) % 3
        if bucket == 0:
            return ["坚毅", "守护欲强"]
        if bucket == 1:
            return ["好奇", "善解人意"]
        return ["精明", "野心勃勃"]
