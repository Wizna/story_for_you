from __future__ import annotations

from collections import Counter
from typing import Iterable

import re

from story_for_you.analysis.context import CharacterState
from story_for_you.llm.base import LLMProvider


class CharacterExtractor:
    """Identifies characters present in the supplied text."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm
        self.personality_analyzer = PersonalityAnalyzer(llm)

    def extract(self, text: str) -> list[CharacterState]:
        """Return structured character states for the text."""
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
        return self.personality_analyzer.analyze(characters)

    def _collect_candidates(self, text: str) -> Counter[str]:
        latin_names = re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", text)
        chinese_names = re.findall(r"[\u4e00-\u9fff]{2,3}", text)
        tokens = latin_names + chinese_names
        return Counter(tokens)


class PersonalityAnalyzer:
    """Derives personality anchors for the detected characters."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def analyze(self, characters: Iterable[CharacterState]) -> list[CharacterState]:
        """Enrich characters with additional traits."""
        enriched: list[CharacterState] = []
        for character in characters:
            traits = self._guess_traits(character.name)
            character.personality = traits
            enriched.append(character)
        return enriched

    def _guess_traits(self, name: str) -> list[str]:
        bucket = sum(ord(ch) for ch in name) % 3
        if bucket == 0:
            return ["resolute", "protective"]
        if bucket == 1:
            return ["curious", "empathetic"]
        return ["calculating", "ambitious"]
