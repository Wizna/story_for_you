from typing import Iterable

from story_for_you.analysis.context import CharacterState
from story_for_you.llm.base import LLMProvider


class CharacterExtractor:
    """Identifies characters present in the supplied text."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def extract(self, text: str) -> list[CharacterState]:
        """Return structured character states for the text."""
        raise NotImplementedError


class PersonalityAnalyzer:
    """Derives personality anchors for the detected characters."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def analyze(self, characters: Iterable[CharacterState]) -> list[CharacterState]:
        """Enrich characters with additional traits."""
        raise NotImplementedError
