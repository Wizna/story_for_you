from __future__ import annotations

import itertools
import re

from story_for_you.analysis.context import Relationship
from story_for_you.llm.base import LLMProvider


class RelationshipMapper:
    """Maps relationship deltas between characters."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def map(self, chapter_text: str, characters: list[str] | None = None) -> list[Relationship]:
        """Return relationship changes observed in the text."""
        if not characters:
            return []
        relationships: list[Relationship] = []
        seen: set[tuple[str, str]] = set()
        sentences = re.split(r"(?<=[。.!?])\s+", chapter_text.strip())
        for sentence in sentences:
            present = [name for name in characters if name in sentence]
            if len(present) < 2:
                continue
            for source, target in itertools.permutations(sorted(set(present)), 2):
                key = (source, target)
                if key in seen:
                    continue
                relation_type = self._infer_type(sentence)
                sentiment = self._infer_sentiment(sentence)
                relationships.append(
                    Relationship(
                        target=target,
                        relation_type=relation_type,
                        sentiment=sentiment,
                        description=sentence.strip()[:160],
                        source=source,
                    )
                )
                seen.add(key)
        return relationships

    def _infer_type(self, sentence: str) -> str:
        lowered = sentence.lower()
        if any(keyword in lowered for keyword in ["battle", "fight", "argue", "敌"]):
            return "rival"
        if any(keyword in lowered for keyword in ["love", "ally", "救"]):
            return "ally"
        return "acquaintance"

    def _infer_sentiment(self, sentence: str) -> str:
        lowered = sentence.lower()
        if any(keyword in lowered for keyword in ["hate", "怒", "betray"]):
            return "negative"
        if any(keyword in lowered for keyword in ["comfort", "love", "拥抱", "救"]):
            return "positive"
        return "neutral"
