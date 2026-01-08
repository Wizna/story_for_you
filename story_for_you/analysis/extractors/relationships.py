from __future__ import annotations

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
        seen: set[tuple[str, tuple[str, ...]]] = set()
        sentences = re.split(r"(?<=[。.!?])\s+", chapter_text.strip())
        for sentence in sentences:
            participants = sorted({name for name in characters if name in sentence})
            if len(participants) < 2:
                continue
            relation_type = self._infer_type(sentence)
            sentiment = self._infer_sentiment(sentence)
            description = self._render_description(sentence)
            for source in participants:
                targets = tuple(name for name in participants if name != source)
                if not targets:
                    continue
                key = (source, targets)
                if key in seen:
                    continue
                relationships.append(
                    Relationship(
                        targets=list(targets),
                        relation_type=relation_type,
                        sentiment=sentiment,
                        description=description,
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

    def _render_description(self, sentence: str) -> str:
        """Return a compact, readable description snippet."""
        collapsed = re.sub(r"\s+", " ", sentence).strip()
        return collapsed
