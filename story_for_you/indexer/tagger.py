from __future__ import annotations

from story_for_you.analysis.context import StoryContext
from story_for_you.indexer.segment import Segment


class CharacterTagger:
    """Annotates segments with character participation."""

    def __init__(self, context: StoryContext):
        self.context = context
        self._patterns = self._prepare_patterns()

    def tag(self, segments: list[Segment]) -> list[Segment]:
        """Populate segment characters using context knowledge."""
        if not self._patterns:
            return segments
        for segment in segments:
            detected = self._detect(segment.content)
            if detected:
                segment.characters = sorted(detected)
        return segments

    def _prepare_patterns(self) -> dict[str, list[str]]:
        patterns: dict[str, list[str]] = {}
        for name, character in self.context.characters.items():
            variants = {name.lower()}
            for alias in character.aliases:
                if alias:
                    variants.add(alias.lower())
            # Single-character aliases are too noisy.
            filtered = [variant for variant in variants if len(variant.strip()) > 1]
            if filtered:
                patterns[name] = filtered
        return patterns

    def _detect(self, content: str) -> set[str]:
        hits: set[str] = set()
        text = content.lower()
        for name, variants in self._patterns.items():
            for variant in variants:
                if variant in text:
                    hits.add(name)
                    break
        return hits
