from __future__ import annotations

from story_for_you.analysis.context import StoryContext
from story_for_you.indexer.segment import Segment


def unambiguous_alias_owners(context: StoryContext) -> dict[str, str]:
    """Map aliases with one owner and no canonical-name collision."""
    canonical_names = {name.casefold() for name in context.characters}
    owners: dict[str, set[str]] = {}
    for name, character in context.characters.items():
        for alias in character.aliases:
            if alias:
                owners.setdefault(alias.casefold(), set()).add(name)
    return {
        alias: next(iter(names))
        for alias, names in owners.items()
        if len(names) == 1 and alias not in canonical_names
    }


class CharacterTagger:
    """Annotates segments with character participation."""

    def __init__(self, context: StoryContext):
        self.context = context
        self._patterns = self._prepare_patterns()

    def tag(self, segments: list[Segment]) -> list[Segment]:
        """Populate segment characters using context knowledge."""
        if not self.context.characters:
            return segments
        for segment in segments:
            segment.characters = sorted(self._detect(segment.content))
        return segments

    def _prepare_patterns(self) -> dict[str, list[str]]:
        alias_owners = unambiguous_alias_owners(self.context)
        patterns: dict[str, list[str]] = {}
        for name, character in self.context.characters.items():
            variants = {name.casefold()}
            for alias in character.aliases:
                if alias and alias_owners.get(alias.casefold()) == name:
                    variants.add(alias.casefold())
            patterns[name] = sorted(variant for variant in variants if len(variant.strip()) > 1)
        return patterns

    def _detect(self, content: str) -> set[str]:
        hits: set[str] = set()
        text = content.casefold()
        for name, variants in self._patterns.items():
            for variant in variants:
                if variant in text:
                    hits.add(name)
                    break
        return hits
