from __future__ import annotations

from copy import deepcopy
from typing import Iterable

from story_for_you.analysis.context import CharacterState, PlotEvent, Relationship, StoryState


class StateStore:
        """Aggregates character and world state over time."""

    def __init__(self) -> None:
        self._characters: dict[str, CharacterState] = {}
        self._story_state: StoryState | None = None
        self._event_log: list[PlotEvent] = []

    def update(
        self,
        characters: Iterable[CharacterState],
        relationships: Iterable[Relationship],
        events: Iterable[PlotEvent],
    ) -> None:
        for character in characters:
            self._merge_character(character)
        for relationship in relationships:
            self._merge_relationship(relationship)
        for event in events:
            self._event_log.append(event)

    def set_story_state(self, story_state: StoryState) -> None:
        """Persist the latest synthesized story state."""
        self._story_state = story_state

    def characters_snapshot(self) -> dict[str, CharacterState]:
        """Return a shallow copy of known characters."""
        return {name: deepcopy(character) for name, character in self._characters.items()}

    def story_snapshot(self) -> StoryState | None:
        """Return the latest synthesized story state."""
        return deepcopy(self._story_state) if self._story_state else None

    def clear(self) -> None:
        """Reset the state store."""
        self._characters.clear()
        self._story_state = None
        self._event_log.clear()

    # Internal helpers -------------------------------------------------
    def _merge_character(self, character: CharacterState) -> None:
        existing = self._characters.get(character.name)
        if not existing:
            self._characters[character.name] = deepcopy(character)
            return
        if character.aliases:
            existing.aliases = sorted(set(existing.aliases + character.aliases))
        if character.personality:
            existing.personality = list(dict.fromkeys(existing.personality + character.personality))
        if character.unresolved:
            existing.unresolved = list(dict.fromkeys(existing.unresolved + character.unresolved))
        existing.role = character.role or existing.role
        existing.realm = character.realm or existing.realm

    def _merge_relationship(self, relationship: Relationship) -> None:
        if not relationship.source or relationship.source not in self._characters:
            return
        owner = self._characters[relationship.source]
        for existing in owner.relationships:
            if existing.target == relationship.target:
                existing.relation_type = relationship.relation_type
                existing.sentiment = relationship.sentiment
                existing.description = relationship.description or existing.description
                return
        owner.relationships.append(relationship)
