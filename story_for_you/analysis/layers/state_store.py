from __future__ import annotations

from copy import deepcopy
from typing import Iterable

from story_for_you.analysis.context import CharacterState, PlotEvent, Relationship, StoryState


class StateStore:
    """Aggregates character and world state over time."""

    ROLE_PRIORITY = {"main": 3, "support": 2, "minor": 1}

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

    def get_prompt_sections(self) -> dict[str, str]:
        """Render character/world state summaries for downstream prompts."""
        world_lines = self._render_world_state()
        character_lines = self._render_character_state()
        unresolved_lines = self._render_unresolved_threads()
        sections: dict[str, str] = {}
        if world_lines:
            sections["world"] = "\n".join(world_lines)
        if character_lines:
            sections["characters"] = "\n".join(character_lines)
        if unresolved_lines:
            sections["unresolved"] = "\n".join(unresolved_lines)
        return sections

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
        if self.ROLE_PRIORITY.get(character.role, 0) > self.ROLE_PRIORITY.get(existing.role, 0):
            existing.role = character.role
        existing.realm = existing.realm or character.realm

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

    def _render_world_state(self) -> list[str]:
        if not self._story_state:
            return []
        lines = [
            f"Arc: {self._story_state.current_arc} | Tension: {self._story_state.world_tension}",
        ]
        if self._story_state.major_conflicts:
            lines.append("Major conflicts: " + "; ".join(self._story_state.major_conflicts[-5:]))
        if self._story_state.time_constraints:
            lines.append("Time constraints: " + "; ".join(self._story_state.time_constraints[-3:]))
        return lines

    def _render_character_state(self) -> list[str]:
        if not self._characters:
            return []
        ordered = sorted(
            self._characters.values(),
            key=lambda item: (self.ROLE_PRIORITY.get(item.role, 0), item.name.lower()),
            reverse=True,
        )
        lines: list[str] = []
        for character in ordered[:6]:
            traits = ", ".join(character.personality[:3]) if character.personality else "traits unknown"
            realm = character.realm or "unaffiliated"
            lines.append(f"- {character.name} ({character.role}, {realm}): {traits}")
        return lines

    def _render_unresolved_threads(self) -> list[str]:
        unresolved: list[str] = []
        for character in self._characters.values():
            if character.unresolved:
                unresolved.append(f"{character.name}: {', '.join(character.unresolved[:2])}")
        return unresolved[:5]
