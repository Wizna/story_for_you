from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import Any, Iterable

from story_for_you.analysis.context import (
    CharacterState,
    EventImpact,
    PlotEvent,
    Relationship,
    StoryState,
)


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
        if not relationship.targets:
            return
        owner = self._characters[relationship.source]
        normalized_targets = self._normalize_targets(relationship.targets)
        for existing in owner.relationships:
            if existing.targets == normalized_targets:
                existing.relation_type = relationship.relation_type
                existing.sentiment = relationship.sentiment
                if relationship.description:
                    existing.description = relationship.description
                return
        owner.relationships.append(
            Relationship(
                targets=normalized_targets,
                relation_type=relationship.relation_type,
                sentiment=relationship.sentiment,
                description=relationship.description,
                source=relationship.source,
            )
        )

    def _normalize_targets(self, targets: Iterable[str]) -> list[str]:
        """Return a deterministic, deduplicated target list."""
        cleaned = [target.strip() for target in targets if target]
        return sorted(dict.fromkeys(cleaned))

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

    def to_dict(self) -> dict[str, Any]:
        """Serialize the store state to a dictionary."""
        return {
            "characters": {name: asdict(c) for name, c in self._characters.items()},
            "story_state": asdict(self._story_state) if self._story_state else None,
            "event_log": [asdict(e) for e in self._event_log],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> StateStore:
        """Restore store state from a dictionary."""
        instance = cls()
        for name, char_data in payload.get("characters", {}).items():
            relationships = [Relationship.from_dict(r) for r in char_data.get("relationships", [])]
            character = CharacterState(
                name=char_data.get("name", name),
                aliases=char_data.get("aliases", []),
                realm=char_data.get("realm"),
                role=char_data.get("role", "minor"),
                personality=char_data.get("personality", []),
                relationships=relationships,
                unresolved=char_data.get("unresolved", []),
            )
            instance._characters[name] = character
        story_state_data = payload.get("story_state")
        if story_state_data:
            instance._story_state = StoryState(**story_state_data)
        for event_data in payload.get("event_log", []):
            impact_data = event_data.get("impact", {})
            impact = EventImpact(**impact_data)
            event = PlotEvent(
                event_id=event_data.get("event_id", ""),
                chapter=event_data.get("chapter", 0),
                type=event_data.get("type", "progress"),
                participants=event_data.get("participants", []),
                summary=event_data.get("summary", ""),
                impact=impact,
                is_irreversible=event_data.get("is_irreversible", False),
            )
            instance._event_log.append(event)
        return instance
