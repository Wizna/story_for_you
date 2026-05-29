from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import Any, Iterable, TYPE_CHECKING

from story_for_you.analysis.context import (
    CharacterState,
    PlotEvent,
    Relationship,
    StoryState,
)
from story_for_you.core.exceptions import LLMResponseError

if TYPE_CHECKING:
    from story_for_you.config.settings import RenderingLimits

_ROLE_PRIORITY: dict[str, int] = {"main": 3, "support": 2, "minor": 1}


class StateStore:
    """Aggregates character and world state over time."""

    def __init__(self) -> None:
        self._characters: dict[str, CharacterState] = {}
        self._story_state: StoryState | None = None
        self._event_log: list[PlotEvent] = []
        self._alias_index: dict[str, str] = {}

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

    def get_prompt_sections(self, limits: RenderingLimits | None = None) -> dict[str, str]:
        """Render character/world state summaries for downstream prompts."""
        if limits is None:
            from story_for_you.config.settings import RenderingLimits
            limits = RenderingLimits()
        world_lines = self._render_world_state(limits)
        character_lines = self._render_character_state(limits)
        unresolved_lines = self._render_unresolved_threads(limits)
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
        self._alias_index.clear()

    # Internal helpers -------------------------------------------------
    def _merge_character(self, character: CharacterState) -> None:
        owner_name = self._resolve_owner(character)
        existing = self._characters.get(owner_name or character.name)
        if not existing:
            self._characters[character.name] = deepcopy(character)
            self._register_aliases(character.name, [character.name, *character.aliases])
            return
        if character.aliases:
            existing.aliases = sorted(set(existing.aliases + character.aliases))
        if character.name != existing.name and character.name not in existing.aliases:
            existing.aliases = sorted(set(existing.aliases + [character.name]))
        if character.personality:
            existing.personality = list(dict.fromkeys(existing.personality + character.personality))
        if character.unresolved:
            existing.unresolved = list(dict.fromkeys(existing.unresolved + character.unresolved))
        if _ROLE_PRIORITY.get(character.role, 0) > _ROLE_PRIORITY.get(existing.role, 0):
            existing.role = character.role
        existing.realm = existing.realm or character.realm
        self._register_aliases(existing.name, [existing.name, *existing.aliases, character.name])

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

    def _render_world_state(self, limits: RenderingLimits) -> list[str]:
        if not self._story_state:
            return []
        lines = [
            f"Arc: {self._story_state.current_arc} | Tension: {self._story_state.world_tension}",
        ]
        if self._story_state.major_conflicts:
            lines.append("Major conflicts: " + "; ".join(self._story_state.major_conflicts[-limits.max_major_conflicts:]))
        if self._story_state.time_constraints:
            lines.append("Time constraints: " + "; ".join(self._story_state.time_constraints[-limits.max_time_constraints:]))
        return lines

    def _render_character_state(self, limits: RenderingLimits) -> list[str]:
        if not self._characters:
            return []
        ordered = sorted(
            self._characters.values(),
            key=lambda item: (_ROLE_PRIORITY.get(item.role, 0), item.name.lower()),
            reverse=True,
        )
        lines: list[str] = []
        for character in ordered[:limits.max_characters]:
            traits = ", ".join(character.personality[:limits.max_personality_traits]) if character.personality else "traits unknown"
            realm = character.realm or "unaffiliated"
            lines.append(f"- {character.name} ({character.role}, {realm}): {traits}")
        return lines

    def _render_unresolved_threads(self, limits: RenderingLimits) -> list[str]:
        unresolved: list[str] = []
        for character in self._characters.values():
            if character.unresolved:
                unresolved.append(f"{character.name}: {', '.join(character.unresolved[:limits.max_unresolved_per_char])}")
        return unresolved[:limits.max_total_unresolved_threads]

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
        if not isinstance(payload, dict):
            raise LLMResponseError("StateStore payload must be a JSON object.")
        instance = cls()
        for field_name in ("characters", "story_state", "event_log"):
            if field_name not in payload:
                raise LLMResponseError(f"StateStore missing required field: {field_name}")
        characters_payload = payload.get("characters")
        if not isinstance(characters_payload, dict):
            raise LLMResponseError("StateStore.characters must be an object.")
        events_payload = payload.get("event_log")
        if not isinstance(events_payload, list):
            raise LLMResponseError("StateStore.event_log must be a list.")
        for name, char_data in characters_payload.items():
            if not isinstance(name, str):
                raise LLMResponseError("StateStore character keys must be strings.")
            character = CharacterState.from_dict(char_data, name_hint=name)
            instance._characters[name] = character
            instance._register_aliases(character.name, [character.name, *character.aliases])
        story_state_data = payload.get("story_state")
        if story_state_data is not None:
            instance._story_state = StoryState.from_dict(story_state_data)
        for event_data in events_payload:
            instance._event_log.append(PlotEvent.from_dict(event_data))
        return instance

    def _resolve_owner(self, character: CharacterState) -> str | None:
        for key in self._alias_keys(character):
            owner = self._alias_index.get(key)
            if owner:
                return owner

        return None

    def _register_aliases(self, canonical_name: str, labels: Iterable[str]) -> None:
        for label in labels:
            for token in self._tokenize_label(label):
                key = self._normalize_token(token)
                if key:
                    self._alias_index[key] = canonical_name

    def _alias_keys(self, character: CharacterState) -> set[str]:
        keys: set[str] = set()
        for label in [character.name, *character.aliases]:
            for token in self._tokenize_label(label):
                normalized = self._normalize_token(token)
                if normalized:
                    keys.add(normalized)
        return keys

    def _tokenize_label(self, label: str) -> list[str]:
        cleaned = (label or "").strip()
        if not cleaned:
            return []
        return [cleaned]

    def _normalize_token(self, token: str) -> str:
        return token.strip().lower()
