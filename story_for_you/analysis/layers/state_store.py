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
from story_for_you.analysis.names import normalize_character_label
from story_for_you.analysis.threads import StoryThread, thread_id
from story_for_you.core.exceptions import LLMResponseError

if TYPE_CHECKING:
    from story_for_you.config.settings import RenderingLimits

_ROLE_PRIORITY: dict[str, int] = {"main": 3, "support": 2, "minor": 1}
_GENERIC_CHARACTER_TITLES = {
    "师父", "师傅", "老师", "师兄", "师姐", "师弟", "师妹", "大哥", "二哥",
    "三哥", "大嫂", "二嫂", "爹", "娘", "父亲", "母亲", "老爷", "夫人",
    "公子", "小姐", "掌门", "长老", "殿下", "陛下", "将军", "先生",
}


class StateStore:
    """Aggregates character and world state over time."""

    def __init__(self) -> None:
        self._characters: dict[str, CharacterState] = {}
        self._story_state: StoryState | None = None
        self._event_log: list[PlotEvent] = []
        self._alias_index: dict[str, str] = {}
        # A generic title such as “师父” may belong to several characters.
        # Ambiguous aliases must never silently choose whichever character was
        # registered last.
        self._ambiguous_aliases: set[str] = set()

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
            for name in event.deceased_characters:
                owner = self._resolve_label(name)
                if owner in self._characters:
                    self._characters[owner].status = "dead"

    def set_story_state(self, story_state: StoryState) -> None:
        """Persist the latest synthesized story state."""
        prior = self._story_state
        existing = CharacterState(
            name="世界", unresolved=list(prior.unresolved_events) if prior else [],
            threads=deepcopy(prior.threads) if prior else [],
        )
        incoming = CharacterState(
            name="世界", unresolved=story_state.unresolved_events,
            threads=story_state.threads, thread_updates=story_state.thread_updates,
        )
        self._merge_threads(existing, incoming)
        updated = deepcopy(story_state)
        updated.threads = existing.threads
        updated.unresolved_events = existing.unresolved
        updated.thread_updates = []
        self._story_state = updated

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
        self._ambiguous_aliases.clear()

    # Internal helpers -------------------------------------------------
    def _merge_character(self, character: CharacterState) -> None:
        owner_name = self._resolve_owner(character)
        existing = self._characters.get(owner_name or character.name)
        if not existing:
            existing = deepcopy(character)
            existing.threads = []
            existing.unresolved = []
            self._merge_threads(existing, character)
            self._apply_state_changes(existing, character)
            self._characters[character.name] = existing
            self._register_aliases(character.name, [character.name, *character.aliases])
            return
        if character.aliases:
            existing.aliases = sorted(set(existing.aliases + character.aliases))
        if character.name != existing.name and character.name not in existing.aliases:
            existing.aliases = sorted(set(existing.aliases + [character.name]))
        if character.personality:
            existing.personality = list(dict.fromkeys(existing.personality + character.personality))
        self._merge_threads(existing, character)
        if _ROLE_PRIORITY.get(character.role, 0) > _ROLE_PRIORITY.get(existing.role, 0):
            existing.role = character.role
        # Character extraction is chapter-local.  A supplied affiliation is a
        # newer observation and may reflect a defection, disguise, or promotion.
        changed_fields = {change.field for change in character.state_changes}
        self._apply_state_changes(existing, character)
        if character.realm and "realm" not in changed_fields:
            existing.realm = character.realm
        if character.status != "unknown":
            existing.status = character.status
        if character.location and "location" not in changed_fields:
            existing.location = character.location
        if character.goal and "goal" not in changed_fields:
            existing.goal = character.goal
        if character.knowledge and "knowledge" not in changed_fields:
            existing.knowledge = list(dict.fromkeys(existing.knowledge + character.knowledge))
        self._register_aliases(existing.name, [existing.name, *existing.aliases, character.name])

    def _apply_state_changes(self, existing: CharacterState, incoming: CharacterState) -> None:
        candidate = deepcopy(existing)
        for change in incoming.state_changes:
            if change.field == "knowledge":
                if change.operation in {"remove", "replace"}:
                    if change.previous not in candidate.knowledge:
                        raise LLMResponseError(f"Unknown previous knowledge for {existing.name}: {change.previous}")
                    candidate.knowledge.remove(change.previous)
                if change.operation in {"add", "replace"} and change.value not in candidate.knowledge:
                    candidate.knowledge.append(change.value)
            else:
                if change.previous is not None and getattr(candidate, change.field) != change.previous:
                    raise LLMResponseError(
                        f"Stale {change.field} update for {existing.name}: "
                        f"expected previous={getattr(candidate, change.field)!r}, received {change.previous!r}."
                    )
                setattr(candidate, change.field, change.value if change.operation == "set" else None)
            candidate.state_history.append(deepcopy(change))
        for field in ("goal", "location", "realm", "knowledge", "state_history"):
            setattr(existing, field, getattr(candidate, field))
        existing.state_changes = []

    def _merge_threads(self, existing: CharacterState, incoming: CharacterState) -> None:
        records = {item.thread_id: deepcopy(item) for item in existing.threads}
        for summary in existing.unresolved:
            if any(item.summary == summary for item in records.values()):
                continue
            key = thread_id(existing.name, summary)
            records.setdefault(key, StoryThread(key, summary))
        for record in incoming.threads:
            records[record.thread_id] = deepcopy(record)
        previous_summaries = {item.summary for item in records.values()}
        for change in incoming.thread_updates:
            key = change.thread_id or thread_id(existing.name, change.summary)
            if change.operation != "add" and key not in records:
                raise LLMResponseError(
                    f"Unknown thread ID for {existing.name}: {key}; known IDs: "
                    + "; ".join(f"{item.thread_id}={item.summary}" for item in records.values())
                )
            if change.operation == "add" and key in records:
                # Repeated observations cannot reopen an already resolved thread.
                continue
            records[key] = StoryThread(
                key, change.summary,
                "resolved" if change.operation == "resolve" else "open", change.evidence,
            )
        known_summaries = previous_summaries | {item.summary for item in records.values()}
        for summary in incoming.unresolved:
            if summary not in known_summaries:
                key = thread_id(existing.name, summary)
                records.setdefault(key, StoryThread(key, summary))
        existing.threads = list(records.values())
        existing.unresolved = [item.summary for item in records.values() if item.status == "open"]
        existing.thread_updates = []

    def _merge_relationship(self, relationship: Relationship) -> None:
        source = self._resolve_label(relationship.source)
        if not source or source not in self._characters:
            return
        if not relationship.targets:
            return
        owner = self._characters[source]
        normalized_targets = self._normalize_targets(relationship.targets)
        for existing in owner.relationships:
            if existing.targets == normalized_targets:
                existing.relation_type = relationship.relation_type
                existing.sentiment = relationship.sentiment
                if relationship.description:
                    existing.description = relationship.description
                if relationship.confidence:
                    existing.confidence = relationship.confidence
                if relationship.evidence:
                    existing.evidence = relationship.evidence
                if relationship.chapter is not None:
                    existing.chapter = relationship.chapter
                return
        owner.relationships.append(
            Relationship(
                targets=normalized_targets,
                relation_type=relationship.relation_type,
                sentiment=relationship.sentiment,
                description=relationship.description,
                source=source,
                confidence=relationship.confidence,
                evidence=relationship.evidence,
                chapter=relationship.chapter,
            )
        )

    def _normalize_targets(self, targets: Iterable[str]) -> list[str]:
        """Return a deterministic, deduplicated target list."""
        cleaned = [self._resolve_label(target) or target.strip() for target in targets if target]
        return sorted(dict.fromkeys(cleaned))

    def _resolve_label(self, label: str | None) -> str | None:
        """Resolve a canonical character name from a name or known alias."""
        if not label or not label.strip():
            return None
        cleaned = label.strip()
        key = self._normalize_token(cleaned)
        if key in self._ambiguous_aliases:
            return None
        return self._alias_index.get(key, cleaned)

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
            instance._merge_threads(character, CharacterState(name=character.name))
            instance._characters[name] = character
            instance._register_aliases(character.name, [character.name, *character.aliases])
        story_state_data = payload.get("story_state")
        if story_state_data is not None:
            instance.set_story_state(StoryState.from_dict(story_state_data))
        for event_data in events_payload:
            instance._event_log.append(PlotEvent.from_dict(event_data))
        return instance

    def _resolve_owner(self, character: CharacterState) -> str | None:
        canonical_key = self._normalize_token(character.name)
        direct_owner = self._alias_index.get(canonical_key)
        if direct_owner and canonical_key not in _GENERIC_CHARACTER_TITLES:
            return direct_owner
        for key in self._alias_keys(character):
            if key == canonical_key:
                continue
            if key in self._ambiguous_aliases:
                continue
            if key in {_normalize for _normalize in (_GENERIC_CHARACTER_TITLES)}:
                continue
            owner = self._alias_index.get(key)
            if owner:
                return owner

        return None

    def _register_aliases(self, canonical_name: str, labels: Iterable[str]) -> None:
        for label in labels:
            for token in self._tokenize_label(label):
                key = self._normalize_token(token)
                if key:
                    previous = self._alias_index.get(key)
                    if previous and previous != canonical_name:
                        self._ambiguous_aliases.add(key)
                        self._alias_index.pop(key, None)
                    elif key not in self._ambiguous_aliases:
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
        return normalize_character_label(token)
