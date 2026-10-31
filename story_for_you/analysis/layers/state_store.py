from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import Any, Iterable, TYPE_CHECKING

from story_for_you.analysis.context import (
    CharacterState,
    PlotEvent,
    Relationship,
    StoryState,
    _bounded_history,
)
from story_for_you.analysis.names import normalize_character_label
from story_for_you.analysis.threads import StoryThread, ThreadEvidence, thread_id
from story_for_you.core.exceptions import LLMResponseError

if TYPE_CHECKING:
    from story_for_you.config.settings import RenderingLimits

_ROLE_PRIORITY: dict[str, int] = {"main": 3, "support": 2, "minor": 1}
_GENERIC_CHARACTER_TITLES = {
    "师父", "师傅", "老师", "师兄", "师姐", "师弟", "师妹", "大哥", "二哥",
    "三哥", "大嫂", "二嫂", "爹", "娘", "父亲", "母亲", "老爷", "夫人",
    "公子", "小姐", "掌门", "长老", "殿下", "陛下", "将军", "先生",
}
_MAX_THREAD_EVIDENCE_HISTORY = 48
_MAX_RELATIONSHIP_HISTORY = 128


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
        *,
        chapter: int | None = None,
        source_unit: int | None = None,
    ) -> None:
        character_list = list(characters)
        for character in character_list:
            self._merge_character(character, chapter=chapter, source_unit=source_unit)
        for character in character_list:
            for relationship in character.relationships:
                self._merge_relationship(relationship, chapter=chapter, source_unit=source_unit)
        for relationship in relationships:
            self._merge_relationship(relationship, chapter=chapter, source_unit=source_unit)
        for event in events:
            event_record = deepcopy(event)
            if chapter is not None and event_record.chapter <= 0:
                event_record.chapter = chapter
            if source_unit is not None and event_record.source_unit is None:
                event_record.source_unit = source_unit
            self._event_log.append(event_record)
            # A non-irreversible event may mention a rumour, a false death, or
            # a witness's mistaken belief.  Only an explicitly irreversible
            # event is allowed to establish the canonical dead state; the
            # chapter-local character extractor still has to provide its own
            # quoted death evidence when it sets ``status=dead``.
            if not event_record.is_irreversible:
                continue
            for name in event_record.deceased_characters:
                owner = self._resolve_label(name)
                if owner in self._characters:
                    self._characters[owner].status = "dead"
                    if event_record.evidence:
                        self._characters[owner].status_evidence = event_record.evidence

    def set_story_state(
        self, story_state: StoryState, *, chapter: int | None = None, source_unit: int | None = None
    ) -> None:
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
        self._merge_threads(existing, incoming, chapter=chapter, source_unit=source_unit)
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
    def _merge_character(
        self, character: CharacterState, *, chapter: int | None = None, source_unit: int | None = None
    ) -> None:
        owner_name = self._resolve_owner(character)
        existing = self._characters.get(owner_name or character.name)
        if not existing:
            existing = deepcopy(character)
            # Relationships are merged below after every character has been
            # registered. Keep the initial snapshot empty so one observation
            # is not recorded during creation and again during that pass.
            existing.relationships = []
            existing.relationship_history = deepcopy(character.relationship_history)
            existing.incoming_relationship_history = deepcopy(character.incoming_relationship_history)
            existing.threads = []
            existing.unresolved = []
            self._merge_threads(existing, character, chapter=chapter, source_unit=source_unit)
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
        self._merge_threads(existing, character, chapter=chapter, source_unit=source_unit)
        for observation in character.relationship_history:
            self._append_relationship_history(existing, observation)
        for observation in character.incoming_relationship_history:
            self._append_incoming_relationship_history(existing, observation)
        if _ROLE_PRIORITY.get(character.role, 0) > _ROLE_PRIORITY.get(existing.role, 0):
            existing.role = character.role
        # Character extraction is chapter-local.  A supplied affiliation is a
        # newer observation and may reflect a defection, disguise, or promotion.
        changed_fields = {change.field for change in character.state_changes}
        self._apply_state_changes(existing, character)
        if character.realm and "realm" not in changed_fields:
            existing.realm = character.realm
        if character.status != "unknown":
            # Death is an irreversible fact. A later chapter-local extraction
            # may omit the death or hallucinate an ``alive`` observation, but
            # it must not resurrect a character already marked dead.
            if existing.status != "dead" or character.status == "dead":
                existing.status = character.status
        if character.status == "dead" and character.status_evidence:
            existing.status_evidence = character.status_evidence
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

    def _merge_threads(
        self,
        existing: CharacterState,
        incoming: CharacterState,
        *,
        chapter: int | None = None,
        source_unit: int | None = None,
    ) -> None:
        records = {item.thread_id: deepcopy(item) for item in existing.threads}
        for summary in existing.unresolved:
            if any(item.summary == summary for item in records.values()):
                continue
            key = thread_id(existing.name, summary)
            records.setdefault(key, StoryThread(key, summary))
        for record in incoming.threads:
            current = records.get(record.thread_id)
            if current is None:
                current = deepcopy(record)
                # Initial extractor records carry the latest evidence in the
                # flat ``evidence`` field but often have no history yet.  Move
                # that observation into the bounded history immediately so a
                # long-range continuation can retrieve the original clue.
                initial_history = list(current.history)
                current.history = []
                for observation in initial_history:
                    self._append_thread_observation(current, observation)
                self._append_thread_history(
                    current, current.summary, current.evidence, chapter, source_unit,
                )
                records[record.thread_id] = current
            else:
                current.summary = record.summary or current.summary
                if not (
                    current.status in {"resolved", "superseded"}
                    and record.status in {"open", "dormant"}
                ):
                    current.status = record.status
                current.evidence = record.evidence or current.evidence
                current.kind = record.kind or current.kind
                current.confidence = record.confidence or current.confidence
                current.candidate_outcomes = list(dict.fromkeys(
                    current.candidate_outcomes + record.candidate_outcomes
                ))
                for observation in record.history:
                    self._append_thread_observation(current, observation)
                self._append_thread_history(
                    current, current.summary, record.evidence, chapter, source_unit,
                )
        previous_summaries = {item.summary for item in records.values()}
        for change in incoming.thread_updates:
            key = change.thread_id or thread_id(existing.name, change.summary)
            if change.operation != "add" and key not in records:
                raise LLMResponseError(
                    f"Unknown thread ID for {existing.name}: {key}; known IDs: "
                    + "; ".join(f"{item.thread_id}={item.summary}" for item in records.values())
                )
            if change.operation == "add" and key in records:
                # Repeated observations cannot reopen an already resolved thread,
                # but their evidence is still useful for long-range retrieval.
                self._append_thread_history(records[key], change.summary, change.evidence, chapter, source_unit)
                continue
            prior = records.get(key)
            if prior is None:
                prior = StoryThread(key, change.summary)
            status = prior.status
            if change.operation == "resolve":
                status = "resolved"
            elif change.operation == "supersede":
                status = "superseded"
            elif change.operation in {"add", "advance"}:
                status = "open"
            # ``update`` preserves a resolved/superseded status. A later
            # observation must not silently reopen a settled thread.
            prior.summary = change.summary
            prior.status = status
            prior.evidence = change.evidence
            if change.kind != "unclassified":
                prior.kind = change.kind
            prior.confidence = change.confidence
            if change.candidate_outcomes:
                prior.candidate_outcomes = list(dict.fromkeys(
                    prior.candidate_outcomes + change.candidate_outcomes
                ))
            self._append_thread_history(prior, change.summary, change.evidence, chapter, source_unit)
            records[key] = prior
        known_summaries = previous_summaries | {item.summary for item in records.values()}
        for summary in incoming.unresolved:
            if summary not in known_summaries:
                key = thread_id(existing.name, summary)
                record = records.setdefault(key, StoryThread(key, summary))
                if record.evidence:
                    self._append_thread_history(record, summary, record.evidence, chapter, source_unit)
        existing.threads = list(records.values())
        existing.unresolved = [item.summary for item in records.values() if item.status == "open"]
        existing.thread_updates = []

    def _append_thread_history(
        self,
        thread: StoryThread,
        summary: str,
        evidence: str,
        chapter: int | None,
        source_unit: int | None,
    ) -> None:
        if not evidence.strip():
            return
        observation = ThreadEvidence(summary.strip(), evidence.strip(), chapter, source_unit)
        self._append_thread_observation(thread, observation)

    def _append_thread_observation(self, thread: StoryThread, observation: ThreadEvidence) -> None:
        """Merge one quoted observation while preserving its best provenance."""
        for index in range(len(thread.history) - 1, -1, -1):
            previous = thread.history[index]
            if previous.summary != observation.summary or previous.evidence != observation.evidence:
                continue
            # A later merge may know the physical unit even when the cached
            # record did not.  Upgrade that row rather than creating a
            # duplicate observation for the same quotation.
            previous_position = previous.source_unit or previous.chapter or -1
            observation_position = observation.source_unit or observation.chapter or -1
            if observation_position >= previous_position and (
                observation.chapter is not None or observation.source_unit is not None
            ):
                thread.history[index] = deepcopy(observation)
            return
        thread.history.append(deepcopy(observation))
        if len(thread.history) > _MAX_THREAD_EVIDENCE_HISTORY:
            thread.history = [thread.history[0], *thread.history[-(_MAX_THREAD_EVIDENCE_HISTORY - 1):]]

    def _merge_relationship(
        self, relationship: Relationship, *, chapter: int | None = None, source_unit: int | None = None
    ) -> None:
        source = self._resolve_label(relationship.source)
        if not source or source not in self._characters:
            return
        if not relationship.targets:
            return
        owner = self._characters[source]
        normalized_targets = self._normalize_targets(relationship.targets)
        observation = deepcopy(relationship)
        observation.source = source
        observation.targets = normalized_targets
        if chapter is not None:
            observation.chapter = chapter
        if source_unit is not None:
            observation.source_unit = source_unit
        self._append_relationship_history(owner, observation)
        for target in normalized_targets:
            target_owner = self._resolve_label(target)
            if not target_owner or target_owner not in self._characters or target_owner == source:
                continue
            incoming = deepcopy(observation)
            incoming.targets = [source]
            self._append_incoming_relationship_history(
                self._characters[target_owner], incoming
            )
        for existing in owner.relationships:
            if existing.targets == normalized_targets:
                existing.relation_type = observation.relation_type
                existing.sentiment = observation.sentiment
                if observation.description:
                    existing.description = observation.description
                if observation.confidence:
                    existing.confidence = observation.confidence
                if observation.evidence:
                    existing.evidence = observation.evidence
                if observation.chapter is not None:
                    existing.chapter = observation.chapter
                if observation.source_unit is not None:
                    existing.source_unit = observation.source_unit
                return
        owner.relationships.append(
            Relationship(
                targets=normalized_targets,
                relation_type=observation.relation_type,
                sentiment=observation.sentiment,
                description=observation.description,
                source=source,
                confidence=observation.confidence,
                evidence=observation.evidence,
                chapter=observation.chapter,
                source_unit=observation.source_unit,
            )
        )

    def _append_relationship_history(self, owner: CharacterState, observation: Relationship) -> None:
        # Long novels repeat the same relationship in many chapters. Keep the
        # latest evidence for an unchanged state, while retaining each actual
        # transition so an old alliance cannot hide a later betrayal.
        for index in range(len(owner.relationship_history) - 1, -1, -1):
            previous = owner.relationship_history[index]
            if previous.targets != observation.targets:
                continue
            if self._same_relationship_state(previous, observation):
                owner.relationship_history[index] = deepcopy(observation)
                return
            break
        owner.relationship_history.append(deepcopy(observation))
        if len(owner.relationship_history) > _MAX_RELATIONSHIP_HISTORY:
            owner.relationship_history = [
                owner.relationship_history[0],
                *owner.relationship_history[-(_MAX_RELATIONSHIP_HISTORY - 1):],
            ]

    def _append_incoming_relationship_history(
        self, owner: CharacterState, observation: Relationship
    ) -> None:
        for index in range(len(owner.incoming_relationship_history) - 1, -1, -1):
            previous = owner.incoming_relationship_history[index]
            if previous.source != observation.source or previous.targets != observation.targets:
                continue
            if self._same_relationship_state(previous, observation):
                owner.incoming_relationship_history[index] = deepcopy(observation)
                return
            break
        owner.incoming_relationship_history.append(deepcopy(observation))
        if len(owner.incoming_relationship_history) > _MAX_RELATIONSHIP_HISTORY:
            owner.incoming_relationship_history = [
                owner.incoming_relationship_history[0],
                *owner.incoming_relationship_history[-(_MAX_RELATIONSHIP_HISTORY - 1):],
            ]

    def _same_relationship_state(self, left: Relationship, right: Relationship) -> bool:
        """Compare semantic relationship state while ignoring provenance."""
        return (
            left.relation_type == right.relation_type
            and left.sentiment == right.sentiment
            and left.description == right.description
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
            state = (
                f"status={character.status}, location={character.location or 'unknown'}, "
                f"goal={character.goal or 'unknown'}"
            )
            if character.status == "dead" and character.status_evidence:
                state += f", death_evidence={character.status_evidence}"
            relation = ""
            if character.relationships:
                relation = " | " + "; ".join(
                    f"{','.join(item.targets)}:{item.relation_type}/{item.sentiment}"
                    for item in character.relationships[:limits.max_relationships_per_char]
                )
            history = ""
            if (
                (character.relationship_history or character.incoming_relationship_history)
                and limits.max_relationship_history > 0
            ):
                observations = [
                    *character.relationship_history,
                    *character.incoming_relationship_history,
                ]
                observations.sort(key=lambda item: item.source_unit or item.chapter or -1)
                observations = _bounded_history(observations, limits.max_relationship_history)
                history = " | historical_relationship_history (source observations): " + "; ".join(
                    (
                        f"来自{item.source}:" if item.source and item.source != character.name else ""
                    )
                    + f"{','.join(item.targets)}:{item.relation_type}/{item.sentiment}"
                    + (f"@{item.chapter}" if item.chapter is not None else "")
                    + (f"（证据：{item.evidence}）" if item.evidence else "")
                    for item in observations
                )
            lines.append(f"- {character.name} ({character.role}, {realm}; {state}): {traits}{relation}{history}")
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
