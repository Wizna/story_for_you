from __future__ import annotations

import re
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
from story_for_you.utils.chinese_name_utils import split_compound_chinese_name


class StateStore:
    """Aggregates character and world state over time."""

    ROLE_PRIORITY = {"main": 3, "support": 2, "minor": 1}

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
        if self.ROLE_PRIORITY.get(character.role, 0) > self.ROLE_PRIORITY.get(existing.role, 0):
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
            instance._register_aliases(character.name, [character.name, *character.aliases])
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

    def _resolve_owner(self, character: CharacterState) -> str | None:
        # 第一步：精确 token 匹配
        for key in self._alias_keys(character):
            owner = self._alias_index.get(key)
            if owner:
                return owner

        # 第二步：子串匹配（针对中文名）
        candidate_names = [character.name] + character.aliases
        for existing_name in self._characters.keys():
            existing_char = self._characters[existing_name]
            existing_names = [existing_char.name] + existing_char.aliases
            if self._names_have_overlap(candidate_names, existing_names):
                return existing_name

        return None

    def _names_have_overlap(self, names1: list[str], names2: list[str]) -> bool:
        """检测两组名字是否有实质性重叠（子串匹配）。

        对于中文名，如果一个名字是另一个的子串且长度>=2，视为匹配。
        例如：傩送 ⊂ 傩送二老 → 匹配
        """
        for n1 in names1:
            n1_clean = self._normalize_token(n1)
            if len(n1_clean) < 2:
                continue
            for n2 in names2:
                n2_clean = self._normalize_token(n2)
                if len(n2_clean) < 2:
                    continue
                # 子串匹配：较短的名字是较长名字的子串
                shorter, longer = (n1_clean, n2_clean) if len(n1_clean) <= len(n2_clean) else (n2_clean, n1_clean)
                # 要求子串长度至少2个字符，且占较短名字的大部分
                if len(shorter) >= 2 and shorter in longer:
                    # 避免过于宽松的匹配（如"老"匹配一切含"老"的名字）
                    if len(shorter) >= 2 and (len(shorter) >= len(longer) * 0.5 or len(shorter) >= 2 and len(longer) <= 4):
                        return True
        return False

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
        without_paren = re.sub(r"[()（）]", " ", cleaned)
        pieces = re.split(r"[、,，/|]+", without_paren)
        tokens = [cleaned]
        tokens.extend(part.strip() for part in pieces if part and part.strip())

        # 对每个 token 尝试拆分复合中文名
        expanded: list[str] = []
        for token in tokens:
            expanded.append(token)
            expanded.extend(split_compound_chinese_name(token))

        # 去重并保持顺序
        seen: set[str] = set()
        result: list[str] = []
        for item in expanded:
            if item and item not in seen:
                seen.add(item)
                result.append(item)
        return result

    def _normalize_token(self, token: str) -> str:
        return re.sub(r"\s+", "", token).lower()
