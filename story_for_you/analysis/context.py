from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, TYPE_CHECKING

from story_for_you.exceptions import LLMResponseError

if TYPE_CHECKING:
    from story_for_you.config.settings import RenderingLimits

_VALID_EVENT_TYPES = {"conflict", "reveal", "progress", "setback"}
_VALID_RELATION_SENTIMENTS = {"positive", "neutral", "negative"}
_VALID_CHARACTER_ROLES = {"main", "support", "minor"}
_VALID_STORY_ARCS = {"setup", "journey", "twist", "climax", "dark-night", "resolution"}
_VALID_WORLD_TENSIONS = {"low", "medium", "high"}


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LLMResponseError(f"{label} must be a JSON object.")
    return value


def _require_fields(payload: dict[str, Any], fields: tuple[str, ...], label: str) -> None:
    for field_name in fields:
        if field_name not in payload:
            raise LLMResponseError(f"{label} missing required field: {field_name}")


def _required_str(payload: dict[str, Any], field_name: str, label: str, *, allow_empty: bool = False) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str):
        raise LLMResponseError(f"{label}.{field_name} must be a string.")
    text = value.strip()
    if not allow_empty and not text:
        raise LLMResponseError(f"{label}.{field_name} must not be empty.")
    return text


def _optional_str(payload: dict[str, Any], field_name: str, label: str) -> str | None:
    value = payload.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise LLMResponseError(f"{label}.{field_name} must be a string or null.")
    return value.strip() or None


def _required_int(payload: dict[str, Any], field_name: str, label: str) -> int:
    value = payload.get(field_name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise LLMResponseError(f"{label}.{field_name} must be an integer.")
    return value


def _required_bool(payload: dict[str, Any], field_name: str, label: str) -> bool:
    value = payload.get(field_name)
    if not isinstance(value, bool):
        raise LLMResponseError(f"{label}.{field_name} must be a boolean.")
    return value


def _required_str_list(payload: dict[str, Any], field_name: str, label: str) -> list[str]:
    value = payload.get(field_name)
    if not isinstance(value, list):
        raise LLMResponseError(f"{label}.{field_name} must be a list.")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise LLMResponseError(f"{label}.{field_name} items must be strings.")
        text = item.strip()
        if text:
            items.append(text)
    return items


def _required_str_dict(payload: dict[str, Any], field_name: str, label: str) -> dict[str, str]:
    value = payload.get(field_name)
    if not isinstance(value, dict):
        raise LLMResponseError(f"{label}.{field_name} must be an object.")
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise LLMResponseError(f"{label}.{field_name} keys and values must be strings.")
        if key.strip() and item.strip():
            result[key.strip()] = item.strip()
    return result


@dataclass
class StyleSample:
    """A representative text sample demonstrating the writing style."""

    source_chapter: int
    content: str  # 20-50字原文片段
    style_notes: str  # 为何典型

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StyleSample":
        payload = _require_mapping(payload, "StyleSample")
        _require_fields(payload, ("source_chapter", "content", "style_notes"), "StyleSample")
        return cls(
            source_chapter=_required_int(payload, "source_chapter", "StyleSample"),
            content=_required_str(payload, "content", "StyleSample", allow_empty=True),
            style_notes=_required_str(payload, "style_notes", "StyleSample", allow_empty=True),
        )


@dataclass
class WritingStyle:
    """Captured writing style characteristics for style-aware generation."""

    # 句式结构
    avg_sentence_length: int
    sentence_variety: str  # uniform | varied | mixed
    paragraph_density: str  # sparse | medium | dense

    # 用词风格
    register: str  # literary | colloquial | classical | mixed
    characteristic_words: list[str] = field(default_factory=list)  # 特征词汇（最多8个）
    idiom_frequency: str = "sparse"  # none | sparse | moderate | heavy

    # 修辞手法
    metaphor_style: str = ""
    description_focus: list[str] = field(default_factory=list)  # landscape, psychological, action
    parallelism_use: str = "rare"  # rare | occasional | frequent

    # 叙事语气
    tone_markers: list[str] = field(default_factory=list)  # 常用语气词
    narrator_style: str = "detached"  # detached | intimate | intrusive

    # 示例与摘要
    representative_samples: list[StyleSample] = field(default_factory=list)
    style_summary: str = ""  # 100-150字风格总结，用于提示词

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WritingStyle":
        payload = _require_mapping(payload, "WritingStyle")
        _require_fields(
            payload,
            (
                "avg_sentence_length",
                "sentence_variety",
                "paragraph_density",
                "register",
                "characteristic_words",
                "idiom_frequency",
                "metaphor_style",
                "description_focus",
                "parallelism_use",
                "tone_markers",
                "narrator_style",
                "representative_samples",
                "style_summary",
            ),
            "WritingStyle",
        )
        sample_payload = payload.get("representative_samples")
        if not isinstance(sample_payload, list):
            raise LLMResponseError("WritingStyle.representative_samples must be a list.")
        return cls(
            avg_sentence_length=_required_int(payload, "avg_sentence_length", "WritingStyle"),
            sentence_variety=_required_str(payload, "sentence_variety", "WritingStyle"),
            paragraph_density=_required_str(payload, "paragraph_density", "WritingStyle"),
            register=_required_str(payload, "register", "WritingStyle"),
            characteristic_words=_required_str_list(payload, "characteristic_words", "WritingStyle"),
            idiom_frequency=_required_str(payload, "idiom_frequency", "WritingStyle"),
            metaphor_style=_required_str(payload, "metaphor_style", "WritingStyle", allow_empty=True),
            description_focus=_required_str_list(payload, "description_focus", "WritingStyle"),
            parallelism_use=_required_str(payload, "parallelism_use", "WritingStyle"),
            tone_markers=_required_str_list(payload, "tone_markers", "WritingStyle"),
            narrator_style=_required_str(payload, "narrator_style", "WritingStyle"),
            representative_samples=[StyleSample.from_dict(item) for item in sample_payload],
            style_summary=_required_str(payload, "style_summary", "WritingStyle", allow_empty=True),
        )


@dataclass
class ChapterSummary:
    chapter: int
    title: str
    pov: str
    beats: list[str]
    mood: str
    synopsis: str
    irreversible_flags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChapterSummary":
        payload = _require_mapping(payload, "ChapterSummary")
        _require_fields(
            payload,
            ("chapter", "title", "pov", "beats", "mood", "synopsis", "irreversible_flags"),
            "ChapterSummary",
        )
        return cls(
            chapter=_required_int(payload, "chapter", "ChapterSummary"),
            title=_required_str(payload, "title", "ChapterSummary"),
            pov=_required_str(payload, "pov", "ChapterSummary"),
            beats=_required_str_list(payload, "beats", "ChapterSummary"),
            mood=_required_str(payload, "mood", "ChapterSummary"),
            synopsis=_required_str(payload, "synopsis", "ChapterSummary"),
            irreversible_flags=_required_str_list(payload, "irreversible_flags", "ChapterSummary"),
        )


@dataclass
class EventImpact:
    power_shifts: dict[str, str] = field(default_factory=dict)
    relation_changes: dict[str, str] = field(default_factory=dict)
    world_flags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EventImpact":
        payload = _require_mapping(payload, "EventImpact")
        _require_fields(payload, ("power_shifts", "relation_changes", "world_flags"), "EventImpact")
        return cls(
            power_shifts=_required_str_dict(payload, "power_shifts", "EventImpact"),
            relation_changes=_required_str_dict(payload, "relation_changes", "EventImpact"),
            world_flags=_required_str_list(payload, "world_flags", "EventImpact"),
        )


@dataclass
class PlotEvent:
    event_id: str
    chapter: int
    type: Literal["conflict", "reveal", "progress", "setback"]
    participants: list[str]
    summary: str
    impact: EventImpact
    is_irreversible: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PlotEvent":
        """Instantiate a PlotEvent from a dictionary payload."""
        payload = _require_mapping(payload, "PlotEvent")
        _require_fields(
            payload,
            ("event_id", "chapter", "type", "participants", "summary", "impact", "is_irreversible"),
            "PlotEvent",
        )
        event_type = _required_str(payload, "type", "PlotEvent")
        if event_type not in _VALID_EVENT_TYPES:
            raise LLMResponseError(f"Invalid PlotEvent.type: {event_type!r}")
        return cls(
            event_id=_required_str(payload, "event_id", "PlotEvent"),
            chapter=_required_int(payload, "chapter", "PlotEvent"),
            type=event_type,
            participants=_required_str_list(payload, "participants", "PlotEvent"),
            summary=_required_str(payload, "summary", "PlotEvent"),
            impact=EventImpact.from_dict(payload.get("impact")),
            is_irreversible=_required_bool(payload, "is_irreversible", "PlotEvent"),
        )


@dataclass
class Relationship:
    targets: list[str] = field(default_factory=list)
    relation_type: str = "acquaintance"
    sentiment: Literal["positive", "neutral", "negative"] = "neutral"
    description: str = ""
    source: str | None = None

    def __post_init__(self) -> None:
        """Ensure targets remain deterministic and deduplicated."""
        cleaned = [target.strip() for target in self.targets if target]
        self.targets = sorted(dict.fromkeys(cleaned))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Relationship":
        """Instantiate from a strict relationship payload."""
        payload = _require_mapping(payload, "Relationship")
        _require_fields(payload, ("targets", "relation_type", "sentiment", "description", "source"), "Relationship")
        sentiment = _required_str(payload, "sentiment", "Relationship")
        if sentiment not in _VALID_RELATION_SENTIMENTS:
            raise LLMResponseError(f"Invalid Relationship.sentiment: {sentiment!r}")
        return cls(
            targets=_required_str_list(payload, "targets", "Relationship"),
            relation_type=_required_str(payload, "relation_type", "Relationship"),
            sentiment=sentiment,
            description=_required_str(payload, "description", "Relationship", allow_empty=True),
            source=_required_str(payload, "source", "Relationship"),
        )


@dataclass
class CharacterState:
    name: str
    aliases: list[str] = field(default_factory=list)
    realm: str | None = None
    role: Literal["main", "support", "minor"] = "minor"
    personality: list[str] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any], name_hint: str = "") -> "CharacterState":
        """Instantiate a CharacterState from a dictionary payload."""
        payload = _require_mapping(payload, "CharacterState")
        _require_fields(
            payload,
            ("name", "aliases", "realm", "role", "personality", "relationships", "unresolved"),
            "CharacterState",
        )
        role = _required_str(payload, "role", "CharacterState")
        if role not in _VALID_CHARACTER_ROLES:
            raise LLMResponseError(f"Invalid CharacterState.role: {role!r}")
        relationship_payload = payload.get("relationships")
        if not isinstance(relationship_payload, list):
            raise LLMResponseError("CharacterState.relationships must be a list.")
        return cls(
            name=_required_str(payload, "name", "CharacterState") or name_hint,
            aliases=_required_str_list(payload, "aliases", "CharacterState"),
            realm=_optional_str(payload, "realm", "CharacterState"),
            role=role,
            personality=_required_str_list(payload, "personality", "CharacterState"),
            relationships=[Relationship.from_dict(r) for r in relationship_payload],
            unresolved=_required_str_list(payload, "unresolved", "CharacterState"),
        )


@dataclass
class StoryState:
    current_arc: str
    world_tension: Literal["low", "medium", "high"]
    major_conflicts: list[str]
    time_constraints: list[str]
    unresolved_events: list[str]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StoryState":
        payload = _require_mapping(payload, "StoryState")
        _require_fields(
            payload,
            ("current_arc", "world_tension", "major_conflicts", "time_constraints", "unresolved_events"),
            "StoryState",
        )
        current_arc = _required_str(payload, "current_arc", "StoryState")
        if current_arc not in _VALID_STORY_ARCS:
            raise LLMResponseError(f"Invalid StoryState.current_arc: {current_arc!r}")
        world_tension = _required_str(payload, "world_tension", "StoryState")
        if world_tension not in _VALID_WORLD_TENSIONS:
            raise LLMResponseError(f"Invalid StoryState.world_tension: {world_tension!r}")
        return cls(
            current_arc=current_arc,
            world_tension=world_tension,
            major_conflicts=_required_str_list(payload, "major_conflicts", "StoryState"),
            time_constraints=_required_str_list(payload, "time_constraints", "StoryState"),
            unresolved_events=_required_str_list(payload, "unresolved_events", "StoryState"),
        )


@dataclass
class StoryContext:
    metadata: dict[str, Any] = field(default_factory=dict)
    chapter_window: list[ChapterSummary] = field(default_factory=list)
    events: list[PlotEvent] = field(default_factory=list)
    characters: dict[str, CharacterState] = field(default_factory=dict)
    story_state: StoryState | None = None
    writing_style: WritingStyle | None = None

    def for_prompt(self, limits: RenderingLimits | None = None) -> dict[str, Any]:
        """Serialize the context into prompt-ready textual sections."""
        if limits is None:
            from story_for_you.config.settings import RenderingLimits
            limits = RenderingLimits()
        sections = {
            "world": self._render_world_section(limits),
            "characters": self._render_character_section(limits),
            "plot": self._render_plot_section(limits),
            "chapters": self._render_chapter_section(limits),
            "style": self._render_style_section(limits),
        }
        return {key: value for key, value in sections.items() if value}

    def to_dict(self) -> dict[str, Any]:
        """Convert the context into a JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StoryContext":
        """Instantiate a StoryContext from a dictionary payload."""
        payload = _require_mapping(payload, "StoryContext")
        _require_fields(
            payload,
            ("metadata", "chapter_window", "events", "characters", "story_state", "writing_style"),
            "StoryContext",
        )
        chapter_payload = payload.get("chapter_window")
        events_payload = payload.get("events")
        characters_payload = payload.get("characters")
        if not isinstance(chapter_payload, list):
            raise LLMResponseError("StoryContext.chapter_window must be a list.")
        if not isinstance(events_payload, list):
            raise LLMResponseError("StoryContext.events must be a list.")
        if not isinstance(characters_payload, dict):
            raise LLMResponseError("StoryContext.characters must be an object.")
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            raise LLMResponseError("StoryContext.metadata must be an object.")
        chapter_window = [ChapterSummary.from_dict(item) for item in chapter_payload]
        events = [PlotEvent.from_dict(item) for item in events_payload]
        characters = {
            name: CharacterState.from_dict(value, name_hint=name)
            for name, value in characters_payload.items()
        }
        story_state_payload = payload.get("story_state")
        story_state = StoryState.from_dict(story_state_payload) if story_state_payload is not None else None
        writing_style_payload = payload.get("writing_style")
        writing_style = WritingStyle.from_dict(writing_style_payload) if writing_style_payload is not None else None
        return cls(
            metadata=metadata,
            chapter_window=chapter_window,
            events=events,
            characters=characters,
            story_state=story_state,
            writing_style=writing_style,
        )

    def add_metadata(self, key: str, value: Any) -> None:
        """Record helper metadata for downstream consumers."""
        self.metadata[key] = value

    # Prompt helpers ---------------------------------------------------
    def _render_world_section(self, limits: RenderingLimits) -> str:
        if not self.story_state:
            return ""
        lines = [
            f"Arc: {self.story_state.current_arc}",
            f"Tension: {self.story_state.world_tension}",
        ]
        if self.story_state.major_conflicts:
            lines.append("Major conflicts: " + "; ".join(self.story_state.major_conflicts[-limits.max_major_conflicts:]))
        if self.story_state.time_constraints:
            lines.append("Time constraints: " + "; ".join(self.story_state.time_constraints[-limits.max_time_constraints:]))
        if self.story_state.unresolved_events:
            lines.append("World-level unresolved: " + "; ".join(self.story_state.unresolved_events[-limits.max_unresolved_events:]))
        return "\n".join(lines)

    def _render_character_section(self, limits: RenderingLimits) -> str:
        if not self.characters:
            return ""
        role_order = {"main": 0, "support": 1, "minor": 2}
        ordered = sorted(
            self.characters.values(),
            key=lambda char: (role_order.get(char.role, 3), char.name.lower()),
        )
        lines: list[str] = []
        for character in ordered[:limits.max_characters]:
            traits = ", ".join(character.personality[:limits.max_personality_traits]) if character.personality else "traits unknown"
            unresolved = ", ".join(character.unresolved[:limits.max_unresolved_per_char]) if character.unresolved else ""
            # Include aliases so LLM knows the name mappings
            aliases_part = f" (别名: {', '.join(character.aliases[:limits.max_aliases])})" if character.aliases else ""
            suffix = f" | unresolved: {unresolved}" if unresolved else ""
            lines.append(f"- {character.name}{aliases_part} ({character.role}): {traits}{suffix}")
        return "\n".join(lines)

    def _render_plot_section(self, limits: RenderingLimits) -> str:
        if not self.events:
            return ""
        lines: list[str] = []
        for event in self.events[-limits.max_events:]:
            scope = f"[CH{event.chapter:03d}]" if event.chapter else "[??]"
            flag = " [irreversible]" if event.is_irreversible else ""
            participants = ", ".join(event.participants[:limits.max_event_participants]) if event.participants else "unknown actors"
            lines.append(f"{scope} {event.type}: {event.summary} ({participants}){flag}")
        return "\n".join(lines)

    def _render_chapter_section(self, limits: RenderingLimits) -> str:
        if not self.chapter_window:
            return ""
        lines = []
        for summary in self.chapter_window[-limits.max_chapters:]:
            flags = f" | flags: {', '.join(summary.irreversible_flags)}" if summary.irreversible_flags else ""
            lines.append(f"Chapter {summary.chapter} - {summary.title}: {summary.synopsis}{flags}")
        return "\n".join(lines)

    def _render_style_section(self, limits: RenderingLimits) -> str:
        if not self.writing_style:
            return ""
        ws = self.writing_style
        lines = [
            f"风格概述: {ws.style_summary}" if ws.style_summary else "",
            f"句式: 平均{ws.avg_sentence_length}字, {ws.sentence_variety}变化",
            f"用词: {ws.register}风格",
        ]
        if ws.characteristic_words:
            lines.append(f"特征词: {', '.join(ws.characteristic_words[:limits.max_characteristic_words])}")
        if ws.tone_markers:
            lines.append(f"语气词: {', '.join(ws.tone_markers[:limits.max_tone_markers])}")
        return "\n".join(line for line in lines if line)
