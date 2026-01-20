from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass
class StyleSample:
    """A representative text sample demonstrating the writing style."""

    source_chapter: int
    content: str  # 20-50字原文片段
    style_notes: str  # 为何典型


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


@dataclass
class ChapterSummary:
    chapter: int
    title: str
    pov: str
    beats: list[str]
    mood: str
    synopsis: str
    irreversible_flags: list[str] = field(default_factory=list)


@dataclass
class EventImpact:
    power_shifts: dict[str, str] = field(default_factory=dict)
    relation_changes: dict[str, str] = field(default_factory=dict)
    world_flags: list[str] = field(default_factory=list)


@dataclass
class PlotEvent:
    event_id: str
    chapter: int
    type: Literal["conflict", "reveal", "progress", "setback"]
    participants: list[str]
    summary: str
    impact: EventImpact
    is_irreversible: bool = False


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
        """Instantiate from legacy payloads supporting both target/targets."""
        raw_targets = payload.get("targets")
        if raw_targets is None:
            raw_target = payload.get("target")
            if isinstance(raw_target, str):
                raw_targets = [raw_target]
            elif isinstance(raw_target, list):
                raw_targets = [item for item in raw_target if isinstance(item, str)]
            else:
                raw_targets = []
        elif isinstance(raw_targets, str):
            raw_targets = [raw_targets]
        else:
            raw_targets = [item for item in raw_targets if isinstance(item, str)]
        return cls(
            targets=raw_targets,
            relation_type=payload.get("relation_type", "acquaintance"),
            sentiment=payload.get("sentiment", "neutral"),
            description=payload.get("description", ""),
            source=payload.get("source"),
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


@dataclass
class StoryState:
    current_arc: str
    world_tension: Literal["low", "medium", "high"]
    major_conflicts: list[str]
    time_constraints: list[str]
    unresolved_events: list[str]


@dataclass
class StoryContext:
    metadata: dict[str, Any] = field(default_factory=dict)
    chapter_window: list[ChapterSummary] = field(default_factory=list)
    events: list[PlotEvent] = field(default_factory=list)
    characters: dict[str, CharacterState] = field(default_factory=dict)
    story_state: StoryState | None = None
    writing_style: WritingStyle | None = None

    def for_prompt(self) -> dict[str, Any]:
        """Serialize the context into prompt-ready textual sections."""
        sections = {
            "world": self._render_world_section(),
            "characters": self._render_character_section(),
            "plot": self._render_plot_section(),
            "chapters": self._render_chapter_section(),
            "style": self._render_style_section(),
        }
        return {key: value for key, value in sections.items() if value}

    def to_dict(self) -> dict[str, Any]:
        """Convert the context into a JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StoryContext":
        """Instantiate a StoryContext from a dictionary payload."""
        chapter_window = [ChapterSummary(**item) for item in payload.get("chapter_window", [])]
        events = []
        for item in payload.get("events", []):
            impact_payload = item.get("impact", {})
            impact = EventImpact(**impact_payload)
            events.append(
                PlotEvent(
                    event_id=item.get("event_id", ""),
                    chapter=item.get("chapter", 0),
                    type=item.get("type", "progress"),
                    participants=item.get("participants", []),
                    summary=item.get("summary", ""),
                    impact=impact,
                    is_irreversible=item.get("is_irreversible", False),
                )
            )
        characters = {}
        for name, value in payload.get("characters", {}).items():
            relationships = [Relationship.from_dict(rel) for rel in value.get("relationships", [])]
            characters[name] = CharacterState(
                name=value.get("name", name),
                aliases=value.get("aliases", []),
                realm=value.get("realm"),
                role=value.get("role", "minor"),
                personality=value.get("personality", []),
                relationships=relationships,
                unresolved=value.get("unresolved", []),
            )
        story_state_payload = payload.get("story_state")
        story_state = None
        if story_state_payload:
            story_state = StoryState(**story_state_payload)
        writing_style_payload = payload.get("writing_style")
        writing_style = None
        if writing_style_payload:
            samples = [
                StyleSample(**s) for s in writing_style_payload.get("representative_samples", [])
            ]
            writing_style = WritingStyle(
                avg_sentence_length=writing_style_payload.get("avg_sentence_length", 20),
                sentence_variety=writing_style_payload.get("sentence_variety", "mixed"),
                paragraph_density=writing_style_payload.get("paragraph_density", "medium"),
                register=writing_style_payload.get("register", "literary"),
                characteristic_words=writing_style_payload.get("characteristic_words", []),
                idiom_frequency=writing_style_payload.get("idiom_frequency", "sparse"),
                metaphor_style=writing_style_payload.get("metaphor_style", ""),
                description_focus=writing_style_payload.get("description_focus", []),
                parallelism_use=writing_style_payload.get("parallelism_use", "rare"),
                tone_markers=writing_style_payload.get("tone_markers", []),
                narrator_style=writing_style_payload.get("narrator_style", "detached"),
                representative_samples=samples,
                style_summary=writing_style_payload.get("style_summary", ""),
            )
        return cls(
            metadata=payload.get("metadata", {}),
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
    def _render_world_section(self) -> str:
        if not self.story_state:
            return ""
        lines = [
            f"Arc: {self.story_state.current_arc}",
            f"Tension: {self.story_state.world_tension}",
        ]
        if self.story_state.major_conflicts:
            lines.append("Major conflicts: " + "; ".join(self.story_state.major_conflicts[-5:]))
        if self.story_state.time_constraints:
            lines.append("Time constraints: " + "; ".join(self.story_state.time_constraints[-3:]))
        if self.story_state.unresolved_events:
            lines.append("World-level unresolved: " + "; ".join(self.story_state.unresolved_events[-5:]))
        return "\n".join(lines)

    def _render_character_section(self) -> str:
        if not self.characters:
            return ""
        role_order = {"main": 0, "support": 1, "minor": 2}
        ordered = sorted(
            self.characters.values(),
            key=lambda char: (role_order.get(char.role, 3), char.name.lower()),
        )
        lines: list[str] = []
        for character in ordered[:6]:
            traits = ", ".join(character.personality[:3]) if character.personality else "traits unknown"
            unresolved = ", ".join(character.unresolved[:2]) if character.unresolved else ""
            # Include aliases so LLM knows the name mappings
            aliases_part = f" (别名: {', '.join(character.aliases[:3])})" if character.aliases else ""
            suffix = f" | unresolved: {unresolved}" if unresolved else ""
            lines.append(f"- {character.name}{aliases_part} ({character.role}): {traits}{suffix}")
        return "\n".join(lines)

    def _render_plot_section(self) -> str:
        if not self.events:
            return ""
        lines: list[str] = []
        for event in self.events[-5:]:
            scope = f"[CH{event.chapter:03d}]" if event.chapter else "[??]"
            flag = " [irreversible]" if event.is_irreversible else ""
            participants = ", ".join(event.participants[:3]) if event.participants else "unknown actors"
            lines.append(f"{scope} {event.type}: {event.summary} ({participants}){flag}")
        return "\n".join(lines)

    def _render_chapter_section(self) -> str:
        if not self.chapter_window:
            return ""
        lines = []
        for summary in self.chapter_window[-5:]:
            flags = f" | flags: {', '.join(summary.irreversible_flags)}" if summary.irreversible_flags else ""
            lines.append(f"Chapter {summary.chapter} - {summary.title}: {summary.synopsis}{flags}")
        return "\n".join(lines)

    def _render_style_section(self) -> str:
        if not self.writing_style:
            return ""
        ws = self.writing_style
        lines = [
            f"风格概述: {ws.style_summary}" if ws.style_summary else "",
            f"句式: 平均{ws.avg_sentence_length}字, {ws.sentence_variety}变化",
            f"用词: {ws.register}风格",
        ]
        if ws.characteristic_words:
            lines.append(f"特征词: {', '.join(ws.characteristic_words[:5])}")
        if ws.tone_markers:
            lines.append(f"语气词: {', '.join(ws.tone_markers[:5])}")
        return "\n".join(line for line in lines if line)
