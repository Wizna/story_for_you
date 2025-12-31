from dataclasses import asdict, dataclass, field
from typing import Any, Literal


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
    target: str
    relation_type: str
    sentiment: Literal["positive", "neutral", "negative"] = "neutral"
    description: str = ""


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

    def for_prompt(self) -> dict[str, Any]:
        """Serialize the context into prompt-ready sections."""
        return {
            "world_state": self.story_state,
            "characters": list(self.characters.values()),
            "recent_events": self.events[-5:],
            "recent_chapters": self.chapter_window[-5:],
        }

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
            relationships = [Relationship(**rel) for rel in value.get("relationships", [])]
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
        return cls(
            metadata=payload.get("metadata", {}),
            chapter_window=chapter_window,
            events=events,
            characters=characters,
            story_state=story_state,
        )

    def add_metadata(self, key: str, value: Any) -> None:
        """Record helper metadata for downstream consumers."""
        self.metadata[key] = value
