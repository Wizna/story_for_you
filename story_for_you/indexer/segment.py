from dataclasses import dataclass, field
from typing import Any


@dataclass
class Segment:
    segment_id: int
    content: str
    chapter: int | None = None
    characters: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Gap:
    start_id: int
    end_id: int
    description: str = ""


@dataclass
class SegmentIndex:
    segments: list[Segment]
    char_index: dict[str, list[int]]
    chapter_index: dict[int, list[int]]
    gap_map: dict[str, Gap]
