from dataclasses import dataclass, field, replace
from typing import Any, Iterable


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


def deduplicate_overlapping_segments(segments: Iterable[Segment]) -> list[Segment]:
    """Return ordered segments with source-span overlap removed.

    Segments produced by the text splitter may overlap so that an LLM sees
    enough context at chunk boundaries.  They must not retain that overlap
    when they are assembled into a user-facing result.  Segments without
    numeric ``start``/``end`` metadata are left untouched because their
    provenance is unknown.
    """
    ordered = sorted(segments, key=lambda item: item.segment_id)
    result: list[Segment] = []
    covered_end: int | None = None
    for segment in ordered:
        start = segment.metadata.get("start")
        end = segment.metadata.get("end")
        if not isinstance(start, int) or isinstance(start, bool) or not isinstance(end, int) or isinstance(end, bool):
            result.append(segment)
            continue

        span_length = end - start
        if span_length < 0 or len(segment.content) != span_length:
            # LLM rewrites retain the source span metadata for ordering, but
            # their generated length no longer maps to source characters.
            result.append(segment)
            covered_end = max(covered_end if covered_end is not None else end, end)
            continue

        trim = max(0, covered_end - start) if covered_end is not None else 0
        content = segment.content
        if trim:
            if trim >= len(content):
                covered_end = max(covered_end if covered_end is not None else end, end)
                continue
            content = content[trim:]
        if content.strip():
            result.append(replace(segment, content=content))
        covered_end = max(covered_end if covered_end is not None else end, end)
    return result
