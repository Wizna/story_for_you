"""Serialization utilities for segment and index data.

Provides functions to convert Segment and SegmentIndex objects to/from
dictionary representations for caching and persistence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from story_for_you.indexer.segment import Gap, Segment, SegmentIndex

__all__ = [
    "serialize_segments",
    "deserialize_segments",
    "serialize_index",
    "deserialize_index",
]


def serialize_segments(segments: list[Segment]) -> list[dict]:
    """Convert a list of Segment objects to serializable dictionaries."""
    serialized: list[dict] = []
    for segment in segments:
        serialized.append(
            {
                "segment_id": segment.segment_id,
                "content": segment.content,
                "chapter": segment.chapter,
                "characters": segment.characters,
                "metadata": segment.metadata,
            }
        )
    return serialized


def deserialize_segments(payload: list[dict]) -> list[Segment]:
    """Restore Segment objects from serialized dictionaries."""
    from story_for_you.indexer.segment import Segment

    return [
        Segment(
            segment_id=item["segment_id"],
            content=item["content"],
            chapter=item.get("chapter"),
            characters=item.get("characters", []),
            metadata=item.get("metadata", {}),
        )
        for item in payload
    ]


def serialize_index(index: SegmentIndex) -> dict:
    """Convert a SegmentIndex to a serializable dictionary."""
    return {
        "char_index": index.char_index,
        "chapter_index": index.chapter_index,
        "gap_map": {
            key: {
                "start_id": gap.start_id,
                "end_id": gap.end_id,
                "description": gap.description,
            }
            for key, gap in index.gap_map.items()
        },
    }


def deserialize_index(payload: dict, segments: list[Segment]) -> SegmentIndex:
    """Restore a SegmentIndex from a serialized dictionary."""
    from story_for_you.indexer.segment import Gap, SegmentIndex

    gap_map = {
        key: Gap(
            start_id=value["start_id"],
            end_id=value["end_id"],
            description=value.get("description", ""),
        )
        for key, value in payload.get("gap_map", {}).items()
    }
    return SegmentIndex(
        segments=segments,
        char_index=payload.get("char_index", {}),
        chapter_index=payload.get("chapter_index", {}),
        gap_map=gap_map,
    )
