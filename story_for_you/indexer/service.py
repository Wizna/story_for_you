from __future__ import annotations

from story_for_you.analysis.context import StoryContext
from story_for_you.indexer.segment import Gap, Segment, SegmentIndex
from story_for_you.indexer.tagger import CharacterTagger


class SegmentIndexService:
    """Creates and caches reusable segment indexes."""

    def __init__(self):
        self._cache: dict[str, SegmentIndex] = {}

    def build(self, context: StoryContext, segments: list[Segment]) -> SegmentIndex:
        """Build a fresh SegmentIndex for the provided context."""
        tagger = CharacterTagger(context)
        tagged_segments = tagger.tag(segments)
        char_index: dict[str, list[int]] = {}
        chapter_index: dict[int, list[int]] = {}
        for segment in tagged_segments:
            for name in segment.characters:
                char_index.setdefault(name, []).append(segment.segment_id)
            if segment.chapter is not None:
                try:
                    chapter_key = int(segment.chapter)
                except (TypeError, ValueError):
                    continue
                chapter_index.setdefault(chapter_key, []).append(segment.segment_id)
        gap_map = self._build_gap_map(tagged_segments)
        index = SegmentIndex(
            segments=tagged_segments,
            char_index=char_index,
            chapter_index=chapter_index,
            gap_map=gap_map,
        )
        fingerprint = context.metadata.get("_fingerprint")
        if fingerprint:
            self._cache[fingerprint] = index
        return index

    def get(self, key: str) -> SegmentIndex | None:
        """Return a cached index by key."""
        return self._cache.get(key)

    def set(self, key: str, index: SegmentIndex) -> None:
        """Store an index for future reuse."""
        self._cache[key] = index

    def _build_gap_map(self, segments: list[Segment]) -> dict[str, Gap]:
        """Create adjacency gaps used for bridge generation."""
        gap_map: dict[str, Gap] = {}
        ordered = sorted(segments, key=lambda item: item.segment_id)
        for first, second in zip(ordered, ordered[1:]):
            key = f"{first.segment_id}->{second.segment_id}"
            chapter_label = f"{first.chapter or '?'} → {second.chapter or '?'}"
            gap_map[key] = Gap(start_id=first.segment_id, end_id=second.segment_id, description=chapter_label)
        return gap_map
