from __future__ import annotations

from story_for_you.indexer.segment import Segment, SegmentIndex


class SegmentRetriever:
    """Retrieves segments by character or exclusion criteria."""

    def __init__(self, segment_index: SegmentIndex):
        self.segment_index = segment_index
        self.segments = list(segment_index.segments)
        self._char_index = segment_index.char_index or self._build_index()
        self._by_id = {segment.segment_id: segment for segment in self.segments}
        self._order = {segment.segment_id: idx for idx, segment in enumerate(self.segments)}

    def _build_index(self) -> dict[str, list[int]]:
        """Create a mapping of character names to segment ids."""
        index: dict[str, list[int]] = {}
        for segment in self.segments:
            for name in segment.characters:
                index.setdefault(name, []).append(segment.segment_id)
        return index

    def retrieve_by_characters(self, include: list[str], mode: str = "soft") -> list[Segment]:
        """Return segments where the target characters appear."""
        if not include:
            return []
        include_ids: set[int] = set()
        for name in include:
            include_ids.update(self._char_index.get(name, []))
        ordered_ids = sorted(include_ids, key=lambda seg_id: self._order.get(seg_id, seg_id))
        segments = [self._by_id[seg_id] for seg_id in ordered_ids]
        if mode == "soft":
            segments = self._expand_with_neighbors(segments)
        return segments

    def retrieve_excluding(self, exclude: list[str], mode: str = "hard") -> list[Segment]:
        """Return segments that exclude the provided characters."""
        if not exclude:
            return self.segments
        excluded = {name.lower() for name in exclude}
        filtered: list[Segment] = []
        for segment in self.segments:
            mentions = {name.lower() for name in segment.characters}
            overlap = mentions.intersection(excluded)
            if mode == "hard":
                if not overlap:
                    filtered.append(segment)
                continue
            # soft mode keeps segments with incidental mentions
            score = self._mention_score(segment.content, excluded)
            if score <= 1:
                filtered.append(segment)
        return filtered

    def _expand_with_neighbors(self, segments: list[Segment]) -> list[Segment]:
        neighbors: set[int] = set()
        for segment in segments:
            idx = self._order.get(segment.segment_id)
            if idx is None:
                continue
            neighbors.add(segment.segment_id)
            if idx > 0:
                neighbors.add(self.segments[idx - 1].segment_id)
            if idx + 1 < len(self.segments):
                neighbors.add(self.segments[idx + 1].segment_id)
        ordered_ids = sorted(neighbors, key=lambda seg_id: self._order.get(seg_id, seg_id))
        return [self._by_id[seg_id] for seg_id in ordered_ids]

    def _mention_score(self, content: str, targets: set[str]) -> int:
        lowered = content.lower()
        score = 0
        for target in targets:
            score += lowered.count(target)
        return score
