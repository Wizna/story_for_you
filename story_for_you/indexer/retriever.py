from story_for_you.indexer.segment import Segment


class SegmentRetriever:
    """Retrieves segments by character or exclusion criteria."""

    def __init__(self, segments: list[Segment]):
        self.segments = segments

    def _build_index(self) -> dict[str, list[int]]:
        """Create a mapping of character names to segment ids."""
        index: dict[str, list[int]] = {}
        for segment in self.segments:
            for name in segment.characters:
                index.setdefault(name, []).append(segment.segment_id)
        return index

    def retrieve_by_characters(self, include: list[str], mode: str = "soft") -> list[Segment]:
        """Return segments where the target characters appear."""
        raise NotImplementedError

    def retrieve_excluding(self, exclude: list[str], mode: str = "hard") -> list[Segment]:
        """Return segments that exclude the provided characters."""
        raise NotImplementedError
