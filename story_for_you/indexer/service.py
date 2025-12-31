from story_for_you.analysis.context import StoryContext
from story_for_you.indexer.segment import Segment, SegmentIndex


class SegmentIndexService:
    """Creates and caches reusable segment indexes."""

    def __init__(self):
        self._cache: dict[str, SegmentIndex] = {}

    def build(self, context: StoryContext, segments: list[Segment]) -> SegmentIndex:
        """Build a fresh SegmentIndex for the provided context."""
        raise NotImplementedError

    def get(self, key: str) -> SegmentIndex | None:
        """Return a cached index by key."""
        return self._cache.get(key)

    def set(self, key: str, index: SegmentIndex) -> None:
        """Store an index for future reuse."""
        self._cache[key] = index
