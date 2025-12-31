from story_for_you.analysis.context import StoryContext
from story_for_you.indexer.segment import SegmentIndex
from story_for_you.llm.base import LLMProvider


class StoryCompressor:
    """Compresses story content while respecting StoryContext cues."""

    def __init__(self, llm: LLMProvider, segment_index: SegmentIndex, level: str = "medium"):
        self.llm = llm
        self.segment_index = segment_index
        self.level = level

    def compress(self, text: str, context: StoryContext) -> str:
        """Return a compressed version of the provided text."""
        raise NotImplementedError
