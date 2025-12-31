from story_for_you.analysis.context import StoryContext
from story_for_you.indexer.segment import SegmentIndex
from story_for_you.llm.base import LLMProvider


class EndingWriter:
    """Produces alternate endings grounded in StoryContext."""

    def __init__(self, llm: LLMProvider, segment_index: SegmentIndex):
        self.llm = llm
        self.segment_index = segment_index

    def continue_story(self, text: str, context: StoryContext, hint: str = "") -> str:
        """Return a continuation for the supplied story text."""
        raise NotImplementedError
