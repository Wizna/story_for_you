from dataclasses import dataclass

from story_for_you.analysis.context import StoryContext
from story_for_you.indexer.retriever import SegmentRetriever
from story_for_you.indexer.segment import Segment
from story_for_you.llm.base import LLMProvider


@dataclass
class RemoveResult:
    content: str
    original_ratio: float
    deleted_segments: int
    rewritten_segments: int
    replaced_segments: int


class CharacterRemover:
    """Removes or rewrites characters from story segments."""

    def __init__(self, llm: LLMProvider, retriever: SegmentRetriever):
        self.llm = llm
        self.retriever = retriever

    def remove(self, text: str, characters: list[str], context: StoryContext, mode: str = "hard") -> RemoveResult:
        """Return a removal result for the supplied characters."""
        raise NotImplementedError

    def _evaluate_segment(self, segment: Segment, characters: list[str], context: StoryContext):
        """Decide how to treat a segment that references removed characters."""
        raise NotImplementedError

    def _replace_names(self, segment: Segment, characters: list[str]) -> Segment:
        """Perform simple textual replacements without LLM help."""
        raise NotImplementedError

    def _minimal_rewrite(self, segment: Segment, characters: list[str], context: StoryContext) -> Segment:
        """Ask the LLM to minimally rewrite a conflicting segment."""
        raise NotImplementedError
