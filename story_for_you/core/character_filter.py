from dataclasses import dataclass, field

from story_for_you.analysis.context import StoryContext
from story_for_you.indexer.retriever import SegmentRetriever
from story_for_you.indexer.segment import Segment
from story_for_you.llm.base import LLMProvider


@dataclass
class BridgeInfo:
    start_id: int
    end_id: int
    content: str


@dataclass
class FilterResult:
    content: str
    original_ratio: float = 1.0
    bridges: list[BridgeInfo] = field(default_factory=list)


class CharacterFilter:
    """Filters story segments for selected characters."""

    def __init__(self, llm: LLMProvider, retriever: SegmentRetriever):
        self.llm = llm
        self.retriever = retriever

    def filter(self, text: str, characters: list[str], context: StoryContext, mode: str = "soft") -> FilterResult:
        """Return filtered content focused on the requested characters."""
        raise NotImplementedError

    def _find_gaps(self, segments: list[Segment]) -> list[BridgeInfo]:
        """Identify structural gaps between the selected segments."""
        raise NotImplementedError

    def _generate_bridge(self, gap: BridgeInfo, context: StoryContext) -> str:
        """Generate a minimal bridge for the supplied gap."""
        raise NotImplementedError
