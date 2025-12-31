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
        segments = self.retriever.retrieve_by_characters(include=characters, mode=mode)
        if not segments:
            return FilterResult(content="", original_ratio=0.0)
        gaps = self._find_gaps(segments)
        bridges: list[BridgeInfo] = []
        for gap in gaps:
            bridge_text = self._generate_bridge(gap, context)
            bridges.append(BridgeInfo(start_id=gap.start_id, end_id=gap.end_id, content=bridge_text))
        content = self._assemble(segments, bridges)
        ratio = len(content) / max(len(text), 1)
        return FilterResult(content=content, original_ratio=ratio, bridges=bridges)

    def _find_gaps(self, segments: list[Segment]) -> list[BridgeInfo]:
        """Identify structural gaps between the selected segments."""
        if not segments:
            return []
        ordered = sorted(segments, key=lambda seg: seg.segment_id)
        gaps: list[BridgeInfo] = []
        for first, second in zip(ordered, ordered[1:]):
            if second.segment_id - first.segment_id > 1:
                gaps.append(BridgeInfo(start_id=first.segment_id, end_id=second.segment_id, content=""))
        return gaps

    def _generate_bridge(self, gap: BridgeInfo, context: StoryContext) -> str:
        """Generate a minimal bridge for the supplied gap."""
        arc = context.story_state.current_arc if context.story_state else "story"
        return f"[Bridge:{arc}] Transition between segments {gap.start_id} and {gap.end_id}."

    def _assemble(self, segments: list[Segment], bridges: list[BridgeInfo]) -> str:
        ordered = sorted(segments, key=lambda seg: seg.segment_id)
        bridge_map = {(bridge.start_id, bridge.end_id): bridge.content for bridge in bridges}
        parts: list[str] = []
        for idx, segment in enumerate(ordered):
            parts.append(segment.content.strip())
            if idx == len(ordered) - 1:
                continue
            next_segment = ordered[idx + 1]
            key = (segment.segment_id, next_segment.segment_id)
            if key in bridge_map:
                parts.append(bridge_map[key])
        return "\n\n".join(parts)
