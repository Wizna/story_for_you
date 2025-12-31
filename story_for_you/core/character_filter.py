from dataclasses import dataclass, field

import logging

from story_for_you.analysis.context import StoryContext
from story_for_you.indexer.retriever import SegmentRetriever
from story_for_you.indexer.segment import Segment
from story_for_you.llm.base import LLMProvider
from story_for_you.core.prompting import fill_template, format_context_sections, load_template

logger = logging.getLogger(__name__)


@dataclass
class BridgeInfo:
    start_id: int
    end_id: int
    content: str
    before_excerpt: str = ""
    after_excerpt: str = ""


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
        self.bridge_template = load_template("filter_bridge")

    def filter(self, text: str, characters: list[str], context: StoryContext, mode: str = "soft") -> FilterResult:
        """Return filtered content focused on the requested characters."""
        segments = self.retriever.retrieve_by_characters(include=characters, mode=mode)
        if not segments:
            return FilterResult(content="", original_ratio=0.0)
        gaps = self._find_gaps(segments)
        context_block = format_context_sections(context.for_prompt())
        bridges: list[BridgeInfo] = []
        for gap in gaps:
            gap.content = self._generate_bridge(gap, context_block, characters)
            bridges.append(gap)
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
                gaps.append(
                    BridgeInfo(
                        start_id=first.segment_id,
                        end_id=second.segment_id,
                        content="",
                        before_excerpt=self._excerpt(first.content, tail=True),
                        after_excerpt=self._excerpt(second.content, tail=False),
                    )
                )
        return gaps

    def _generate_bridge(self, gap: BridgeInfo, context_block: str, characters: list[str]) -> str:
        """Generate a minimal bridge for the supplied gap."""
        prompt = fill_template(
            self.bridge_template,
            context_block=context_block,
            bridge_label=f"{gap.start_id}->{gap.end_id}",
            characters=", ".join(characters) or "目标人物",
            before_excerpt=gap.before_excerpt or "(无前文摘录)",
            after_excerpt=gap.after_excerpt or "(无后文摘录)",
        )
        try:
            response = self.llm.generate(prompt=prompt)
            text = response.content.strip()
            if text:
                return text
        except Exception as exc:  # pragma: no cover
            logger.warning("Bridge generation failed, falling back to marker: %s", exc)
        fallback_name = characters[0] if characters else "人物"
        return f"[Bridge] 期间与 {fallback_name} 相关的情节被省略。"

    def _excerpt(self, content: str, tail: bool) -> str:
        snippet = content.strip()
        if len(snippet) <= 280:
            return snippet
        return snippet[-280:] if tail else snippet[:280]

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
