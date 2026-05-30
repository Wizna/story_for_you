from dataclasses import dataclass, field

from story_for_you.analysis.context import StoryContext
from story_for_you.config.settings import RenderingLimits
from story_for_you.core.exceptions import LLMResponseError
from story_for_you.indexer.retriever import SegmentRetriever
from story_for_you.indexer.segment import Segment
from story_for_you.llm.base import LLMProvider
from story_for_you.llm.telemetry import telemetry_options
from story_for_you.core.prompting import (
    fill_template,
    format_context_sections,
    format_style_guide,
    load_template,
)
from story_for_you.utils.prompting import SNIPPET_EXCERPT_LEN, cache_prompt

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

    def __init__(self, llm: LLMProvider, retriever: SegmentRetriever, rendering_limits: RenderingLimits | None = None):
        self.llm = llm
        self.retriever = retriever
        self._limits = rendering_limits or RenderingLimits()
        self.bridge_template = load_template("filter_bridge")

    def filter(self, text: str, characters: list[str], context: StoryContext, mode: str = "soft") -> FilterResult:
        """Return filtered content focused on the requested characters."""
        segments = self.retriever.retrieve_by_characters(include=characters, mode=mode)
        if not segments:
            return FilterResult(content="", original_ratio=0.0)
        gaps = self._find_gaps(segments)
        context_block = format_context_sections(context.for_prompt(limits=self._limits))
        style_guide = format_style_guide(context.writing_style)
        bridges: list[BridgeInfo] = []
        for gap in gaps:
            gap.content = self._generate_bridge(gap, context_block, characters, style_guide)
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

    def _generate_bridge(self, gap: BridgeInfo, context_block: str, characters: list[str], style_guide: str) -> str:
        """Generate a minimal bridge for the supplied gap."""
        prompt = fill_template(
            self.bridge_template,
            context_block=context_block,
            bridge_label=f"{gap.start_id}->{gap.end_id}",
            characters=", ".join(characters) or "目标人物",
            before_excerpt=gap.before_excerpt or "(无前文摘录)",
            after_excerpt=gap.after_excerpt or "(无后文摘录)",
            style_guide=style_guide,
        )
        response = self.llm.generate(
            prompt=cache_prompt(prompt),
            options=telemetry_options(
                phase="filter",
                step=f": bridge segment gap {gap.start_id}->{gap.end_id}",
            ),
        )
        text = response.content.strip()
        if not text:
            raise LLMResponseError("Bridge generation returned empty content.")
        return text

    def _excerpt(self, content: str, tail: bool) -> str:
        snippet = content.strip()
        if len(snippet) <= SNIPPET_EXCERPT_LEN:
            return snippet
        return snippet[-SNIPPET_EXCERPT_LEN:] if tail else snippet[:SNIPPET_EXCERPT_LEN]

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
