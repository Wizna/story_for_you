from __future__ import annotations

from dataclasses import dataclass

from story_for_you.analysis.context import PlotEvent, StoryContext
from story_for_you.indexer.segment import Segment, SegmentIndex
from story_for_you.llm.base import LLMProvider


class StoryCompressor:
    """Compresses story content while respecting StoryContext cues."""

    DEFAULT_LEVELS = {"light": 0.8, "medium": 0.5, "heavy": 0.3}

    def __init__(
        self,
        llm: LLMProvider,
        segment_index: SegmentIndex,
        level: str = "medium",
        levels: dict[str, float] | None = None,
    ):
        self.llm = llm
        self.segment_index = segment_index
        self.level = level
        self.levels = levels or self.DEFAULT_LEVELS

    def compress(self, text: str, context: StoryContext) -> str:
        """Return a compressed version of the provided text."""
        targets = self._select_segments(context)
        ordered = sorted(targets, key=lambda item: item.segment.segment_id)
        content = "\n\n".join(segment.content.strip() for segment in (item.segment for item in ordered))
        if context.story_state and context.story_state.current_arc:
            marker = f"[Compression:{context.story_state.current_arc}] "
            return marker + content
        return content

    # Internal helpers -------------------------------------------------
    def _select_segments(self, context: StoryContext) -> list["SegmentScore"]:
        ratio = self.levels.get(self.level, self.levels["medium"])
        total = len(self.segment_index.segments)
        keep = max(1, int(total * ratio))
        scored: list[SegmentScore] = []
        for segment in self.segment_index.segments:
            score = self._score_segment(segment, context)
            scored.append(SegmentScore(segment=segment, score=score))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:keep]

    def _score_segment(self, segment: Segment, context: StoryContext) -> float:
        score = 1.0
        for character in context.characters.values():
            if character.name in segment.characters:
                score += 2.0 if character.role == "main" else 1.0
        for event in self._events_for_segment(context, segment):
            if event.is_irreversible:
                score += 3.0
            else:
                score += 1.0
        return score

    def _events_for_segment(self, context: StoryContext, segment: Segment) -> list[PlotEvent]:
        matches = []
        for event in context.events:
            if segment.chapter and event.chapter == segment.chapter:
                matches.append(event)
        return matches


@dataclass
class SegmentScore:
    segment: Segment
    score: float
