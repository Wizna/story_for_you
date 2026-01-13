from __future__ import annotations

from dataclasses import dataclass

import logging

from story_for_you.analysis.context import PlotEvent, StoryContext
from story_for_you.indexer.segment import Segment, SegmentIndex
from story_for_you.llm.base import LLMProvider
from story_for_you.core.prompting import (
    fill_template,
    format_context_sections,
    format_style_guide,
    format_style_samples,
    load_template,
)

logger = logging.getLogger(__name__)


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
        self.template = load_template("compress")

    def compress(self, text: str, context: StoryContext) -> str:
        """Return a compressed version of the provided text."""
        targets = self._select_segments(context)
        ordered = sorted(targets, key=lambda item: item.segment.segment_id)
        context_block = format_context_sections(context.for_prompt())
        style_guide = format_style_guide(context.writing_style)
        style_samples = format_style_samples(context.writing_style)
        segments_payload = "\n---\n".join(item.segment.content.strip() for item in ordered)
        prompt = fill_template(
            self.template,
            level=self.level,
            context_block=context_block,
            segments=segments_payload,
            style_guide=style_guide,
            style_samples=style_samples,
        )
        try:
            response = self.llm.generate(prompt=prompt)
            content = response.content.strip()
            if content:
                return content
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Story compression failed, falling back to raw segments: %s", exc)
        return self._fallback_content(ordered, context)

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

    def _fallback_content(self, targets: list["SegmentScore"], context: StoryContext) -> str:
        """Fallback concatenation when LLM compression fails."""
        content = "\n\n".join(item.segment.content.strip() for item in targets)
        if context.story_state and context.story_state.current_arc:
            marker = f"[Compression:{context.story_state.current_arc}] "
            return marker + content
        return content


@dataclass
class SegmentScore:
    segment: Segment
    score: float
