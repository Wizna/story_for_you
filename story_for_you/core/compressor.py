from __future__ import annotations

from dataclasses import dataclass

from story_for_you.analysis.context import PlotEvent, StoryContext
from story_for_you.config.settings import RenderingLimits
from story_for_you.core.exceptions import LLMResponseError
from story_for_you.indexer.segment import Segment, SegmentIndex
from story_for_you.llm.base import LLMProvider
from story_for_you.llm.telemetry import telemetry_options
from story_for_you.utils.prompting import cache_prompt
from story_for_you.core.prompting import (
    fill_template,
    format_context_sections,
    format_style_guide,
    format_style_samples,
    load_template,
)

_SCORE_MAIN_CHARACTER = 2.0
_SCORE_SUPPORT_CHARACTER = 1.0
_SCORE_IRREVERSIBLE_EVENT = 3.0
_SCORE_NORMAL_EVENT = 1.0


class StoryCompressor:
    """Compresses story content while respecting StoryContext cues."""

    DEFAULT_LEVELS = {"light": 0.8, "medium": 0.5, "heavy": 0.3}

    def __init__(
        self,
        llm: LLMProvider,
        segment_index: SegmentIndex,
        level: str = "medium",
        levels: dict[str, float] | None = None,
        rendering_limits: RenderingLimits | None = None,
    ):
        self.llm = llm
        self.segment_index = segment_index
        self.level = level
        self.levels = levels or self.DEFAULT_LEVELS
        self._limits = rendering_limits or RenderingLimits()
        self.template = load_template("compress")

    def compress(self, text: str, context: StoryContext) -> str:
        """Return a compressed version of the provided text."""
        targets = self._select_segments(context)
        ordered = sorted(targets, key=lambda item: item.segment.segment_id)
        context_block = format_context_sections(context.for_prompt(limits=self._limits))
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
        response = self.llm.generate(
            prompt=cache_prompt(prompt),
            options=telemetry_options(phase="compress", step=": rewrite selected segments"),
        )
        content = response.content.strip()
        if not content:
            raise LLMResponseError("Story compression returned empty content.")
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
                score += _SCORE_MAIN_CHARACTER if character.role == "main" else _SCORE_SUPPORT_CHARACTER
        for event in self._events_for_segment(context, segment):
            if event.is_irreversible:
                score += _SCORE_IRREVERSIBLE_EVENT
            else:
                score += _SCORE_NORMAL_EVENT
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
