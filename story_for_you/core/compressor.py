from __future__ import annotations

from collections.abc import Iterator

from story_for_you.analysis.context import PlotEvent, StoryContext
from story_for_you.config.settings import RenderingLimits
from story_for_you.core.exceptions import LLMResponseError
from story_for_you.indexer.segment import Segment, SegmentIndex, deduplicate_overlapping_segments
from story_for_you.llm.base import LLMProvider
from story_for_you.llm.telemetry import telemetry_options
from story_for_you.parser.text_splitter import TextSplitter
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
_SEGMENT_SEPARATOR = "\n---\n"


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
        context_window: int = 1_000_000,
        max_output_tokens: int = 32768,
    ):
        self.llm = llm
        self.segment_index = segment_index
        self.level = level
        self.levels = levels or self.DEFAULT_LEVELS
        self._limits = rendering_limits or RenderingLimits()
        self.context_window = context_window
        self.max_output_tokens = max_output_tokens
        self.template = load_template("compress")

    def compress(self, context: StoryContext) -> str:
        """Rewrite selected story segments in bounded source-order batches."""
        ordered = deduplicate_overlapping_segments(self._select_segments(context))
        if not ordered:
            raise LLMResponseError("No story segments available for compression.")
        placeholders = {
            "level": self.level,
            "context_block": format_context_sections(context.for_prompt(limits=self._limits)),
            "style_guide": format_style_guide(context.writing_style),
            "style_samples": format_style_samples(context.writing_style),
        }
        output_reserve = self.max_output_tokens
        overhead = len(fill_template(self.template, segments="", **placeholders).encode("utf-8"))
        payload_budget = min(self.max_output_tokens, self.context_window - output_reserve - overhead)
        if payload_budget <= 0:
            raise LLMResponseError("Compression context window is too small for the prompt and output budget.")

        results: list[str] = []
        for batch in self._batches(ordered, payload_budget):
            prompt = fill_template(self.template, segments=batch, **placeholders)
            response = self.llm.generate(
                prompt=cache_prompt(prompt),
                options=telemetry_options(phase="compress", step=": rewrite selected segments"),
            )
            content = response.content.strip()
            if not content:
                raise LLMResponseError("Story compression returned empty content.")
            results.append(content)
        return "\n\n".join(results)

    # Internal helpers -------------------------------------------------
    def _batches(self, segments: list[Segment], budget: int) -> Iterator[str]:
        splitter = TextSplitter(chunk_size=max(1, budget // 4), overlap=0)
        parts: list[str] = []
        size = 0
        for segment in segments:
            for chunk in splitter.split(segment.content):
                content = chunk.content.strip()
                if not content:
                    continue
                content_size = len(content.encode("utf-8"))
                if content_size > budget:
                    raise LLMResponseError("Compression input contains a character larger than the batch budget.")
                added = content_size + (len(_SEGMENT_SEPARATOR) if parts else 0)
                if parts and size + added > budget:
                    yield _SEGMENT_SEPARATOR.join(parts)
                    parts = []
                    size = 0
                    added = content_size
                parts.append(content)
                size += added
        if parts:
            yield _SEGMENT_SEPARATOR.join(parts)

    def _select_segments(self, context: StoryContext) -> list[Segment]:
        ratio = self.levels.get(self.level, self.levels["medium"])
        total = len(self.segment_index.segments)
        keep = max(1, int(total * ratio))
        ranked = sorted(
            self.segment_index.segments,
            key=lambda segment: self._score_segment(segment, context),
            reverse=True,
        )
        return ranked[:keep]

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
