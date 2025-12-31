from __future__ import annotations

import logging

from story_for_you.analysis.context import StoryContext
from story_for_you.indexer.segment import SegmentIndex
from story_for_you.llm.base import LLMProvider
from story_for_you.core.prompting import fill_template, format_context_sections, load_template

logger = logging.getLogger(__name__)


class EndingWriter:
    """Produces alternate endings grounded in StoryContext."""

    def __init__(self, llm: LLMProvider, segment_index: SegmentIndex):
        self.llm = llm
        self.segment_index = segment_index
        self.template = load_template("ending")

    def continue_story(self, text: str, context: StoryContext, hint: str = "") -> str:
        """Return a continuation for the supplied story text."""
        context_block = format_context_sections(context.for_prompt())
        recent_segments = self._recent_segment_digest()
        prompt = fill_template(
            self.template,
            context_block=context_block,
            recent_segments=recent_segments,
            hint=hint or "无特别需求，按原作基调收束",
        )
        try:
            response = self.llm.generate(prompt=prompt)
            text = response.content.strip()
            if text:
                return text
        except Exception as exc:  # pragma: no cover
            logger.warning("Ending continuation failed, falling back to heuristic ending: %s", exc)
        return self._fallback(context, hint)

    def _recent_segment_digest(self) -> str:
        segments = self.segment_index.segments[-3:]
        if not segments:
            return "(无可参考片段)"
        parts = []
        for segment in segments:
            content = segment.content.strip().replace("\n", " ")
            parts.append(f"[Segment {segment.segment_id}] {content[:280]}")
        return "\n".join(parts)

    def _fallback(self, context: StoryContext, hint: str) -> str:
        last_summary = context.chapter_window[-1] if context.chapter_window else None
        recent_events = context.events[-3:]
        tone = context.story_state.world_tension if context.story_state else "medium"
        paragraphs: list[str] = []
        if last_summary:
            paragraphs.append(
                f"After {last_summary.title}, {last_summary.synopsis.lower()} sets the stage for the finale."
            )
        if recent_events:
            beats = "; ".join(event.summary for event in recent_events)
            paragraphs.append(f"Consequences converge: {beats}.")
        if hint:
            paragraphs.append(f"Guided by the request ({hint}), the protagonists choose a fitting resolution.")
        tension_line = "The atmosphere eases." if tone == "low" else "Tension peaks before dissolving."
        paragraphs.append(tension_line)
        paragraphs.append("Loose threads are acknowledged, promising future tales without contradicting the past.")
        paragraphs.append("(本段为读者定制版本的兜底续写)")
        return "\n\n".join(paragraphs)
