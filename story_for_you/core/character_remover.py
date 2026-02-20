from dataclasses import dataclass
from typing import Literal

import logging

from story_for_you.analysis.context import StoryContext
from story_for_you.core.exceptions import LLMError
from story_for_you.indexer.retriever import SegmentRetriever
from story_for_you.indexer.segment import Segment
from story_for_you.llm.base import LLMProvider
from story_for_you.core.prompting import (
    fill_template,
    format_context_sections,
    format_style_guide,
    load_template,
)

logger = logging.getLogger(__name__)


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
        self.rewrite_template = load_template("remove_rewrite")

    def remove(self, text: str, characters: list[str], context: StoryContext, mode: str = "hard") -> RemoveResult:
        """Return a removal result for the supplied characters."""
        kept_segments = self.retriever.retrieve_excluding(exclude=characters, mode=mode)
        affected_segments = self.retriever.retrieve_by_characters(include=characters, mode="strict")
        processed: list[Segment] = []
        deleted = rewritten = replaced = 0
        context_block = format_context_sections(context.for_prompt())
        style_guide = format_style_guide(context.writing_style)
        for segment in affected_segments:
            action = self._evaluate_segment(segment, characters, context, mode)
            if action == "delete":
                deleted += 1
                continue
            if action == "minimal_rewrite":
                processed.append(self._minimal_rewrite(segment, characters, context_block, mode, style_guide))
                rewritten += 1
            else:
                processed.append(self._replace_names(segment, characters))
                replaced += 1
        merged = sorted(kept_segments + processed, key=lambda seg: seg.segment_id)
        content = "\n\n".join(seg.content.strip() for seg in merged)
        ratio = len(content) / max(len(text), 1)
        return RemoveResult(
            content=content,
            original_ratio=ratio,
            deleted_segments=deleted,
            rewritten_segments=rewritten,
            replaced_segments=replaced,
        )

    def _evaluate_segment(
        self,
        segment: Segment,
        characters: list[str],
        context: StoryContext,
        mode: str,
    ) -> Literal["delete", "keep_with_replace", "minimal_rewrite"]:
        """Decide how to treat a segment that references removed characters."""
        lowered = segment.content.lower()
        occurrences = sum(lowered.count(name.lower()) for name in characters)
        if occurrences >= 3 or (mode == "hard" and occurrences >= 2):
            return "delete"
        if occurrences == 2 and context.story_state and context.story_state.world_tension != "low":
            return "minimal_rewrite"
        return "keep_with_replace"

    def _replace_names(self, segment: Segment, characters: list[str]) -> Segment:
        """Perform simple textual replacements without LLM help."""
        content = segment.content
        for name in characters:
            content = content.replace(name, "")
        return Segment(
            segment_id=segment.segment_id,
            content=" ".join(content.split()),
            chapter=segment.chapter,
            characters=[name for name in segment.characters if name not in characters],
            metadata=segment.metadata,
        )

    def _minimal_rewrite(
        self, segment: Segment, characters: list[str], context_block: str, mode: str, style_guide: str
    ) -> Segment:
        """Ask the LLM to minimally rewrite a conflicting segment."""
        prompt = fill_template(
            self.rewrite_template,
            context_block=context_block,
            mode=mode,
            characters=", ".join(characters),
            segment_text=segment.content.strip(),
            style_guide=style_guide,
        )
        rewritten = self._call_llm(prompt) or "[Adjusted] 该段落已删除关键人物引用并保留结果。"
        return Segment(
            segment_id=segment.segment_id,
            content=rewritten,
            chapter=segment.chapter,
            characters=[name for name in segment.characters if name not in characters],
            metadata=segment.metadata,
        )

    def _call_llm(self, prompt: str) -> str | None:
        try:
            response = self.llm.generate(prompt=prompt)
        except LLMError as exc:  # pragma: no cover
            logger.warning("Character remover rewrite failed: %s", exc)
            return None
        text = response.content.strip()
        return text or None
