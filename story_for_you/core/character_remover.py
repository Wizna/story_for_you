from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from story_for_you.analysis.context import StoryContext
from story_for_you.config.settings import RenderingLimits
from story_for_you.core.exceptions import LLMResponseError
from story_for_you.indexer.retriever import SegmentRetriever
from story_for_you.indexer.segment import Segment
from story_for_you.llm.base import LLMProvider
from story_for_you.core.prompting import (
    fill_template,
    format_context_sections,
    format_style_guide,
    load_template,
)
from story_for_you.utils.json_utils import load_json_response

@dataclass
class RemoveResult:
    content: str
    original_ratio: float
    deleted_segments: int
    rewritten_segments: int
    replaced_segments: int


class CharacterRemover:
    """Removes or rewrites characters from story segments using the configured LLM."""

    def __init__(self, llm: LLMProvider, retriever: SegmentRetriever, rendering_limits: RenderingLimits | None = None):
        self.llm = llm
        self.retriever = retriever
        self._limits = rendering_limits or RenderingLimits()
        self.rewrite_template = load_template("remove_rewrite")

    def remove(self, text: str, characters: list[str], context: StoryContext, mode: str = "hard") -> RemoveResult:
        """Return a removal result for the supplied characters."""
        kept_segments = self.retriever.retrieve_excluding(exclude=characters, mode="hard")
        affected_segments = self.retriever.retrieve_by_characters(include=characters, mode="strict")
        processed: list[Segment] = []
        deleted = rewritten = 0
        context_block = format_context_sections(context.for_prompt(limits=self._limits))
        style_guide = format_style_guide(context.writing_style)
        for segment in affected_segments:
            action, content = self._rewrite_or_delete(segment, characters, context_block, mode, style_guide)
            if action == "delete":
                deleted += 1
                continue
            processed.append(
                Segment(
                    segment_id=segment.segment_id,
                    content=content,
                    chapter=segment.chapter,
                    characters=[name for name in segment.characters if name not in characters],
                    metadata=segment.metadata,
                )
            )
            rewritten += 1
        merged = sorted(kept_segments + processed, key=lambda seg: seg.segment_id)
        content = "\n\n".join(seg.content.strip() for seg in merged)
        ratio = len(content) / max(len(text), 1)
        return RemoveResult(
            content=content,
            original_ratio=ratio,
            deleted_segments=deleted,
            rewritten_segments=rewritten,
            replaced_segments=0,
        )

    def _rewrite_or_delete(
        self,
        segment: Segment,
        characters: list[str],
        context_block: str,
        mode: str,
        style_guide: str,
    ) -> tuple[str, str]:
        """Ask the LLM whether to delete or rewrite an affected segment."""
        prompt = fill_template(
            self.rewrite_template,
            context_block=context_block,
            mode=mode,
            characters=", ".join(characters),
            segment_text=segment.content.strip(),
            style_guide=style_guide,
        )
        response = self.llm.generate(prompt=prompt, options={"no_think": True})
        payload = load_json_response(response.content)
        if not isinstance(payload, dict):
            raise LLMResponseError("Character remover returned invalid JSON object.")
        return self._parse_action(payload)

    def _parse_action(self, payload: dict[str, Any]) -> tuple[str, str]:
        action_payload = payload.get("action")
        if not isinstance(action_payload, str):
            raise LLMResponseError("Character removal action must be a string.")
        action = action_payload.strip().lower()
        if action not in {"delete", "rewrite"}:
            raise LLMResponseError(f"Invalid character removal action: {action!r}")
        content_payload = payload.get("content")
        if content_payload is None:
            content = ""
        elif isinstance(content_payload, str):
            content = content_payload.strip()
        else:
            raise LLMResponseError("Character removal content must be a string or null.")
        if action == "rewrite" and not content:
            raise LLMResponseError("Character removal rewrite action requires content.")
        if action == "delete":
            content = ""
        return action, content
