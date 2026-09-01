from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from story_for_you.analysis.context import StoryContext
from story_for_you.config.settings import RenderingLimits
from story_for_you.core.exceptions import LLMResponseError
from story_for_you.indexer.retriever import SegmentRetriever
from story_for_you.indexer.segment import Segment, deduplicate_overlapping_segments
from story_for_you.llm.base import LLMProvider
from story_for_you.llm.telemetry import telemetry_options
from story_for_you.utils.prompting import cache_prompt
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
        if mode not in {"hard", "soft"}:
            raise LLMResponseError(f"Invalid character removal mode: {mode!r}")
        kept_segments = self.retriever.retrieve_excluding(exclude=characters, mode="hard")
        affected_segments = self.retriever.retrieve_by_characters(include=characters, mode="strict")
        processed: list[Segment] = []
        deleted = rewritten = 0
        context_block = format_context_sections(context.for_prompt(limits=self._limits))
        style_guide = format_style_guide(context.writing_style)
        forbidden_labels = self._forbidden_labels(characters, context)
        for segment in affected_segments:
            action, content = self._rewrite_or_delete(
                segment,
                characters,
                context_block,
                mode,
                style_guide,
                forbidden_labels=forbidden_labels,
            )
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
        merged = deduplicate_overlapping_segments(kept_segments + processed)
        content = "\n\n".join(seg.content.strip() for seg in merged)
        ratio = len(content) / max(len(text), 1)
        return RemoveResult(
            content=content,
            original_ratio=ratio,
            deleted_segments=deleted,
            rewritten_segments=rewritten,
            replaced_segments=0,
        )

    def _forbidden_labels(self, characters: list[str], context: StoryContext) -> list[str]:
        """Return target names and known aliases for hard-mode output checks."""
        labels: set[str] = {name.strip() for name in characters if name.strip()}
        lowered_targets = {name.casefold() for name in labels}
        for character in context.characters.values():
            known = {character.name, *character.aliases}
            if character.name.casefold() in lowered_targets or any(
                alias.casefold() in lowered_targets for alias in character.aliases
            ):
                labels.update(item.strip() for item in known if item and item.strip())
        return sorted(labels, key=len, reverse=True)

    def _rewrite_or_delete(
        self,
        segment: Segment,
        characters: list[str],
        context_block: str,
        mode: str,
        style_guide: str,
        *,
        forbidden_labels: list[str],
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
        response = self.llm.generate(
            prompt=cache_prompt(prompt),
            options=telemetry_options(
                {"no_think": True},
                phase="remove",
                step=f": decide rewrite/delete for segment {segment.segment_id}",
            ),
        )
        payload = load_json_response(response.content)
        if not isinstance(payload, dict):
            raise LLMResponseError("Character remover returned invalid JSON object.")
        action, content = self._parse_action(payload)
        if mode == "hard" and action == "rewrite":
            lowered_content = content.casefold()
            remaining = [label for label in forbidden_labels if label.casefold() in lowered_content]
            if remaining:
                raise LLMResponseError(
                    "Hard character removal rewrite still mentions target character(s): "
                    + ", ".join(remaining)
                )
        return action, content

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
