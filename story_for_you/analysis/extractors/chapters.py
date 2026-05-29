from __future__ import annotations

import json
import logging
from typing import Any

from story_for_you.analysis.context import ChapterSummary
from story_for_you.analysis.prompting import load_template, render_prompt_with_budget
from story_for_you.core.exceptions import LLMResponseError
from story_for_you.llm.base import LLMProvider
from story_for_you.utils.json_utils import load_json_response

logger = logging.getLogger(__name__)

_MAX_BEATS = 6
_VALID_POVS = {"first-person", "third-person", "omniscient", "multi-pov", "unknown"}
_VALID_MOODS = {"tense", "hopeful", "somber", "whimsical", "neutral", "unknown"}


class ChapterSummarizer:
    """LLM-backed chapter summarizer using structured prompt templates."""

    MAX_TITLE_LEN = 80
    MAX_BEAT_LEN = 160
    MAX_SYNOPSIS_LEN = 360

    def __init__(self, llm: LLMProvider, prompt_budget: int | None = None):
        self.llm = llm
        self.template = load_template("chapter_summary")
        self.prompt_budget = prompt_budget

    def summarize(
        self,
        chapter_text: str,
        chapter_no: int,
        recent_context: str,
        chapter_meta: dict[str, Any] | None = None,
    ) -> ChapterSummary:
        """Summarize the given chapter text via the configured LLM."""
        meta_payload = {"chapter_no": chapter_no}
        if chapter_meta:
            meta_payload.update(chapter_meta)
        recent_context_text = recent_context.strip() or "暂无历史上下文。"
        chapter_body = chapter_text.strip()
        prompt, truncated = render_prompt_with_budget(
            self.template,
            budget=self.prompt_budget,
            text_key="chapter_text",
            text_value=chapter_body,
            chapter_meta=json.dumps(meta_payload, ensure_ascii=False),
            recent_context=recent_context_text,
        )
        if truncated:
            logger.debug("Chapter summary prompt truncated to %s chars", len(prompt))
        response = self.llm.generate(prompt=prompt, options={"no_think": True})
        data = load_json_response(response.content)
        if not isinstance(data, dict):
            raise LLMResponseError("Chapter summary response is not a JSON object.")
        for field_name in ("chapter", "title", "pov", "beats", "mood", "synopsis", "irreversible_flags"):
            if field_name not in data:
                raise LLMResponseError(f"Chapter summary response missing required field: {field_name}")
        title_text = str(data.get("title", "")).strip()
        if not title_text:
            raise LLMResponseError("Chapter summary response is missing 'title'.")
        title = self._clamp_text(title_text, self.MAX_TITLE_LEN)
        beats_payload = data.get("beats", [])
        if not isinstance(beats_payload, list):
            raise LLMResponseError("Chapter summary 'beats' must be a list.")
        beats = [
            self._clamp_text(str(item), self.MAX_BEAT_LEN) for item in beats_payload
        ][:_MAX_BEATS]
        irreversible_payload = data.get("irreversible_flags")
        if not isinstance(irreversible_payload, list):
            raise LLMResponseError("Chapter summary 'irreversible_flags' must be a list.")
        synopsis_text = str(data.get("synopsis", "")).strip()
        if not synopsis_text:
            raise LLMResponseError("Chapter summary response is missing 'synopsis'.")
        synopsis = self._clamp_text(synopsis_text, self.MAX_SYNOPSIS_LEN)
        pov = str(data.get("pov")).strip()
        mood = str(data.get("mood")).strip()
        if pov not in _VALID_POVS:
            raise LLMResponseError(f"Invalid chapter pov: {pov!r}")
        if mood not in _VALID_MOODS:
            raise LLMResponseError(f"Invalid chapter mood: {mood!r}")
        return ChapterSummary(
            chapter=int(data.get("chapter", chapter_no)),
            title=title,
            pov=pov,
            beats=beats,
            mood=mood,
            synopsis=synopsis,
            irreversible_flags=[str(flag) for flag in irreversible_payload],
        )

    def _clamp_text(self, text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[: max(limit - 3, 0)].rstrip() + "..."
