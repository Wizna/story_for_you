from __future__ import annotations

import json
import logging
from typing import Any

from story_for_you.analysis.context import ChapterSummary
from story_for_you.analysis.prompting import (
    clamp_text_middle,
    fill_template,
    load_template,
    render_prompt_with_budget,
)
from story_for_you.core.exceptions import LLMResponseError
from story_for_you.llm.base import LLMProvider
from story_for_you.llm.telemetry import telemetry_options
from story_for_you.utils.json_utils import load_json_response

logger = logging.getLogger(__name__)

_MAX_BEATS = 6
_MAX_CHAPTER_ATTEMPTS = 2
_REPAIR_SNIPPET_BUDGET = 4000
_JSON_OBJECT_OPTIONS = {
    "no_think": True,
    "temperature": 0.1,
    "response_format": {"type": "json_object"},
}
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
        self.repair_template = load_template("chapter_summary_repair")
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
        last_error: str | None = None
        for attempt in range(1, _MAX_CHAPTER_ATTEMPTS + 1):
            response = self.llm.generate(
                prompt=prompt,
                options=telemetry_options(
                    _JSON_OBJECT_OPTIONS,
                    phase=f"analyze chapter {chapter_no}",
                    step=": chapter summary",
                    attempt=attempt,
                    max_attempts=_MAX_CHAPTER_ATTEMPTS,
                ),
            )
            summary, error = self._parse_response(response.content)
            if summary is not None:
                return summary
            last_error = error
            if error:
                logger.debug("Chapter summary parse failed (%s). Attempting repair.", error)
            repaired_content = self._attempt_repair(response.content, error, chapter_no)
            if repaired_content:
                summary, repair_error = self._parse_response(repaired_content)
                if summary is not None:
                    return summary
                last_error = repair_error or last_error
            if attempt < _MAX_CHAPTER_ATTEMPTS:
                logger.debug("Retrying chapter summary after schema failure: %s", last_error)
        raise LLMResponseError(f"Chapter summary failed after retry: {last_error or 'unknown parse failure'}")

    def _parse_response(self, content: str | None) -> tuple[ChapterSummary | None, str | None]:
        if not content:
            return None, "Empty response body."
        data = load_json_response(content)
        if not isinstance(data, dict):
            return None, "Chapter summary response is not a JSON object."
        try:
            return self._from_payload(data), None
        except LLMResponseError as exc:
            return None, str(exc)

    def _from_payload(self, data: dict[str, Any]) -> ChapterSummary:
        for field_name in ("chapter", "title", "pov", "beats", "mood", "synopsis", "irreversible_flags"):
            if field_name not in data:
                raise LLMResponseError(f"Chapter summary response missing required field: {field_name}")
        title_text = self._required_str(data.get("title"), "title")
        if not title_text:
            raise LLMResponseError("Chapter summary response is missing 'title'.")
        title = self._clamp_text(title_text, self.MAX_TITLE_LEN)
        beats_payload = data.get("beats", [])
        if not isinstance(beats_payload, list):
            raise LLMResponseError("Chapter summary 'beats' must be a list.")
        beats = [
            self._clamp_text(item.strip(), self.MAX_BEAT_LEN)
            for item in beats_payload
            if self._valid_str_item(item, "beats")
        ][:_MAX_BEATS]
        irreversible_payload = data.get("irreversible_flags")
        if not isinstance(irreversible_payload, list):
            raise LLMResponseError("Chapter summary 'irreversible_flags' must be a list.")
        synopsis_text = self._required_str(data.get("synopsis"), "synopsis")
        if not synopsis_text:
            raise LLMResponseError("Chapter summary response is missing 'synopsis'.")
        synopsis = self._clamp_text(synopsis_text, self.MAX_SYNOPSIS_LEN)
        pov = self._required_str(data.get("pov"), "pov")
        mood = self._required_str(data.get("mood"), "mood")
        if pov not in _VALID_POVS:
            raise LLMResponseError(f"Invalid chapter pov: {pov!r}")
        if mood not in _VALID_MOODS:
            raise LLMResponseError(f"Invalid chapter mood: {mood!r}")
        chapter_value = data.get("chapter")
        if not isinstance(chapter_value, int) or isinstance(chapter_value, bool):
            raise LLMResponseError("Chapter summary 'chapter' must be an integer.")
        return ChapterSummary(
            chapter=chapter_value,
            title=title,
            pov=pov,
            beats=beats,
            mood=mood,
            synopsis=synopsis,
            irreversible_flags=[
                item.strip()
                for item in irreversible_payload
                if self._valid_str_item(item, "irreversible_flags")
            ],
        )

    def _required_str(self, value: Any, field_name: str) -> str:
        if not isinstance(value, str):
            raise LLMResponseError(f"Chapter summary '{field_name}' must be a string.")
        return value.strip()

    def _valid_str_item(self, value: Any, field_name: str) -> bool:
        if not isinstance(value, str):
            raise LLMResponseError(f"Chapter summary '{field_name}' items must be strings.")
        return bool(value.strip())

    def _clamp_text(self, text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[: max(limit - 3, 0)].rstrip() + "..."

    def _attempt_repair(self, raw_response: str | None, error: str | None, chapter_no: int) -> str | None:
        """Ask the LLM to repair malformed chapter-summary JSON once."""
        if not raw_response:
            return None
        snippet = clamp_text_middle(raw_response, _REPAIR_SNIPPET_BUDGET)
        prompt = fill_template(
            self.repair_template,
            error_message=error or "JSON parsing failed.",
            invalid_output=snippet,
        )
        repaired = self.llm.generate(
            prompt=prompt,
            options=telemetry_options(
                _JSON_OBJECT_OPTIONS,
                phase=f"analyze chapter {chapter_no}",
                step=": repair chapter summary JSON",
            ),
        )
        return repaired.content
