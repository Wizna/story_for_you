from __future__ import annotations

import json
import logging
import re
from typing import Any

from story_for_you.analysis.context import ChapterSummary
from story_for_you.analysis.prompting import load_template, render_prompt_with_budget
from story_for_you.llm.base import LLMProvider
from story_for_you.utils.json_utils import load_json_response

logger = logging.getLogger(__name__)


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
        response = self.llm.generate(prompt=prompt)
        try:
            data = load_json_response(response.content)
            if not isinstance(data, dict):
                raise ValueError("Chapter summary response is not a JSON object.")
            title = self._clamp_text(
                data.get("title") or meta_payload.get("title_hint") or f"Chapter {chapter_no}",
                self.MAX_TITLE_LEN,
            )
            beats = [
                self._clamp_text(str(item), self.MAX_BEAT_LEN) for item in data.get("beats", [])
            ][:6]
            synopsis = self._clamp_text(
                str(data.get("synopsis", "")).strip() or chapter_text[:200].strip(),
                self.MAX_SYNOPSIS_LEN,
            )
            return ChapterSummary(
                chapter=int(data.get("chapter", chapter_no)),
                title=title,
                pov=data.get("pov", "third-person"),
                beats=beats,
                mood=data.get("mood", "neutral"),
                synopsis=synopsis,
                irreversible_flags=[str(flag) for flag in data.get("irreversible_flags", [])],
            )
        except (TypeError, ValueError) as exc:
            logger.warning("Failed to parse chapter summary response: %s", exc)
            return self._fallback_summary(chapter_text, chapter_no)

    # Fallback heuristics preserved for robustness ---------------------------------
    def _fallback_summary(self, chapter_text: str, chapter_no: int) -> ChapterSummary:
        lines = [line.strip() for line in chapter_text.splitlines() if line.strip()]
        title = self._clamp_text(lines[0] if lines else f"Chapter {chapter_no}", self.MAX_TITLE_LEN)
        pov = self._detect_pov(chapter_text)
        beats = [self._clamp_text(item, self.MAX_BEAT_LEN) for item in self._extract_beats(chapter_text)]
        mood = self._infer_mood(chapter_text)
        synopsis_source = " ".join(beats[:2]) if beats else chapter_text[:200]
        synopsis = self._clamp_text(synopsis_source, self.MAX_SYNOPSIS_LEN)
        irreversible_flags = self._find_irreversible(chapter_text)
        return ChapterSummary(
            chapter=chapter_no,
            title=title,
            pov=pov,
            beats=beats[:4],
            mood=mood,
            synopsis=synopsis.strip(),
            irreversible_flags=irreversible_flags,
        )

    def _detect_pov(self, text: str) -> str:
        first_person = sum(text.count(token) for token in [" I ", " I'm", " my ", " me "])
        third_person = sum(text.count(token) for token in [" he ", " she ", "他们", "他们的"])
        return "first-person" if first_person > third_person else "third-person"

    def _extract_beats(self, text: str) -> list[str]:
        sentences = re.split(r"(?<=[。.!?])\s+", text.strip())
        beats = [sentence.strip() for sentence in sentences if sentence.strip()]
        return beats[:6]

    def _infer_mood(self, text: str) -> str:
        heuristics = {
            "battle": "tense",
            "cry": "somber",
            "death": "somber",
            "love": "hopeful",
            "笑": "whimsical",
        }
        lowered = text.lower()
        for keyword, mood in heuristics.items():
            if keyword in lowered:
                return mood
        return "neutral"

    def _find_irreversible(self, text: str) -> list[str]:
        keywords = ["death", "married", "destroyed", "牺牲", "灭亡"]
        flags = []
        lowered = text.lower()
        for keyword in keywords:
            if keyword in lowered:
                flags.append(keyword)
        return flags

    def _clamp_text(self, text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[: max(limit - 3, 0)].rstrip() + "..."
