from __future__ import annotations

import json
import logging
import re
from typing import Any

from story_for_you.analysis.context import ChapterSummary
from story_for_you.analysis.prompting import fill_template, load_template
from story_for_you.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class ChapterSummarizer:
    """LLM-backed chapter summarizer using structured prompt templates."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm
        self.template = load_template("chapter_summary")

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
        prompt = fill_template(
            self.template,
            chapter_meta=json.dumps(meta_payload, ensure_ascii=False),
            recent_context=recent_context.strip() or "暂无历史上下文。",
            chapter_text=chapter_text.strip(),
        )
        response = self.llm.generate(prompt=prompt)
        try:
            data = json.loads(response.content)
            return ChapterSummary(
                chapter=int(data.get("chapter", chapter_no)),
                title=(data.get("title") or meta_payload.get("title_hint") or f"Chapter {chapter_no}")[:80],
                pov=data.get("pov", "third-person"),
                beats=[str(item) for item in data.get("beats", [])][:6],
                mood=data.get("mood", "neutral"),
                synopsis=str(data.get("synopsis", "")).strip() or chapter_text[:200].strip(),
                irreversible_flags=[str(flag) for flag in data.get("irreversible_flags", [])],
            )
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning("Failed to parse chapter summary response: %s", exc)
            return self._fallback_summary(chapter_text, chapter_no)

    # Fallback heuristics preserved for robustness ---------------------------------
    def _fallback_summary(self, chapter_text: str, chapter_no: int) -> ChapterSummary:
        lines = [line.strip() for line in chapter_text.splitlines() if line.strip()]
        title = lines[0] if lines else f"Chapter {chapter_no}"
        pov = self._detect_pov(chapter_text)
        beats = self._extract_beats(chapter_text)
        mood = self._infer_mood(chapter_text)
        synopsis = " ".join(beats[:2]) if beats else chapter_text[:200]
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
