from __future__ import annotations

import re

from story_for_you.analysis.context import ChapterSummary
from story_for_you.llm.base import LLMProvider


class ChapterSummarizer:
    """LLM-backed chapter summarizer placeholder."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def summarize(self, chapter_text: str, chapter_no: int) -> ChapterSummary:
        """Summarize the given chapter text."""
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
            "cry": "sad",
            "death": "tragic",
            "love": "hopeful",
            "笑": "light",
        }
        lowered = text.lower()
        for keyword, mood in heuristics.items():
            if keyword in lowered:
                return mood
        return "neutral"

    def _find_irreversible(self, text: str) -> list[str]:
        keywords = ["death", "married", "destroyed", "牺牲", "灭亡"]
        flags = []
        for keyword in keywords:
            if keyword in text.lower():
                flags.append(keyword)
        return flags
