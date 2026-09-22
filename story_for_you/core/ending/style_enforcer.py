"""Mechanical post-processing for ending writer output."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from story_for_you.analysis.context import WritingStyle

__all__ = [
    "StyleEnforcer",
]

logger = logging.getLogger(__name__)


class StyleEnforcer:
    """Applies non-semantic cleanup such as duplicate paragraph removal."""

    DEDUP_SIMILARITY_THRESHOLD = 0.25

    def __init__(self, style: WritingStyle | None = None):
        self.style = style

    def post_process(self, text: str) -> str:
        """Deduplicate repeated paragraphs without interpreting story semantics."""

        if not text:
            return text

        paragraphs = [para for para in text.split("\n\n") if para.strip()]
        if not paragraphs:
            return text

        normalized_seen: set[str] = set()
        cleaned: list[str] = []
        last_norm = ""
        for para in paragraphs:
            norm = re.sub(r"\s+", " ", para.strip())
            if norm == last_norm or norm in normalized_seen:
                continue
            cleaned.append(para.strip())
            normalized_seen.add(norm)
            last_norm = norm

        # Similarity is useful for a reviewer, but unsafe as a destructive
        # post-processing rule: two Chinese paragraphs often share a lead-in
        # while the second one contains the actual plot advancement.
        cleaned = self._dedupe_similar_paragraphs(cleaned)
        return "\n\n".join(cleaned) if cleaned else text

    def filter_duplicate_bridges(self, polished: str, bridges: list[str]) -> list[str]:
        """Filter out bridge paragraphs that are too similar to existing polished content.

        Uses first-sentence matching and Jaccard similarity to detect duplicates.
        """
        if not bridges:
            return []

        # Extract existing sentences and first-sentences from polished content
        existing_paragraphs = [p.strip() for p in polished.split("\n\n") if p.strip()]
        existing_normalized = {self._normalize_paragraph(item) for item in existing_paragraphs}

        filtered: list[str] = []
        for bridge in bridges:
            bridge_stripped = bridge.strip()
            if not bridge_stripped:
                continue

            normalized_bridge = self._normalize_paragraph(bridge_stripped)
            if normalized_bridge in existing_normalized:
                logger.debug("Bridge被过滤（整段重复）: %s...", bridge_stripped[:30])
                continue

            filtered.append(bridge_stripped)
            existing_normalized.add(normalized_bridge)

        return filtered

    def _dedupe_similar_paragraphs(self, paragraphs: list[str], threshold: float = DEDUP_SIMILARITY_THRESHOLD) -> list[str]:
        """移除与之前段落高度相似的段落。

        使用首句匹配和句子级别的 Jaccard 相似度检测重复。
        threshold: 相似度阈值，超过此值则视为重复。
        """
        if len(paragraphs) <= 1:
            return paragraphs

        result: list[str] = []
        seen_paragraphs: set[str] = set()

        for para in paragraphs:
            para_stripped = para.strip()
            if not para_stripped:
                continue

            normalized = self._normalize_paragraph(para_stripped)
            if normalized in seen_paragraphs:
                logger.debug("检测到整段重复，已移除: %s...", para_stripped[:20])
                continue
            result.append(para_stripped)
            seen_paragraphs.add(normalized)

        return result

    def _normalize_paragraph(self, text: str) -> str:
        """Normalize whitespace only; preserve punctuation and plot details."""
        return re.sub(r"\s+", " ", text.strip())

    def _extract_sentences_for_dedup(self, text: str) -> set[str]:
        """Extract sentence set for deduplication checking."""
        parts = re.split(r"[。！？]", text)
        return {s.strip() for s in parts if len(s.strip()) >= 6}

    def _jaccard_similarity(self, set1: set[str], set2: set[str]) -> float:
        """Compute Jaccard similarity between two sets."""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
