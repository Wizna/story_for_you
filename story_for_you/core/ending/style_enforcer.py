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
        existing_sentences = self._extract_sentences_for_dedup(polished)
        existing_first_sentences: set[str] = set()
        for para in existing_paragraphs:
            first = para.split("。")[0].strip()
            if first:
                existing_first_sentences.add(first)

        filtered: list[str] = []
        for bridge in bridges:
            bridge_stripped = bridge.strip()
            if not bridge_stripped:
                continue

            # Check first-sentence overlap
            bridge_first = bridge_stripped.split("。")[0].strip()
            if bridge_first and bridge_first in existing_first_sentences:
                logger.debug("Bridge被过滤（首句重复）: %s...", bridge_first[:30])
                continue

            # Check Jaccard similarity with existing content
            bridge_sentences = self._extract_sentences_for_dedup(bridge_stripped)
            if self._jaccard_similarity(bridge_sentences, existing_sentences) >= self.DEDUP_SIMILARITY_THRESHOLD:
                logger.debug("Bridge被过滤（相似度过高）: %s...", bridge_stripped[:30])
                continue

            filtered.append(bridge_stripped)
            # Add this bridge's sentences to existing set to avoid inter-bridge duplicates
            existing_sentences.update(bridge_sentences)
            if bridge_first:
                existing_first_sentences.add(bridge_first)

        return filtered

    def _dedupe_similar_paragraphs(self, paragraphs: list[str], threshold: float = DEDUP_SIMILARITY_THRESHOLD) -> list[str]:
        """移除与之前段落高度相似的段落。

        使用首句匹配和句子级别的 Jaccard 相似度检测重复。
        threshold: 相似度阈值，超过此值则视为重复。
        """
        if len(paragraphs) <= 1:
            return paragraphs

        def extract_sentences(text: str) -> set[str]:
            """提取段落中的句子集合。"""
            parts = re.split(r"[。！？]", text)
            return {s.strip() for s in parts if len(s.strip()) >= 6}

        def extract_first_sentence(text: str) -> str:
            """提取段落的首句（用于快速去重）。"""
            first = text.split("。")[0].strip()
            # 如果首句过短，尝试取更长的开头
            if len(first) < 10:
                first = text[:30].strip()
            return first

        result: list[str] = []
        seen_sentences: list[set[str]] = []
        seen_first_sentences: set[str] = set()

        for para in paragraphs:
            para_stripped = para.strip()
            if not para_stripped:
                continue

            # 首句去重：相同首句直接视为重复
            first_sentence = extract_first_sentence(para_stripped)
            if first_sentence and first_sentence in seen_first_sentences:
                logger.debug("检测到首句重复段落，已移除: %s...", first_sentence[:20])
                continue

            current_sentences = extract_sentences(para_stripped)
            is_duplicate = False

            for prev_sentences in seen_sentences:
                similarity = self._jaccard_similarity(current_sentences, prev_sentences)
                if similarity >= threshold:
                    logger.debug("检测到相似段落（相似度 %.2f），已移除", similarity)
                    is_duplicate = True
                    break

            if not is_duplicate:
                result.append(para_stripped)
                seen_sentences.append(current_sentences)
                if first_sentence:
                    seen_first_sentences.add(first_sentence)

        return result

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
