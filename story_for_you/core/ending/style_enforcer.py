"""Style enforcement and post-processing for ending writer.

Provides deduplication, quality filtering, and text cleanup utilities.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from story_for_you.core.ending.constants import (
    LOW_QUALITY_PHRASES,
    QUESTION_MARKERS,
)

if TYPE_CHECKING:
    from story_for_you.analysis.context import WritingStyle

__all__ = [
    "StyleEnforcer",
]

logger = logging.getLogger(__name__)


class StyleEnforcer:
    """Enforces style consistency and quality in generated text."""

    def __init__(self, style: WritingStyle | None = None):
        self.style = style

    def post_process(self, text: str) -> str:
        """Deduplicate重复段落，强制过滤低质量短语，保留终端标记。"""

        if not text:
            return text
        marker = "（读者定制版本）"
        working = text.rstrip()
        marker_present = working.endswith(marker)
        if marker_present:
            working = working[: -len(marker)].rstrip()

        paragraphs = [para for para in working.split("\n\n") if para.strip()]
        if not paragraphs:
            return text

        # 第一步：完全相同段落去重
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

        # 第二步：高相似度段落去重（基于句子重叠）
        cleaned = self._dedupe_similar_paragraphs(cleaned)

        # 第三步：对文学/古典风格强制过滤低质量短语
        register = getattr(self.style, "register", "mixed").lower() if self.style else "mixed"
        if register in ("literary", "classical"):
            cleaned = self._filter_low_quality_sentences(cleaned)

        # 第四步：过滤非对话问句段落（可能是调试残余或草稿提问）
        cleaned = self._filter_question_paragraphs(cleaned)

        combined = "\n\n".join(cleaned) if cleaned else working

        if marker_present:
            combined = combined.rstrip() + "\n\n" + marker
        return combined

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
            if self._jaccard_similarity(bridge_sentences, existing_sentences) >= 0.25:
                logger.debug("Bridge被过滤（相似度过高）: %s...", bridge_stripped[:30])
                continue

            filtered.append(bridge_stripped)
            # Add this bridge's sentences to existing set to avoid inter-bridge duplicates
            existing_sentences.update(bridge_sentences)
            if bridge_first:
                existing_first_sentences.add(bridge_first)

        return filtered

    def _filter_low_quality_sentences(self, paragraphs: list[str]) -> list[str]:
        """从段落中移除包含低质量短语的句子。"""
        result: list[str] = []
        for para in paragraphs:
            # 按句号分割
            sentences = re.split(r"(。|！|？)", para)
            filtered_parts: list[str] = []
            i = 0
            while i < len(sentences):
                sentence = sentences[i]
                # 保留标点符号
                punct = sentences[i + 1] if i + 1 < len(sentences) and sentences[i + 1] in "。！？" else ""

                # 检查是否包含低质量短语
                has_bad_phrase = any(phrase in sentence for phrase in LOW_QUALITY_PHRASES)
                if has_bad_phrase:
                    logger.info("移除低质量句子: %s", sentence[:30] + "...")
                elif sentence.strip():
                    filtered_parts.append(sentence + punct)

                i += 2 if punct else 1

            # 重新组合段落
            if filtered_parts:
                new_para = "".join(filtered_parts).strip()
                if new_para:
                    result.append(new_para)
        return result

    def _filter_question_paragraphs(self, paragraphs: list[str]) -> list[str]:
        """过滤以问号结尾的非对话段落。

        这类段落通常是调试残余、草稿提问或LLM的元叙述，不应出现在正文中。
        保留对话中的问句（引号内的问句）。
        """
        result: list[str] = []
        for para in paragraphs:
            para_stripped = para.strip()
            if not para_stripped:
                continue

            # 检查段落是否以问号结尾（非对话）
            if para_stripped.endswith("？"):
                # 检查是否是对话（引号内的内容）
                # 对话格式：..."xxx？" 或 "xxx？"
                is_dialogue = (
                    para_stripped.endswith('"？"')
                    or para_stripped.endswith('"')
                    or '："' in para_stripped
                    or ':"' in para_stripped
                )
                if not is_dialogue:
                    # 检查是否包含问句特征词
                    has_question_marker = any(marker in para_stripped for marker in QUESTION_MARKERS)
                    if has_question_marker:
                        logger.debug("移除非对话问句段落: %s...", para_stripped[:30])
                        continue

            result.append(para_stripped)
        return result

    def _dedupe_similar_paragraphs(self, paragraphs: list[str], threshold: float = 0.2) -> list[str]:
        """移除与之前段落高度相似的段落。

        使用首句匹配和句子级别的 Jaccard 相似度检测重复。
        threshold: 相似度阈值，超过此值则视为重复（默认 0.2）。
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
