from __future__ import annotations

import logging
from typing import Any

from story_for_you.analysis.context import ChapterSummary, StyleSample, WritingStyle
from story_for_you.analysis.prompting import fill_template, load_template
from story_for_you.core.exceptions import LLMError
from story_for_you.llm.base import LLMProvider
from story_for_you.utils.json_utils import load_json_response

logger = logging.getLogger(__name__)


class StyleExtractor:
    """Extracts writing style characteristics from chapter samples."""

    SAMPLE_CHAPTERS = 3  # 首、中、尾章节
    SAMPLE_SIZE = 2000  # 每样本字符数

    def __init__(self, llm: LLMProvider, prompt_budget: int | None = None):
        self.llm = llm
        self.prompt_budget = prompt_budget
        self.template = load_template("style_extraction")

    def extract(self, chapters: list[str], summaries: list[ChapterSummary]) -> WritingStyle:
        """从已分析的章节摘要中提取风格。"""
        samples = self._select_samples(chapters)
        pov_summary = self._summarize_pov(summaries)
        mood_summary = self._summarize_mood(summaries)
        prompt = self._build_prompt(samples, pov_summary, mood_summary)
        return self._execute_and_parse(prompt)

    def _select_samples(self, chapters: list[str]) -> list[tuple[int, str]]:
        """选取代表性章节样本：首、中、尾。"""
        if not chapters:
            return []
        samples: list[tuple[int, str]] = []
        indices = self._sample_indices(len(chapters))
        for idx in indices:
            content = chapters[idx].strip()
            if len(content) > self.SAMPLE_SIZE:
                content = content[: self.SAMPLE_SIZE]
            samples.append((idx + 1, content))
        return samples

    def _sample_indices(self, total: int) -> list[int]:
        """根据章节总数选取样本索引。"""
        if total <= 0:
            return []
        if total == 1:
            return [0]
        if total == 2:
            return [0, 1]
        # 首、中、尾
        mid = total // 2
        near_end = min(total - 1, int(total * 0.8))
        return [0, mid, near_end]

    def _summarize_pov(self, summaries: list[ChapterSummary]) -> str:
        """汇总章节的叙事视角。"""
        if not summaries:
            return "未知"
        povs = [s.pov for s in summaries if s.pov]
        if not povs:
            return "未知"
        from collections import Counter

        counter = Counter(povs)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def _summarize_mood(self, summaries: list[ChapterSummary]) -> str:
        """汇总章节的情绪基调。"""
        if not summaries:
            return "未知"
        moods = [s.mood for s in summaries if s.mood]
        if not moods:
            return "未知"
        from collections import Counter

        counter = Counter(moods)
        top_moods = counter.most_common(3)
        return ", ".join(m[0] for m in top_moods)

    def _build_prompt(self, samples: list[tuple[int, str]], pov: str, mood: str) -> str:
        """构建提示词。"""
        chapter_samples = self._format_samples(samples)
        return fill_template(
            self.template,
            chapter_samples=chapter_samples,
            pov_summary=pov,
            mood_summary=mood,
        )

    def _format_samples(self, samples: list[tuple[int, str]]) -> str:
        """格式化样本为提示词文本。"""
        if not samples:
            return "(无可用样本)"
        parts = []
        for chapter_num, content in samples:
            parts.append(f"### 第 {chapter_num} 章节样本\n{content}")
        return "\n\n".join(parts)

    def _execute_and_parse(self, prompt: str) -> WritingStyle:
        """执行 LLM 调用并解析结果。"""
        try:
            response = self.llm.generate(prompt=prompt)
            return self._parse_response(response.content)
        except LLMError as exc:
            logger.warning("Style extraction failed, using fallback: %s", exc)
            return self._fallback_style()

    def _parse_response(self, content: str) -> WritingStyle:
        """解析 LLM 返回的 JSON 响应。"""
        payload = load_json_response(content)
        if payload is None:
            logger.warning("Style extractor returned invalid JSON, using fallback.")
            return self._fallback_style()
        return self._payload_to_style(payload)

    def _payload_to_style(self, data: dict[str, Any]) -> WritingStyle:
        """将 JSON payload 转换为 WritingStyle 对象。"""
        samples = []
        for item in data.get("representative_samples", [])[:3]:
            if isinstance(item, dict):
                samples.append(
                    StyleSample(
                        source_chapter=item.get("source_chapter", 0),
                        content=str(item.get("content", "")),
                        style_notes=str(item.get("style_notes", "")),
                    )
                )
        return WritingStyle(
            avg_sentence_length=int(data.get("avg_sentence_length", 20)),
            sentence_variety=str(data.get("sentence_variety", "mixed")),
            paragraph_density=str(data.get("paragraph_density", "medium")),
            register=str(data.get("register", "literary")),
            characteristic_words=self._normalize_list(data.get("characteristic_words", [])),
            idiom_frequency=str(data.get("idiom_frequency", "sparse")),
            metaphor_style=str(data.get("metaphor_style", "")),
            description_focus=self._normalize_list(data.get("description_focus", [])),
            parallelism_use=str(data.get("parallelism_use", "rare")),
            tone_markers=self._normalize_list(data.get("tone_markers", [])),
            narrator_style=str(data.get("narrator_style", "detached")),
            representative_samples=samples,
            style_summary=str(data.get("style_summary", "")),
        )

    def _normalize_list(self, value: Any) -> list[str]:
        """确保返回字符串列表。"""
        if isinstance(value, list):
            return [str(item) for item in value if item][:8]
        if isinstance(value, str):
            return [value]
        return []

    def _fallback_style(self) -> WritingStyle:
        """返回默认风格（当 LLM 调用失败时）。"""
        return WritingStyle(
            avg_sentence_length=20,
            sentence_variety="mixed",
            paragraph_density="medium",
            register="literary",
            characteristic_words=[],
            idiom_frequency="sparse",
            metaphor_style="",
            description_focus=["landscape", "psychological"],
            parallelism_use="rare",
            tone_markers=[],
            narrator_style="detached",
            representative_samples=[],
            style_summary="（风格分析失败，请重新尝试）",
        )
