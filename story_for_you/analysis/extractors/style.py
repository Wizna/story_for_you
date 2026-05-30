from __future__ import annotations

import logging
from typing import Any

from story_for_you.analysis.context import ChapterSummary, StyleSample, WritingStyle
from story_for_you.analysis.prompting import fill_template, load_template
from story_for_you.core.exceptions import LLMResponseError
from story_for_you.llm.base import LLMProvider
from story_for_you.llm.telemetry import telemetry_options
from story_for_you.utils.json_utils import load_json_response

logger = logging.getLogger(__name__)

_MAX_STYLE_LIST_ITEMS = 8
_STRUCTURED_OPTIONS = {"no_think": True, "temperature": 0.1}


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
            content = self._sample_body_text(chapters[idx].strip(), position=idx, total=len(chapters))
            samples.append((idx + 1, content))
        return samples

    def _sample_body_text(self, content: str, *, position: int, total: int) -> str:
        """Sample narrative body text, avoiding front matter when possible."""
        if not content:
            return ""
        cleaned = content
        if len(cleaned) <= self.SAMPLE_SIZE:
            return cleaned
        if total == 1:
            start = max(0, len(cleaned) // 2 - self.SAMPLE_SIZE // 2)
            return cleaned[start : start + self.SAMPLE_SIZE]
        if position == 0:
            return cleaned[: self.SAMPLE_SIZE]
        if position >= total - 1:
            return cleaned[-self.SAMPLE_SIZE :]
        start = max(0, len(cleaned) // 2 - self.SAMPLE_SIZE // 2)
        return cleaned[start : start + self.SAMPLE_SIZE]

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
        response = self.llm.generate(
            prompt=prompt,
            options=telemetry_options(
                _STRUCTURED_OPTIONS,
                phase="analyze",
                step=": extract writing style",
            ),
        )
        return self._parse_response(response.content)

    def _parse_response(self, content: str) -> WritingStyle:
        """解析 LLM 返回的 JSON 响应。"""
        payload = load_json_response(content)
        if not isinstance(payload, dict):
            raise LLMResponseError("Style extractor returned invalid JSON object.")
        return self._payload_to_style(payload)

    def _payload_to_style(self, data: dict[str, Any]) -> WritingStyle:
        """将 JSON payload 转换为 WritingStyle 对象。"""
        for field_name in (
            "avg_sentence_length",
            "sentence_variety",
            "paragraph_density",
            "register",
            "characteristic_words",
            "idiom_frequency",
            "metaphor_style",
            "description_focus",
            "parallelism_use",
            "tone_markers",
            "narrator_style",
            "representative_samples",
            "style_summary",
        ):
            if field_name not in data:
                raise LLMResponseError(f"Style extractor response missing required field: {field_name}")
        samples = []
        sample_payload = data.get("representative_samples")
        if not isinstance(sample_payload, list):
            raise LLMResponseError("Style representative_samples must be a list.")
        for item in sample_payload[:3]:
            if not isinstance(item, dict):
                raise LLMResponseError("Style sample item must be an object.")
            for field_name in ("source_chapter", "content", "style_notes"):
                if field_name not in item:
                    raise LLMResponseError(f"Style sample missing required field: {field_name}")
            samples.append(
                StyleSample(
                    source_chapter=self._required_int(item.get("source_chapter"), "source_chapter"),
                    content=self._required_str(item.get("content"), "content", allow_empty=True),
                    style_notes=self._required_str(item.get("style_notes"), "style_notes", allow_empty=True),
                )
            )
        for field_name in ("characteristic_words", "description_focus", "tone_markers"):
            if not isinstance(data.get(field_name), list):
                raise LLMResponseError(f"Style field must be a list: {field_name}")
        return WritingStyle(
            avg_sentence_length=self._required_int(data.get("avg_sentence_length"), "avg_sentence_length"),
            sentence_variety=self._required_str(data.get("sentence_variety"), "sentence_variety"),
            paragraph_density=self._required_str(data.get("paragraph_density"), "paragraph_density"),
            register=self._required_str(data.get("register"), "register"),
            characteristic_words=self._normalize_list(data.get("characteristic_words")),
            idiom_frequency=self._required_str(data.get("idiom_frequency"), "idiom_frequency"),
            metaphor_style=self._required_str(data.get("metaphor_style"), "metaphor_style", allow_empty=True),
            description_focus=self._normalize_list(data.get("description_focus")),
            parallelism_use=self._required_str(data.get("parallelism_use"), "parallelism_use"),
            tone_markers=self._normalize_list(data.get("tone_markers")),
            narrator_style=self._required_str(data.get("narrator_style"), "narrator_style"),
            representative_samples=samples,
            style_summary=self._required_str(data.get("style_summary"), "style_summary", allow_empty=True),
        )

    def _required_int(self, value: Any, field_name: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise LLMResponseError(f"Style field must be an integer: {field_name}")
        return value

    def _required_str(self, value: Any, field_name: str, *, allow_empty: bool = False) -> str:
        if not isinstance(value, str):
            raise LLMResponseError(f"Style field must be a string: {field_name}")
        text = value.strip()
        if not allow_empty and not text:
            raise LLMResponseError(f"Style field must not be empty: {field_name}")
        return text

    def _normalize_list(self, value: Any) -> list[str]:
        """确保返回字符串列表。"""
        if not isinstance(value, list):
            raise LLMResponseError("Style list field must be a JSON array.")
        items: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise LLMResponseError("Style list items must be strings.")
            text = item.strip()
            if text:
                items.append(text)
        return items[:_MAX_STYLE_LIST_ITEMS]
