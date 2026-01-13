from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from story_for_you.analysis.context import StoryContext, WritingStyle
from story_for_you.indexer.segment import SegmentIndex
from story_for_you.llm.base import LLMProvider
from story_for_you.core.prompting import (
    fill_template,
    format_context_sections,
    format_style_guide,
    format_style_samples,
    load_template,
)
from story_for_you.utils.json_utils import load_json_response

logger = logging.getLogger(__name__)


@dataclass
class EndingOutline:
    """续写大纲结构"""

    theme: str = ""
    ending_direction: str = ""  # HE/BE/OE
    key_beats: list[str] = field(default_factory=list)
    emotional_arc: str = ""
    final_image: str = ""


@dataclass
class MultiStageEndingResult:
    """多阶段续写结果"""

    outline: EndingOutline
    draft: str
    final_content: str
    revision_notes: list[str] = field(default_factory=list)


class EndingWriter:
    """多阶段续写器，模拟人类作者创作流程。"""

    def __init__(self, llm: LLMProvider, segment_index: SegmentIndex):
        self.llm = llm
        self.segment_index = segment_index
        self._load_templates()

    def _load_templates(self) -> None:
        """Load all stage templates with fallback to legacy template."""
        try:
            self.inspiration_template = load_template("ending_inspiration")
            self.outline_template = load_template("ending_outline")
            self.draft_template = load_template("ending_draft")
            self.revision_template = load_template("ending_revision")
            self.polish_template = load_template("ending_polish")
            self._multi_stage_enabled = True
        except FileNotFoundError:
            # Fallback to legacy single-stage template
            self.legacy_template = load_template("ending")
            self._multi_stage_enabled = False
            logger.info("Multi-stage templates not found, using legacy single-stage mode")

    def continue_story(self, text: str, context: StoryContext, hint: str = "") -> str:
        """执行完整的多阶段续写流程。"""
        if not self._multi_stage_enabled:
            return self._legacy_continue(text, context, hint)

        style = context.writing_style

        try:
            # 阶段 1: 构思灵感
            inspiration = self._phase_inspiration(context, hint)

            # 阶段 2: 规划大纲
            outline = self._phase_outline(context, inspiration, hint)

            # 阶段 3: 草稿写作
            draft = self._phase_draft(context, outline, style)

            # 阶段 4: 修订编辑
            revised = self._phase_revision(draft, style, context)

            # 阶段 5: 反馈优化
            final = self._phase_polish(revised, style, outline)

            return final
        except Exception as exc:
            logger.warning("Multi-stage continuation failed, falling back to legacy: %s", exc)
            return self._legacy_continue(text, context, hint)

    def _phase_inspiration(self, context: StoryContext, hint: str) -> dict:
        """阶段1: 分析故事主题、情感基调、可能的结局方向。"""
        context_block = format_context_sections(context.for_prompt())
        recent_events = self._format_recent_events(context)
        unresolved = self._format_unresolved(context)

        prompt = fill_template(
            self.inspiration_template,
            context_block=context_block,
            recent_events=recent_events,
            unresolved_threads=unresolved,
            hint=hint or "无特别要求",
        )

        try:
            response = self.llm.generate(prompt=prompt)
            result = load_json_response(response.content)
            if result:
                return result
        except Exception as exc:
            logger.warning("Inspiration phase failed: %s", exc)

        return {
            "core_theme": "命运与抉择",
            "emotional_tone": "沉稳",
            "possible_endings": ["开放式结局"],
            "recommended_direction": "OE",
            "key_resolution": "核心冲突的自然收束",
        }

    def _phase_outline(self, context: StoryContext, inspiration: dict, hint: str) -> EndingOutline:
        """阶段2: 基于灵感构思规划具体大纲。"""
        characters = self._format_main_characters(context)
        conflicts = self._format_conflicts(context)

        prompt = fill_template(
            self.outline_template,
            inspiration=json.dumps(inspiration, ensure_ascii=False),
            characters=characters,
            conflicts=conflicts,
            hint=hint or "无特别要求",
        )

        try:
            response = self.llm.generate(prompt=prompt)
            result = load_json_response(response.content)
            if result:
                return EndingOutline(
                    theme=result.get("theme", ""),
                    ending_direction=result.get("ending_direction", "OE"),
                    key_beats=result.get("key_beats", []),
                    emotional_arc=result.get("emotional_arc", ""),
                    final_image=result.get("final_image", ""),
                )
        except Exception as exc:
            logger.warning("Outline phase failed: %s", exc)

        return EndingOutline(
            theme="故事收束",
            ending_direction="OE",
            key_beats=["情节推进", "情感转折", "结局揭示"],
            emotional_arc="平稳→高潮→释然",
            final_image="留白意象",
        )

    def _phase_draft(self, context: StoryContext, outline: EndingOutline, style: WritingStyle | None) -> str:
        """阶段3: 按大纲写作初稿，应用风格指南。"""
        outline_text = self._format_outline(outline)
        style_guide = format_style_guide(style)
        style_samples = format_style_samples(style)
        recent_segments = self._recent_segment_digest()

        prompt = fill_template(
            self.draft_template,
            outline=outline_text,
            style_guide=style_guide,
            style_samples=style_samples,
            recent_segments=recent_segments,
        )

        try:
            response = self.llm.generate(prompt=prompt)
            content = response.content.strip()
            if content:
                return content
        except Exception as exc:
            logger.warning("Draft phase failed: %s", exc)

        return f"[初稿生成失败] 基于大纲「{outline.theme}」的续写内容。"

    def _phase_revision(self, draft: str, style: WritingStyle | None, context: StoryContext) -> str:
        """阶段4: 检查并修订初稿，确保风格一致性。"""
        style_guide = format_style_guide(style)
        characteristic_words = ", ".join(style.characteristic_words) if style and style.characteristic_words else ""
        tone_markers = ", ".join(style.tone_markers) if style and style.tone_markers else ""
        checklist = self._revision_checklist(style)

        prompt = fill_template(
            self.revision_template,
            draft=draft,
            style_guide=style_guide,
            characteristic_words=characteristic_words,
            tone_markers=tone_markers,
            checklist=checklist,
        )

        try:
            response = self.llm.generate(prompt=prompt)
            content = response.content.strip()
            if content:
                return content
        except Exception as exc:
            logger.warning("Revision phase failed: %s", exc)

        return draft  # Fallback to draft if revision fails

    def _phase_polish(self, revised: str, style: WritingStyle | None, outline: EndingOutline) -> str:
        """阶段5: 最终润色，强化意象和情感。"""
        style_summary = style.style_summary if style else ""

        prompt = fill_template(
            self.polish_template,
            revised_content=revised,
            final_image=outline.final_image or "自然意象",
            emotional_arc=outline.emotional_arc or "情感收束",
            style_summary=style_summary,
        )

        try:
            response = self.llm.generate(prompt=prompt)
            content = response.content.strip()
            if content:
                return content
        except Exception as exc:
            logger.warning("Polish phase failed: %s", exc)

        return revised  # Fallback to revised if polish fails

    # Helper methods ---------------------------------------------------------

    def _format_recent_events(self, context: StoryContext) -> str:
        """Format recent plot events for prompts."""
        if not context.events:
            return "(无近期重要事件)"
        events = context.events[-5:]
        lines = [f"- {event.summary}" for event in events]
        return "\n".join(lines)

    def _format_unresolved(self, context: StoryContext) -> str:
        """Format unresolved threads from characters and story state."""
        threads: list[str] = []
        for char in context.characters.values():
            if char.unresolved:
                for item in char.unresolved[:2]:
                    threads.append(f"- {char.name}: {item}")
        if context.story_state and context.story_state.unresolved_events:
            for item in context.story_state.unresolved_events[:3]:
                threads.append(f"- 世界: {item}")
        return "\n".join(threads) if threads else "(无明显未解决伏笔)"

    def _format_main_characters(self, context: StoryContext) -> str:
        """Format main characters for outline prompt."""
        if not context.characters:
            return "(无主要人物信息)"
        lines = []
        for char in context.characters.values():
            if char.role in ("main", "support"):
                traits = ", ".join(char.personality[:3]) if char.personality else "特征未知"
                lines.append(f"- {char.name} ({char.role}): {traits}")
        return "\n".join(lines) if lines else "(无主要人物信息)"

    def _format_conflicts(self, context: StoryContext) -> str:
        """Format major conflicts for outline prompt."""
        if not context.story_state or not context.story_state.major_conflicts:
            return "(无明确核心冲突)"
        conflicts = context.story_state.major_conflicts[-3:]
        return "\n".join(f"- {c}" for c in conflicts)

    def _format_outline(self, outline: EndingOutline) -> str:
        """Format outline for draft prompt."""
        lines = [
            f"主题: {outline.theme}",
            f"结局方向: {outline.ending_direction}",
            f"情感曲线: {outline.emotional_arc}",
            "关键情节点:",
        ]
        for beat in outline.key_beats:
            lines.append(f"  - {beat}")
        lines.append(f"结尾意象: {outline.final_image}")
        return "\n".join(lines)

    def _revision_checklist(self, style: WritingStyle | None) -> str:
        """Generate revision checklist based on style."""
        items = [
            "□ 句式节奏与原作一致",
            "□ 段落间过渡自然",
            "□ 人物行为符合设定",
        ]
        if style:
            if style.characteristic_words:
                items.append("□ 适当使用特征词汇")
            if style.tone_markers:
                items.append("□ 适当使用语气词")
            if style.metaphor_style:
                items.append(f"□ 比喻风格: {style.metaphor_style}")
        return "\n".join(items)

    def _recent_segment_digest(self) -> str:
        """Get digest of recent segments for context."""
        segments = self.segment_index.segments[-3:]
        if not segments:
            return "(无可参考片段)"
        parts = []
        for segment in segments:
            content = segment.content.strip().replace("\n", " ")
            parts.append(f"[Segment {segment.segment_id}] {content[:280]}")
        return "\n".join(parts)

    # Legacy fallback --------------------------------------------------------

    def _legacy_continue(self, text: str, context: StoryContext, hint: str) -> str:
        """Legacy single-stage continuation."""
        context_block = format_context_sections(context.for_prompt())
        recent_segments = self._recent_segment_digest()
        prompt = fill_template(
            self.legacy_template,
            context_block=context_block,
            recent_segments=recent_segments,
            hint=hint or "无特别需求，按原作基调收束",
        )
        try:
            response = self.llm.generate(prompt=prompt)
            content = response.content.strip()
            if content:
                return content
        except Exception as exc:
            logger.warning("Ending continuation failed, falling back to heuristic ending: %s", exc)
        return self._fallback(context, hint)

    def _fallback(self, context: StoryContext, hint: str) -> str:
        """Fallback heuristic ending when all else fails."""
        last_summary = context.chapter_window[-1] if context.chapter_window else None
        recent_events = context.events[-3:]
        tone = context.story_state.world_tension if context.story_state else "medium"
        paragraphs: list[str] = []
        if last_summary:
            paragraphs.append(
                f"After {last_summary.title}, {last_summary.synopsis.lower()} sets the stage for the finale."
            )
        if recent_events:
            beats = "; ".join(event.summary for event in recent_events)
            paragraphs.append(f"Consequences converge: {beats}.")
        if hint:
            paragraphs.append(f"Guided by the request ({hint}), the protagonists choose a fitting resolution.")
        tension_line = "The atmosphere eases." if tone == "low" else "Tension peaks before dissolving."
        paragraphs.append(tension_line)
        paragraphs.append("Loose threads are acknowledged, promising future tales without contradicting the past.")
        paragraphs.append("(本段为读者定制版本的兜底续写)")
        return "\n\n".join(paragraphs)
