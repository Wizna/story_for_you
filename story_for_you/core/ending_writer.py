from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from story_for_you.analysis.context import StoryContext, WritingStyle
from story_for_you.config.settings import EndingPhaseTemperatures, RenderingLimits
from story_for_you.core.ending import (
    BANNED_EXPRESSIONS_PROMPT,
    SCENE_KEYWORDS,
    HintInterpreter,
    StyleEnforcer,
)
from story_for_you.core.exceptions import GenerationError, LLMError
from story_for_you.core.prompting import (
    fill_template,
    format_context_sections,
    format_style_constraints,
    format_style_guide,
    format_style_samples,
    load_template,
)
from story_for_you.indexer.segment import Segment, SegmentIndex
from story_for_you.llm.base import LLMProvider
from story_for_you.utils.json_utils import load_json_response
from story_for_you.utils.prompting import SNIPPET_EXCERPT_LEN

logger = logging.getLogger(__name__)

_MAX_FOCUS_CHARACTERS = 4
_FOCUS_CHARACTERS_FALLBACK = 2
_MAX_SEGMENT_SNIPPETS = 4
_MAX_CAST_IN_ANCHORS = 4
_MAX_CONFLICT_ANCHORS = 2
_MAX_SCENE_ANCHORS = 3
_MIN_SCENE_SENTENCE_LEN = 15
_MAX_SCENE_SENTENCE_LEN = 80
_MIN_DRAFT_PARAGRAPHS = 3
_MIN_PARAGRAPH_CHARS = 120
_MAX_BEAT_PARAGRAPHS = 4


@dataclass
class EndingOutline:
    """续写大纲结构"""

    core_theme: str = ""
    ending_direction: str = ""  # HE/BE/OE
    emotional_tone: str = ""
    timeline: str = ""  # 时间跨度描述
    key_beats: list[str] = field(default_factory=list)
    emotional_arc: str = ""
    final_image: str = ""
    key_resolution: str = ""


class EndingWriter:
    """多阶段续写器，模拟人类作者创作流程。

    4 阶段流程：构思大纲 → 初稿写作 → 修订润色 → 伏笔检查
    """

    def __init__(
        self,
        llm: LLMProvider,
        segment_index: SegmentIndex,
        temperatures: EndingPhaseTemperatures | None = None,
        rendering_limits: RenderingLimits | None = None,
    ):
        self.llm = llm
        self.segment_index = segment_index
        self._limits = rendering_limits or RenderingLimits()
        self._hint_interpreter = HintInterpreter()
        temps = temperatures or EndingPhaseTemperatures()
        self._phase_llm_options = {
            "outline": {"temperature": temps.outline, "no_think": True},
            "draft": {"temperature": temps.draft},
            "polish": {"temperature": temps.polish},
            "resolution": {"temperature": temps.resolution, "no_think": True},
        }
        self._load_templates()

    def _load_templates(self) -> None:
        """Load all stage templates."""
        self.outline_template = load_template("ending_outline")
        self.draft_template = load_template("ending_draft")
        self.polish_template = load_template("ending_polish")
        self.resolution_template = load_template("ending_resolution")

    def continue_story(self, text: str, context: StoryContext, hint: str = "") -> str:
        """执行完整的 4 阶段续写流程。"""
        style = context.writing_style
        context_block = format_context_sections(context.for_prompt(limits=self._limits))
        style_anchors = self._build_style_anchors(context)
        directives = self._hint_interpreter.interpret(hint, context)
        hint_payload = directives.for_prompt()

        try:
            outline = self._phase_outline(context, context_block, hint_payload)
            if directives.ending_direction:
                outline.ending_direction = directives.ending_direction

            draft = self._phase_draft(
                context, outline, style, context_block, hint_payload, style_anchors
            )
            polished = self._phase_polish(
                draft, style, outline, context_block, hint_payload, style_anchors
            )
            final = self._phase_resolution_review(polished, context, context_block, style, hint_payload)

            enforcer = StyleEnforcer(style)
            return enforcer.post_process(final)
        except (LLMError, GenerationError) as exc:
            logger.warning("Multi-stage continuation failed, falling back to simple draft: %s", exc)
            return self._fallback_draft(context, style, context_block, hint_payload, style_anchors)

    def _phase_outline(self, context: StoryContext, context_block: str, hint: str) -> EndingOutline:
        """阶段1: 分析主题、情感、方向，并规划具体大纲（合并原 inspiration + outline）。"""
        recent_events = self._format_recent_events(context)
        unresolved = self._format_unresolved(context)
        characters = self._format_main_characters(context)
        conflicts = self._format_conflicts(context)

        prompt = fill_template(
            self.outline_template,
            context_block=context_block,
            recent_events=recent_events,
            unresolved_threads=unresolved,
            characters=characters,
            conflicts=conflicts,
            hint=hint,
        )

        try:
            response = self.llm.generate(prompt=prompt, options=self._phase_options("outline"))
            result = load_json_response(response.content)
            if result:
                return EndingOutline(
                    core_theme=result.get("core_theme", ""),
                    ending_direction=result.get("ending_direction", "OE"),
                    emotional_tone=result.get("emotional_tone", ""),
                    timeline=result.get("timeline", ""),
                    key_beats=result.get("key_beats", []),
                    emotional_arc=result.get("emotional_arc", ""),
                    final_image=result.get("final_image", ""),
                    key_resolution=result.get("key_resolution", ""),
                )
        except LLMError as exc:
            logger.warning("Outline phase failed: %s", exc)

        return EndingOutline(
            core_theme="命运与抉择",
            ending_direction="OE",
            emotional_tone="沉稳",
            timeline="当天",
            key_beats=["情节推进", "情感转折", "结局揭示"],
            emotional_arc="平稳→高潮→释然",
            final_image="留白意象",
            key_resolution="核心冲突的自然收束",
        )

    def _phase_draft(
        self,
        context: StoryContext,
        outline: EndingOutline,
        style: WritingStyle | None,
        context_block: str,
        hint: str,
        style_anchors: str,
    ) -> str:
        """阶段2: 按大纲写作初稿，应用风格指南。"""
        outline_text = self._format_outline(outline)
        style_guide = format_style_guide(style)
        style_samples = format_style_samples(style)
        style_constraints = format_style_constraints(style)
        recent_segments = self._recent_segment_digest(context)
        required_characters = self._format_main_characters(context)
        loose_threads = self._format_unresolved(context)

        prompt = fill_template(
            self.draft_template,
            outline=outline_text,
            style_guide=style_guide,
            style_samples=style_samples,
            style_constraints=style_constraints,
            recent_segments=recent_segments,
            context_block=context_block or "(无上下文)",
            hint=hint,
            required_characters=required_characters or "(未提供人物信息)",
            loose_threads=loose_threads or "(暂无伏笔)",
            beat_constraints=self._draft_paragraph_plan(outline),
            style_anchors=style_anchors,
            banned_expressions=BANNED_EXPRESSIONS_PROMPT,
        )

        try:
            response = self.llm.generate(prompt=prompt, options=self._phase_options("draft"))
            content = response.content.strip()
            if content:
                return content
        except LLMError as exc:
            logger.warning("Draft phase failed: %s", exc)

        return f"[初稿生成失败] 基于大纲「{outline.core_theme}」的续写内容。"

    def _phase_polish(
        self,
        draft: str,
        style: WritingStyle | None,
        outline: EndingOutline,
        context_block: str,
        hint: str,
        style_anchors: str,
    ) -> str:
        """阶段3: 修订并润色初稿（合并原 revision + polish）。"""
        style_guide = format_style_guide(style)
        style_samples = format_style_samples(style)
        style_constraints = format_style_constraints(style)
        characteristic_words = ", ".join(style.characteristic_words) if style and style.characteristic_words else ""
        tone_markers = ", ".join(style.tone_markers) if style and style.tone_markers else ""
        checklist = self._revision_checklist(style)

        prompt = fill_template(
            self.polish_template,
            draft=draft,
            final_image=outline.final_image or "自然意象",
            emotional_arc=outline.emotional_arc or "情感收束",
            style_guide=style_guide,
            style_samples=style_samples,
            style_constraints=style_constraints,
            characteristic_words=characteristic_words,
            tone_markers=tone_markers,
            checklist=checklist,
            context_block=context_block or "(无上下文)",
            hint=hint,
            beat_constraints=self._draft_paragraph_plan(outline),
            style_anchors=style_anchors,
            banned_expressions=BANNED_EXPRESSIONS_PROMPT,
        )

        try:
            response = self.llm.generate(prompt=prompt, options=self._phase_options("polish"))
            content = response.content.strip()
            if content:
                return content
        except LLMError as exc:
            logger.warning("Polish phase failed: %s", exc)

        return draft

    def _phase_resolution_review(
        self,
        polished: str,
        context: StoryContext,
        context_block: str,
        style: WritingStyle | None,
        hint: str,
    ) -> str:
        """阶段4: 校验伏笔收束情况，必要时补写桥段。"""
        threads = self._collect_unresolved_threads(context)
        if not threads:
            return polished

        unresolved = "\n".join(f"- {item}" for item in threads)
        style_guide = format_style_guide(style)
        style_samples = format_style_samples(style)
        prompt = fill_template(
            self.resolution_template,
            final_content=polished,
            unresolved_threads=unresolved,
            style_guide=style_guide or "(无风格约束)",
            style_samples=style_samples or "(暂无示例)",
            context_block=context_block or "(无上下文)",
            hint=hint,
        )

        try:
            response = self.llm.generate(prompt=prompt, options=self._phase_options("resolution"))
            payload = load_json_response(response.content)
            if not payload:
                return polished
            status = payload.get("status", "ok")
            if status == "ok":
                return polished
            bridges = [item.strip() for item in payload.get("bridges", []) if isinstance(item, str) and item.strip()]
            if not bridges:
                return polished
            enforcer = StyleEnforcer(style)
            bridges = enforcer.filter_duplicate_bridges(polished, bridges)
            if not bridges:
                logger.debug("All bridges filtered as duplicates")
                return polished
            notes = payload.get("notes")
            note_text = notes.strip() if isinstance(notes, str) else None
            return self._append_bridges(polished, bridges, note_text)
        except LLMError as exc:
            logger.warning("Resolution phase failed: %s", exc)
            return polished

    # Helper methods ---------------------------------------------------------

    def _format_recent_events(self, context: StoryContext) -> str:
        if not context.events:
            return "(无近期重要事件)"
        events = context.events[-self._limits.max_events:]
        lines = [f"- {event.summary}" for event in events]
        return "\n".join(lines)

    def _format_unresolved(self, context: StoryContext) -> str:
        threads = self._collect_unresolved_threads(context)
        return "\n".join(f"- {item}" for item in threads) if threads else "(无明显未解决伏笔)"

    def _format_main_characters(self, context: StoryContext) -> str:
        if not context.characters:
            return "(无主要人物信息)"
        lines = []
        for char in context.characters.values():
            if char.role in ("main", "support"):
                traits = ", ".join(char.personality[:self._limits.max_personality_traits]) if char.personality else "特征未知"
                alias_info = ""
                if char.aliases:
                    alias_info = f"，别名/称呼：{'、'.join(char.aliases[:self._limits.max_aliases])}"
                lines.append(f"- {char.name} ({char.role}): {traits}{alias_info}")
        return "\n".join(lines) if lines else "(无主要人物信息)"

    def _format_conflicts(self, context: StoryContext) -> str:
        if not context.story_state or not context.story_state.major_conflicts:
            return "(无明确核心冲突)"
        conflicts = context.story_state.major_conflicts[-self._limits.max_major_conflicts:]
        return "\n".join(f"- {c}" for c in conflicts)

    def _format_outline(self, outline: EndingOutline) -> str:
        lines = [
            f"主题: {outline.core_theme}",
            f"结局方向: {outline.ending_direction}",
            f"时间跨度: {outline.timeline}",
            f"情感曲线: {outline.emotional_arc}",
            "关键情节点:",
        ]
        for beat in outline.key_beats:
            lines.append(f"  - {beat}")
        lines.append(f"结尾意象: {outline.final_image}")
        return "\n".join(lines)

    def _draft_paragraph_plan(self, outline: EndingOutline) -> str:
        beats = [
            beat.strip()
            for beat in outline.key_beats
            if isinstance(beat, str) and beat and beat.strip()
        ]
        if not beats:
            return f"至少写满{_MIN_DRAFT_PARAGRAPHS}段，每段{_MIN_PARAGRAPH_CHARS}字以上，段落按因果衔接展开。"

        lines: list[str] = []
        for idx, beat in enumerate(beats[:_MAX_BEAT_PARAGRAPHS], start=1):
            lines.append(f"- 段落{idx}: {beat}")
        if len(beats) > _MAX_BEAT_PARAGRAPHS:
            lines.append(f"- 其余段落: 融合剩余{len(beats) - _MAX_BEAT_PARAGRAPHS}个情节点，避免遗漏。")
        lines.append(f"- 每段≥{_MIN_PARAGRAPH_CHARS}字，首句承接上一段尾句，整体保持因果衔接。")
        return "\n".join(lines)

    def _revision_checklist(self, style: WritingStyle | None) -> str:
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

    def _recent_segment_digest(self, context: StoryContext) -> str:
        if not self.segment_index.segments:
            return "(无可参考片段)"

        focus_characters = [
            name
            for name, char in context.characters.items()
            if char.role in ("main", "support")
        ]
        if not focus_characters and context.characters:
            focus_characters = list(context.characters.keys())[:_FOCUS_CHARACTERS_FALLBACK]

        snippets: list[str] = []
        seen_segments: set[int] = set()

        for name in focus_characters[:_MAX_FOCUS_CHARACTERS]:
            segment = self._latest_segment_for_character(name)
            if not segment:
                continue
            seen_segments.add(segment.segment_id)
            snippets.append(self._format_segment_snippet(segment, prefix=f"[{name}]"))

        for segment in reversed(self.segment_index.segments):
            if segment.segment_id in seen_segments:
                continue
            snippets.append(self._format_segment_snippet(segment))
            if len(snippets) >= _MAX_SEGMENT_SNIPPETS:
                break

        recent_event = self._latest_irreversible_event(context)
        if recent_event:
            flag = f"[CH{recent_event.chapter:03d}]" if recent_event.chapter else "[事件]"
            snippets.append(f"{flag} {recent_event.summary}")

        return "\n".join(snippets) if snippets else "(无可参考片段)"

    def _latest_segment_for_character(self, name: str) -> Segment | None:
        segment_ids = self.segment_index.char_index.get(name)
        if not segment_ids:
            return None
        target_id = max(segment_ids)
        for segment in self.segment_index.segments:
            if segment.segment_id == target_id:
                return segment
        return None

    def _format_segment_snippet(self, segment: Segment, prefix: str | None = None) -> str:
        content = segment.content.strip().replace("\n", " ")
        snippet = content[:SNIPPET_EXCERPT_LEN]
        label = prefix or f"[Segment {segment.segment_id}]"
        return f"{label} {snippet}"

    def _latest_irreversible_event(self, context: StoryContext):
        for event in reversed(context.events):
            if event.is_irreversible:
                return event
        return context.events[-1] if context.events else None

    def _collect_unresolved_threads(self, context: StoryContext) -> list[str]:
        threads: list[str] = []
        for char in context.characters.values():
            for item in char.unresolved[:self._limits.max_unresolved_per_char]:
                threads.append(f"{char.name}: {item}")
        if context.story_state and context.story_state.unresolved_events:
            for item in context.story_state.unresolved_events[-self._limits.max_unresolved_events:]:
                threads.append(f"世界: {item}")
        return threads

    def _build_style_anchors(self, context: StoryContext) -> str:
        anchors: list[str] = []
        style = context.writing_style
        if style and style.characteristic_words:
            keywords = "、".join(style.characteristic_words[:self._limits.max_characteristic_words])
            anchors.append(f"词汇锚点：{keywords}（至少使用其中2个）")

        scene_images = self._scene_anchors(limit=_MAX_SCENE_ANCHORS)
        if scene_images:
            anchors.append("场景意象：")
            anchors.extend(f"  · {scene}" for scene in scene_images)

        cast = [
            char.name
            for char in context.characters.values()
            if char.role in ("main", "support")
        ]
        if cast:
            anchors.append("关键人物：" + "、".join(cast[:_MAX_CAST_IN_ANCHORS]))

        state = context.story_state
        if state:
            if state.major_conflicts:
                anchors.append("冲突焦点：" + " / ".join(state.major_conflicts[-_MAX_CONFLICT_ANCHORS:]))
            if state.unresolved_events:
                anchors.append("伏笔提示：" + "；".join(state.unresolved_events[:_MAX_CONFLICT_ANCHORS]))
            if state.time_constraints:
                anchors.append("时间线提示：" + state.time_constraints[0])

        if not anchors:
            return "(无锚点，保持对场景与人物细致描写)"

        lines: list[str] = []
        for item in anchors:
            if item.startswith("  "):
                lines.append(item)
            else:
                lines.append(f"- {item}")
        return "\n".join(lines)

    def _scene_anchors(self, limit: int = _MAX_SCENE_ANCHORS) -> list[str]:
        scenes: list[str] = []
        for segment in reversed(self.segment_index.segments):
            candidates = self._extract_scene_sentences(segment.content)
            for sentence in candidates:
                if sentence not in scenes:
                    scenes.append(sentence)
                    if len(scenes) >= limit:
                        return scenes
        return scenes

    def _extract_scene_sentences(self, text: str) -> list[str]:
        normalized = text.strip()
        if not normalized:
            return []
        parts = re.split(r"[。！？]", normalized)
        results: list[str] = []
        for part in parts:
            sentence = part.strip()
            if sentence.startswith('"') or sentence.startswith('\u201c') or '：\u201c' in sentence:
                continue
            if sentence.startswith("…") or sentence.startswith("。"):
                continue
            if _MIN_SCENE_SENTENCE_LEN <= len(sentence) <= _MAX_SCENE_SENTENCE_LEN and self._has_scene_keyword(sentence):
                results.append(sentence)
        return results

    def _has_scene_keyword(self, text: str) -> bool:
        return any(kw in text for kw in SCENE_KEYWORDS)

    def _append_bridges(self, polished: str, bridges: list[str], notes: str | None = None) -> str:
        baseline = polished.rstrip()
        bridge_text = "\n\n".join(bridges)
        combined_parts = [baseline, bridge_text.strip()]
        if notes:
            logger.debug("Resolution notes (not included in output): %s", notes.strip())
        return "\n\n".join(part for part in combined_parts if part)

    def _phase_options(self, phase: str) -> dict | None:
        return self._phase_llm_options.get(phase)

    # Fallback ---------------------------------------------------------------

    def _fallback_draft(
        self,
        context: StoryContext,
        style: WritingStyle | None,
        context_block: str,
        hint: str,
        style_anchors: str,
    ) -> str:
        """Simplified single-call fallback using the draft template."""
        outline = EndingOutline(
            core_theme="故事收束",
            ending_direction="OE",
            timeline="当天",
            key_beats=["情节推进", "情感转折", "结局揭示"],
            emotional_arc="平稳→高潮→释然",
            final_image="留白意象",
            key_resolution="核心冲突的自然收束",
        )

        try:
            content = self._phase_draft(context, outline, style, context_block, hint, style_anchors)
            if content and not content.startswith("[初稿生成失败]"):
                enforcer = StyleEnforcer(style)
                return enforcer.post_process(content)
        except LLMError as exc:
            logger.warning("Fallback draft also failed: %s", exc)

        return self._heuristic_ending(context, hint)

    def _heuristic_ending(self, context: StoryContext, hint: str) -> str:
        """Last-resort heuristic ending when all LLM calls fail."""
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
        paragraphs.append("余音袅袅，故事就此落幕。")
        return "\n\n".join(paragraphs)
