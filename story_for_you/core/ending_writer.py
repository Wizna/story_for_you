from __future__ import annotations

import logging
from dataclasses import dataclass, field

from story_for_you.analysis.context import StoryContext, WritingStyle
from story_for_you.config.settings import EndingPhaseTemperatures, RenderingLimits
from story_for_you.core.ending import (
    BANNED_EXPRESSIONS_PROMPT,
    EndingValidator,
    HintInterpreter,
    StyleEnforcer,
)
from story_for_you.core.exceptions import GenerationError, LLMResponseError
from story_for_you.core.prompting import (
    format_context_sections,
    format_style_constraints,
    format_style_guide,
    format_style_samples,
    load_template,
)
from story_for_you.indexer.segment import Segment, SegmentIndex
from story_for_you.llm.base import LLMProvider
from story_for_you.llm.telemetry import telemetry_options
from story_for_you.utils.json_utils import load_json_response
from story_for_you.utils.prompting import SNIPPET_EXCERPT_LEN, build_cacheable_prompt

logger = logging.getLogger(__name__)

_MAX_FOCUS_CHARACTERS = 4
_FOCUS_CHARACTERS_FALLBACK = 2
_MAX_SEGMENT_SNIPPETS = 4
_MAX_CAST_IN_ANCHORS = 4
_MAX_CONFLICT_ANCHORS = 2
_MIN_DRAFT_PARAGRAPHS = 3
_MIN_PARAGRAPH_CHARS = 120
_MAX_BEAT_PARAGRAPHS = 4
_VALID_RESOLUTION_STATUSES = {"ok", "needs_bridges", "blocked"}
_MAX_FINAL_REPAIR_ATTEMPTS = 1
_MAX_HARD_FACTS = 20
_SOURCE_TAIL_CHARS = 6000
_MAX_RECENT_SCENES = 4
_DEATH_MARKERS = ("死亡", "身亡", "战死", "尸体", "死去", "毙命", "殒命", "去世", "葬")


@dataclass
class EndingOutline:
    """续写大纲结构"""

    core_theme: str = ""
    # Kept as a compatibility field for older cached prompts. New planning
    # uses continuation_intent and closure_level instead of a fixed ending
    # taxonomy such as HE/BE/OE.
    ending_direction: str = ""
    continuation_intent: str = ""
    closure_level: str = ""
    emotional_tone: str = ""
    timeline: str = ""  # 时间跨度描述
    key_beats: list[str] = field(default_factory=list)
    emotional_arc: str = ""
    final_image: str = ""
    key_resolution: str = ""


@dataclass
class EndingChapterPlan:
    """One narrative movement in an adaptive continuation plan."""

    number: int
    title: str
    purpose: str
    viewpoint: str
    setting: str
    beats: list[str] = field(default_factory=list)
    focus_characters: list[str] = field(default_factory=list)
    resolutions: list[str] = field(default_factory=list)
    carry_forward: list[str] = field(default_factory=list)
    end_state: str = ""


@dataclass
class ContinuationPlan:
    """Global continuation plan chosen from story evidence and reader intent.

    ``chapter_count`` and ``ending_direction`` remain as compatibility
    properties for callers written against the earlier ending-only planner.
    The model is free to choose an ongoing arc, a partial resolution, or a
    full closure; the writer does not assume that every continuation is an
    ending.
    """

    unit_count: int
    rationale: str
    continuation_intent: str
    narrative_horizon: str
    closure_level: str
    unit_type: str
    chapters: list[EndingChapterPlan] = field(default_factory=list)

    @property
    def chapter_count(self) -> int:
        return self.unit_count

    @property
    def ending_direction(self) -> str:
        """Compatibility view for old prompt consumers."""

        return self.closure_level


# Public compatibility alias. The command and old integrations used this
# name before continuation became a generic operation.
EndingPlan = ContinuationPlan


class EndingWriter:
    """Adaptive continuation writer: plan the required units, then write them."""

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
        self._hint_interpreter = HintInterpreter(llm, rendering_limits=self._limits)
        self._ending_validator = EndingValidator(llm)
        temps = temperatures or EndingPhaseTemperatures()
        self._phase_llm_options = {
            "outline": {"temperature": temps.outline, "no_think": True},
            "draft": {"temperature": temps.draft},
            "polish": {"temperature": temps.polish},
            "resolution": {"temperature": temps.resolution, "no_think": True},
            "final_repair": {"temperature": temps.polish, "no_think": True},
        }
        self._load_templates()

    def _load_templates(self) -> None:
        """Load all stage templates."""
        self.outline_template = load_template("ending_outline")
        self.draft_template = load_template("ending_draft")
        self.polish_template = load_template("ending_polish")
        self.resolution_template = load_template("ending_resolution")
        self.final_repair_template = load_template("ending_final_repair")
        self.plan_template = load_template("ending_plan")

    def continue_story(
        self,
        text: str,
        context: StoryContext,
        hint: str = "",
        *,
        max_chapters: int = 6,
    ) -> str:
        """Plan and write an adaptive continuation."""
        if max_chapters < 1:
            raise ValueError("max_chapters must be positive")
        style = context.writing_style
        context_block = self._build_continuation_context(text, context)
        style_anchors = self._build_style_anchors(context)
        directives = self._hint_interpreter.interpret(hint, context)
        hint_payload = directives.for_prompt()

        plan = self._phase_plan(context_block, hint_payload, max_chapters)
        if directives.continuation_intent:
            plan.continuation_intent = directives.continuation_intent
        if directives.closure != "unspecified":
            plan.closure_level = directives.closure

        generated: list[str] = []
        for chapter in plan.chapters:
            chapter_context = context_block
            if generated:
                prior = "\n\n".join(generated)[-8000:]
                chapter_context += "\n\n## 已生成的前文（只用于承接，不要重复）\n" + prior
            outline = self._outline_from_chapter(plan, chapter)
            instruction = self._chapter_instruction(plan, chapter, directives.closure)
            draft = self._phase_draft(
                context, outline, style, chapter_context, hint_payload, style_anchors, instruction
            )
            polished = self._phase_polish(
                draft, style, outline, chapter_context, hint_payload, style_anchors, instruction
            )
            generated.append(f"第{chapter.number}{plan.unit_type} {chapter.title}\n\n{polished}")

        final = "\n\n".join(generated).strip()
        final = self._phase_resolution_review(
            final, context, context_block, style, hint_payload, plan.closure_level
        )
        validation_context = (
            context_block
            + "\n\n## Continuation Plan\n"
            + self._format_continuation_plan(plan)
        )

        enforcer = StyleEnforcer(style)
        final = enforcer.post_process(final)
        final = self._validate_or_repair_final(
            final,
            directives,
            validation_context,
            style,
            hint_payload,
            enforcer,
        )
        return final

    def _phase_plan(self, context_block: str, hint: str, max_chapters: int) -> ContinuationPlan:
        prompt = build_cacheable_prompt(
            context_block or "(无上下文)",
            self.plan_template,
            prefix_placeholder="context_block",
            hint=hint,
            max_chapters=str(max_chapters),
        )
        response = self.llm.generate(
            prompt=prompt,
            options=telemetry_options(
                self._phase_options("outline"), phase="continue", step=": continuation plan"
            ),
        )
        payload = load_json_response(response.content)
        if not isinstance(payload, dict):
            raise LLMResponseError("Ending plan returned invalid JSON object.")
        for field_name in ("rationale", "chapters"):
            if field_name not in payload:
                raise LLMResponseError(f"Continuation plan missing required field: {field_name}")

        # Accept the old schema so cached/fake providers and user integrations
        # do not break during the migration. New prompts use unit_count and
        # free-form continuation semantics.
        count = payload.get("unit_count", payload.get("chapter_count"))
        if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= max_chapters:
            raise LLMResponseError(f"Continuation plan unit_count must be between 1 and {max_chapters}.")
        intent = payload.get("continuation_intent", payload.get("core_theme", "继续推进当前叙事"))
        horizon = payload.get("narrative_horizon", payload.get("timeline", "由本章计划决定"))
        closure = payload.get("closure_level", payload.get("ending_direction", "adaptive"))
        unit_type = payload.get("unit_type", "章")
        intent = self._required_str(intent, "continuation_intent")
        horizon = self._required_str(horizon, "narrative_horizon")
        closure = self._required_str(closure, "closure_level")
        unit_type = self._required_str(unit_type, "unit_type")
        chapters_payload = payload.get("chapters")
        if not isinstance(chapters_payload, list) or len(chapters_payload) != count:
            raise LLMResponseError("Continuation plan chapters must match unit_count.")
        chapters: list[EndingChapterPlan] = []
        for index, item in enumerate(chapters_payload, start=1):
            if not isinstance(item, dict):
                raise LLMResponseError("Ending plan chapter must be an object.")
            for field_name in ("title", "purpose", "viewpoint", "setting", "beats", "focus_characters"):
                if field_name not in item:
                    raise LLMResponseError(f"Continuation plan unit missing required field: {field_name}")
            resolutions = item.get("resolved_threads", item.get("resolutions", []))
            carry_forward = item.get("carry_forward", [])
            chapters.append(
                EndingChapterPlan(
                    number=index,
                    title=self._required_str(item.get("title"), "title"),
                    purpose=self._required_str(item.get("purpose"), "purpose"),
                    viewpoint=self._required_str(item.get("viewpoint"), "viewpoint"),
                    setting=self._required_str(item.get("setting"), "setting"),
                    beats=self._required_str_list(item.get("beats"), "beats"),
                    focus_characters=self._required_str_list(item.get("focus_characters"), "focus_characters"),
                    resolutions=self._required_str_list(resolutions, "resolved_threads"),
                    carry_forward=self._required_str_list(carry_forward, "carry_forward"),
                    end_state=self._required_str(item.get("end_state", item.get("result", "自然形成下一状态")), "end_state"),
                )
            )
        return ContinuationPlan(
            unit_count=count,
            rationale=self._required_str(payload.get("rationale"), "rationale"),
            continuation_intent=intent,
            narrative_horizon=horizon,
            closure_level=closure,
            unit_type=unit_type,
            chapters=chapters,
        )

    def _outline_from_chapter(self, plan: ContinuationPlan, chapter: EndingChapterPlan) -> EndingOutline:
        return EndingOutline(
            core_theme=plan.continuation_intent,
            ending_direction=plan.closure_level,
            continuation_intent=plan.continuation_intent,
            closure_level=plan.closure_level,
            emotional_tone=chapter.purpose,
            timeline=chapter.setting,
            key_beats=chapter.beats,
            emotional_arc="；".join(chapter.carry_forward),
            final_image="本章结尾由场景自然落点决定",
            key_resolution="；".join(chapter.resolutions) or chapter.end_state or "推进本章目标",
        )

    def _chapter_instruction(self, plan: EndingPlan, chapter: EndingChapterPlan, closure: str) -> str:
        is_last = chapter.number == plan.chapter_count
        if plan.closure_level in {"full", "closed"} and is_last:
            close = "本章可以完成本次计划允许的主要收束，但只交代有证据支持的结果"
        elif is_last:
            close = "本章结束于当前叙事自然形成的状态，保留尚未到时的线索与余波"
        else:
            close = "本章只完成列出的推进，保留下一单元所需的因果张力"
        return (
            f"本次续写计划共 {plan.chapter_count} 个{plan.unit_type}，本单元是第 {chapter.number} 个。"
            f"目的：{chapter.purpose}；视角：{chapter.viewpoint}；场景：{chapter.setting}。"
            f"重点人物：{'、'.join(chapter.focus_characters) or '以当前场景人物为准'}。"
            f"本单元结果：{chapter.end_state or '由行动和场景自然形成'}。{close}。"
            f"读者收束要求为 {closure}；不要擅自改变它。"
        )

    def _format_continuation_plan(self, plan: ContinuationPlan) -> str:
        lines = [
            f"续写单元数: {plan.chapter_count}（粒度：{plan.unit_type}）",
            f"续写意图: {plan.continuation_intent}",
            f"叙事跨度: {plan.narrative_horizon}",
            f"收束程度: {plan.closure_level}",
            f"判断理由: {plan.rationale}",
        ]
        for chapter in plan.chapters:
            lines.append(
                f"- 单元{chapter.number}《{chapter.title}》: {chapter.purpose}; "
                f"已处理={'; '.join(chapter.resolutions) or '无'}; "
                f"结果={chapter.end_state or '自然形成'}; "
                f"承接={'; '.join(chapter.carry_forward) or '无'}"
            )
        return "\n".join(lines)

    def _format_ending_plan(self, plan: ContinuationPlan) -> str:
        """Compatibility alias for integrations using the old method name."""

        return self._format_continuation_plan(plan)

    def _phase_outline(
        self, context: StoryContext, context_block: str, hint: str, chapter_instruction: str
    ) -> EndingOutline:
        """阶段1: 分析主题、情感、方向，并规划具体大纲（合并原 inspiration + outline）。"""
        recent_events = self._format_recent_events(context)
        unresolved = self._format_unresolved(context)
        characters = self._format_main_characters(context)
        conflicts = self._format_conflicts(context)

        prompt = build_cacheable_prompt(
            context_block or "(无上下文)",
            self.outline_template,
            prefix_placeholder="context_block",
            recent_events=recent_events,
            unresolved_threads=unresolved,
            characters=characters,
            conflicts=conflicts,
            hint=hint,
            chapter_instruction=chapter_instruction,
        )

        response = self.llm.generate(
            prompt=prompt,
            options=telemetry_options(
                self._phase_options("outline"),
                phase="continue",
                step=": outline",
            ),
        )
        result = load_json_response(response.content)
        if not isinstance(result, dict):
            raise LLMResponseError("Outline phase returned invalid JSON object.")
        for field_name in ("timeline", "key_beats", "emotional_arc", "final_image", "key_resolution"):
            if field_name not in result:
                raise LLMResponseError(f"Continuation outline missing required field: {field_name}")
        ending_direction = self._required_str(
            result.get("ending_direction", result.get("closure_level", "adaptive")),
            "ending_direction",
        )
        key_beats_payload = result.get("key_beats")
        if not isinstance(key_beats_payload, list):
            raise LLMResponseError("Outline key_beats must be a list.")
        key_beats = self._required_str_list(key_beats_payload, "key_beats")
        if not key_beats:
            raise LLMResponseError("Outline key_beats must not be empty.")
        return EndingOutline(
            core_theme=self._required_str(
                result.get("core_theme", result.get("continuation_intent", "继续推进当前叙事")),
                "core_theme",
            ),
            ending_direction=ending_direction,
            continuation_intent=self._required_str(
                result.get("continuation_intent", result.get("core_theme")),
                "continuation_intent",
            ),
            closure_level=self._required_str(
                result.get("closure_level", ending_direction), "closure_level"
            ),
            emotional_tone=self._required_str(
                result.get("emotional_tone", "由当前场景和原作基调决定"), "emotional_tone"
            ),
            timeline=self._required_str(result.get("timeline"), "timeline"),
            key_beats=key_beats,
            emotional_arc=self._required_str(result.get("emotional_arc"), "emotional_arc"),
            final_image=self._required_str(result.get("final_image"), "final_image"),
            key_resolution=self._required_str(result.get("key_resolution"), "key_resolution"),
        )

    def _phase_draft(
        self,
        context: StoryContext,
        outline: EndingOutline,
        style: WritingStyle | None,
        context_block: str,
        hint: str,
        style_anchors: str,
        chapter_instruction: str,
    ) -> str:
        """阶段2: 按大纲写作初稿，应用风格指南。"""
        outline_text = self._format_outline(outline)
        style_guide = format_style_guide(style)
        style_samples = format_style_samples(style)
        style_constraints = format_style_constraints(style)
        recent_segments = self._recent_segment_digest(context)
        required_characters = self._format_main_characters(context)
        loose_threads = self._format_unresolved(context)

        prompt = build_cacheable_prompt(
            context_block or "(无上下文)",
            self.draft_template,
            prefix_placeholder="context_block",
            outline=outline_text,
            style_guide=style_guide,
            style_samples=style_samples,
            style_constraints=style_constraints,
            recent_segments=recent_segments,
            hint=hint,
            required_characters=required_characters or "(未提供人物信息)",
            loose_threads=loose_threads or "(暂无伏笔)",
            beat_constraints=self._draft_paragraph_plan(outline),
            style_anchors=style_anchors,
            banned_expressions=BANNED_EXPRESSIONS_PROMPT,
            chapter_instruction=chapter_instruction,
        )

        response = self.llm.generate(
            prompt=prompt,
            options=telemetry_options(
                self._phase_options("draft"),
                phase="continue",
                step=": draft",
            ),
        )
        content = response.content.strip()
        if not content:
            raise LLMResponseError("Draft phase returned empty content.")
        return content

    def _phase_polish(
        self,
        draft: str,
        style: WritingStyle | None,
        outline: EndingOutline,
        context_block: str,
        hint: str,
        style_anchors: str,
        chapter_instruction: str,
    ) -> str:
        """阶段3: 修订并润色初稿（合并原 revision + polish）。"""
        style_guide = format_style_guide(style)
        style_samples = format_style_samples(style)
        style_constraints = format_style_constraints(style)
        characteristic_words = ", ".join(style.characteristic_words) if style and style.characteristic_words else ""
        tone_markers = ", ".join(style.tone_markers) if style and style.tone_markers else ""
        checklist = self._revision_checklist(style)

        prompt = build_cacheable_prompt(
            context_block or "(无上下文)",
            self.polish_template,
            prefix_placeholder="context_block",
            draft=draft,
            final_image=outline.final_image,
            emotional_arc=outline.emotional_arc,
            style_guide=style_guide,
            style_samples=style_samples,
            style_constraints=style_constraints,
            characteristic_words=characteristic_words,
            tone_markers=tone_markers,
            checklist=checklist,
            hint=hint,
            beat_constraints=self._draft_paragraph_plan(outline),
            style_anchors=style_anchors,
            banned_expressions=BANNED_EXPRESSIONS_PROMPT,
            chapter_instruction=chapter_instruction,
        )

        response = self.llm.generate(
            prompt=prompt,
            options=telemetry_options(
                self._phase_options("polish"),
                phase="continue",
                step=": polish",
            ),
        )
        content = response.content.strip()
        if not content:
            raise LLMResponseError("Polish phase returned empty content.")
        return content

    def _phase_resolution_review(
        self,
        polished: str,
        context: StoryContext,
        context_block: str,
        style: WritingStyle | None,
        hint: str,
        closure_level: str = "closed",
    ) -> str:
        """阶段4: 检查承接与伏笔，按模型判断的收束程度决定是否补桥。"""
        threads = self._collect_unresolved_threads(context)
        if not threads:
            return polished

        unresolved = "\n".join(f"- {item}" for item in threads)
        style_guide = format_style_guide(style)
        style_samples = format_style_samples(style)
        prompt = build_cacheable_prompt(
            context_block or "(无上下文)",
            self.resolution_template,
            prefix_placeholder="context_block",
            final_content=polished,
            unresolved_threads=unresolved,
            style_guide=style_guide or "(无风格约束)",
            style_samples=style_samples or "(暂无示例)",
            hint=hint,
            closure_level=closure_level,
        )

        response = self.llm.generate(
            prompt=prompt,
            options=telemetry_options(
                self._phase_options("resolution"),
                phase="continue",
                step=": resolution review",
            ),
        )
        payload = load_json_response(response.content)
        if not isinstance(payload, dict):
            raise LLMResponseError("Resolution phase returned invalid JSON object.")
        for field_name in ("status", "missing_threads", "bridges", "notes"):
            if field_name not in payload:
                raise LLMResponseError(f"Resolution phase missing required field: {field_name}")
        status = self._required_str(payload.get("status"), "status").lower()
        if status not in _VALID_RESOLUTION_STATUSES:
            raise LLMResponseError(f"Invalid resolution status: {status!r}")
        missing_threads = self._required_str_list(payload.get("missing_threads"), "missing_threads")
        bridges_payload = payload.get("bridges")
        if not isinstance(bridges_payload, list):
            raise LLMResponseError("Resolution bridges must be a list.")
        bridges = self._required_str_list(bridges_payload, "bridges")
        notes = self._required_str(payload.get("notes"), "notes", allow_empty=True)
        if status == "ok":
            if missing_threads or bridges:
                raise LLMResponseError("Resolution status ok requires empty missing_threads and bridges.")
            return polished
        if status == "blocked" and closure_level not in {"full", "closed"}:
            # An ongoing or partial continuation is allowed to leave threads
            # deliberately open. Keep the generated prose and record no
            # artificial bridge in the output.
            logger.info("Leaving unresolved threads open for continuation level %s", closure_level)
            return polished
        if status == "blocked":
            details = "; ".join(missing_threads) or notes
            raise GenerationError("Resolution phase blocked: " + (details or "unresolved threads remain"))
        if not bridges:
            raise GenerationError("Resolution phase requested changes but returned no bridge text.")
        enforcer = StyleEnforcer(style)
        bridges = enforcer.filter_duplicate_bridges(polished, bridges)
        if not bridges:
            raise GenerationError("Resolution bridge text duplicated existing content.")
        return self._append_bridges(polished, bridges, notes or None)

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
                if self._is_dead_character(char.name, context):
                    continue
                traits = ", ".join(char.personality[:self._limits.max_personality_traits]) if char.personality else "特征未知"
                alias_info = ""
                if char.aliases:
                    alias_info = f"，别名/称呼：{'、'.join(char.aliases[:self._limits.max_aliases])}"
                lines.append(f"- {char.name} ({char.role}): {traits}{alias_info}")
        return "\n".join(lines) if lines else "(无主要人物信息)"

    def _is_dead_character(self, name: str, context: StoryContext) -> bool:
        """Return true only for an explicit irreversible death fact."""
        character = context.characters.get(name)
        if character and character.status == "dead":
            return True
        for event in context.events:
            if not event.is_irreversible or name not in event.participants:
                continue
            evidence = " ".join(
                [event.summary, *event.impact.world_flags, *event.impact.power_shifts.values()]
            )
            if any(marker in evidence for marker in _DEATH_MARKERS):
                return True
        return False

    def _format_conflicts(self, context: StoryContext) -> str:
        if not context.story_state or not context.story_state.major_conflicts:
            return "(无明确核心冲突)"
        conflicts = context.story_state.major_conflicts[-self._limits.max_major_conflicts:]
        return "\n".join(f"- {c}" for c in conflicts)

    def _format_outline(self, outline: EndingOutline) -> str:
        lines = [
            f"主题: {outline.core_theme}",
            f"续写意图: {outline.continuation_intent or outline.core_theme}",
            f"本单元收束程度: {outline.closure_level or outline.ending_direction}",
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

        snippets: list[str] = []
        for segment in reversed(self.segment_index.segments[-_MAX_RECENT_SCENES:]):
            snippets.append(self._format_segment_tail_snippet(segment))
        snippets = snippets[:_MAX_RECENT_SCENES]

        recent_event = self._latest_irreversible_event(context)
        if recent_event:
            flag = f"[CH{recent_event.chapter:03d}]" if recent_event.chapter else "[事件]"
            snippets.append(f"{flag} {recent_event.summary}")

        return "\n".join(snippets) if snippets else "(无可参考片段)"

    def _build_continuation_context(self, text: str, context: StoryContext) -> str:
        """Build one evidence block reused by outline, draft, polish and review."""
        sections = context.for_prompt(limits=self._limits)
        base = format_context_sections(sections)
        hard_facts = [
            event for event in context.events if event.is_irreversible
        ][-_MAX_HARD_FACTS:]
        fact_lines = [
            f"- [CH{event.chapter:03d}] {event.type}: {event.summary} "
            f"({', '.join(event.participants) or 'unknown'})"
            for event in hard_facts
        ]
        source_tail = (text or "").strip()[-_SOURCE_TAIL_CHARS:]
        evidence = [
            base,
            "## Hard Facts\n" + ("\n".join(fact_lines) if fact_lines else "(无已记录不可逆事实)"),
            "## Recent Source Scene\n" + (source_tail or "(无原文尾部)"),
            "## Indexed Scene Tails\n" + self._recent_segment_digest(context),
        ]
        return "\n\n".join(item for item in evidence if item)

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

    def _format_segment_tail_snippet(self, segment: Segment, prefix: str | None = None) -> str:
        content = segment.content.strip().replace("\n", " ")
        snippet = content[-SNIPPET_EXCERPT_LEN:]
        label = prefix or f"[Segment {segment.segment_id}, source chapter {segment.chapter or '?'}]"
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

    def _append_bridges(self, polished: str, bridges: list[str], notes: str | None = None) -> str:
        baseline = polished.rstrip()
        bridge_text = "\n\n".join(bridges)
        combined_parts = [baseline, bridge_text.strip()]
        if notes:
            logger.debug("Resolution notes (not included in output): %s", notes.strip())
        return "\n\n".join(part for part in combined_parts if part)

    def _phase_options(self, phase: str) -> dict | None:
        return self._phase_llm_options.get(phase)

    def _required_str_list(self, value, field_name: str) -> list[str]:
        if not isinstance(value, list):
            raise LLMResponseError(f"Resolution {field_name} must be a list.")
        items: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise LLMResponseError(f"Resolution {field_name} items must be strings.")
            text = item.strip()
            if text:
                items.append(text)
        return items

    def _required_str(self, value, field_name: str, *, allow_empty: bool = False) -> str:
        if not isinstance(value, str):
            raise LLMResponseError(f"Resolution {field_name} must be a string.")
        text = value.strip()
        if not allow_empty and not text:
            raise LLMResponseError(f"Resolution {field_name} must not be empty.")
        return text

    def _validate_final(self, text: str, directives, context_block: str) -> None:
        result = self._ending_validator.validate(text, directives, context_block=context_block)
        if not result.passed:
            details = "; ".join(result.issues + result.repair_instructions)
            raise GenerationError("Ending validation failed: " + details)

    def _validate_or_repair_final(
        self,
        text: str,
        directives,
        context_block: str,
        style: WritingStyle | None,
        hint: str,
        enforcer: StyleEnforcer,
    ) -> str:
        current = text
        last_details = ""
        for attempt in range(_MAX_FINAL_REPAIR_ATTEMPTS + 1):
            result = self._ending_validator.validate(current, directives, context_block=context_block)
            if result.passed:
                return current
            last_details = "; ".join(result.issues + result.repair_instructions)
            if attempt >= _MAX_FINAL_REPAIR_ATTEMPTS:
                break
            current = self._phase_final_repair(
                current,
                context_block,
                hint,
                style,
                result.issues,
                result.repair_instructions,
                self._closure_instruction(directives),
            )
            current = enforcer.post_process(current)
        raise GenerationError("Ending validation failed: " + last_details)

    def _phase_final_repair(
        self,
        final_text: str,
        context_block: str,
        hint: str,
        style: WritingStyle | None,
        issues: list[str],
        repair_instructions: list[str],
        closure_instruction: str = "按用户要求决定是否收束，不要改变开放/闭合方向。",
    ) -> str:
        prompt = build_cacheable_prompt(
            context_block or "(无上下文)",
            self.final_repair_template,
            prefix_placeholder="context_block",
            final_text=final_text,
            hint=hint,
            issues="\n".join(f"- {item}" for item in issues) or "(无)",
            repair_instructions="\n".join(f"- {item}" for item in repair_instructions) or "(无)",
            style_guide=format_style_guide(style),
            style_samples=format_style_samples(style),
            style_constraints=format_style_constraints(style),
            banned_expressions=BANNED_EXPRESSIONS_PROMPT,
            closure_instruction=closure_instruction,
        )
        response = self.llm.generate(
            prompt=prompt,
            options=telemetry_options(
                self._phase_options("final_repair"),
                phase="continue",
                step=": final repair",
            ),
        )
        content = response.content.strip()
        if not content:
            raise LLMResponseError("Final repair phase returned empty content.")
        return content

    def _closure_instruction(self, directives) -> str:
        closure = getattr(directives, "closure", "unspecified")
        if closure == "open":
            return "保持开放或留白的本轮落点；不要把尚未到时的问题强行解释完，也不要无故改成确定终止。"
        if closure == "closed":
            return "保持本轮要求的闭合程度；明确交代要求的结果，但不要加入原文没有依据的新人物或复盘。"
        return "保持当前稿件自然的落点；不得擅自把尚未到终止点的续写改成全书结局。"


# Generic public name; keep EndingWriter for source compatibility.
ContinuationWriter = EndingWriter
