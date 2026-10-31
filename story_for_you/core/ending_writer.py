from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
import json
import hashlib
import re
from pathlib import Path

from story_for_you.analysis.context import StoryContext, WritingStyle
from story_for_you.config.settings import EndingPhaseTemperatures, RenderingLimits
from story_for_you.core.ending import (
    BANNED_EXPRESSIONS_PROMPT,
    EndingValidator,
    HintInterpreter,
    StyleEnforcer,
)
from story_for_you.core.ending.memory import ContinuationMemory
from story_for_you.core.ending.checkpoint import ContinuationCheckpoint
from story_for_you.core.ending.revision import TextEdit, apply_edits
from story_for_you.core.ending.retrieval import EvidenceRetriever
from story_for_you.core.exceptions import GenerationError, LLMResponseError
from story_for_you.core.prompting import (
    format_context_sections,
    format_style_constraints,
    format_style_guide,
    format_style_samples,
    load_template,
)
from story_for_you.indexer.segment import Segment, SegmentIndex
from story_for_you.indexer.tagger import unambiguous_alias_owners
from story_for_you.llm.base import LLMProvider
from story_for_you.llm.budget import BudgetedLLMProvider
from story_for_you.llm.telemetry import telemetry_options
from story_for_you.parser.text_splitter import strip_trailing_promotion
from story_for_you.utils.json_utils import load_json_response
from story_for_you.utils.prompting import SNIPPET_EXCERPT_LEN, build_cacheable_prompt, cache_prompt
from story_for_you.utils.artifacts import digest_json, implementation_digest

logger = logging.getLogger(__name__)

_MAX_CONFLICT_ANCHORS = 2
_VALID_RESOLUTION_STATUSES = {"ok", "needs_changes", "blocked"}
_MAX_FINAL_REPAIR_ATTEMPTS = 1
# A final chapter often plants several linked clues well before its last
# paragraph (for example the poison remark and the Garuda/dragon mural in
# 《英雄志》). Keep a whole chapter-sized window so planning can connect them;
# the caller still supplies the full source for quote validation.
_SOURCE_TAIL_CHARS = 24000
_MAX_RECENT_SCENES = 4
_SOURCE_ANCHOR_KINDS = {"object", "message", "clue", "threat", "promise", "symbol"}
_SOURCE_ANCHOR_HANDLINGS = {"advance", "hold", "background"}
_NARRATIVE_SIGNAL_KINDS = {
    "foreshadowing", "metaphor", "double_meaning", "identity", "threat", "promise", "misdirection",
}
_NARRATIVE_CONFIDENCE = {"high", "medium", "low"}
_SUBTEXT_CUE_TERMS = (
    "毒", "死", "钳", "迦楼罗", "金翅", "龙王", "真龙", "龙袍", "咬龙鸟",
    "有没有道理", "有何道理", "不是", "好像", "像是", "竟", "突然", "凄厉", "喃喃",
)


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
class SourceAnchor:
    """A high-salience object, message, clue, or action from the source tail."""

    anchor_id: str
    kind: str
    label: str
    evidence: str
    significance: str
    handling: str = "hold"
    interpretation: str = ""
    candidate_consequences: list[str] = field(default_factory=list)


@dataclass
class NarrativeSignal:
    """An evidence-backed but non-canonical reading of subtext or foreshadowing."""

    signal_id: str
    kind: str
    label: str
    evidence: str
    reading: str
    confidence: str = "medium"
    candidate_referents: list[str] = field(default_factory=list)
    candidate_consequences: list[str] = field(default_factory=list)


@dataclass
class PlotRecommendation:
    """A candidate continuation direction derived from source signals."""

    recommendation_id: str
    title: str
    premise: str
    based_on: list[str] = field(default_factory=list)
    beats: list[str] = field(default_factory=list)
    tradeoff: str = ""
    recommended: bool = False


@dataclass
class AnchorAction:
    """How one continuation unit handles a source anchor."""

    anchor_id: str
    handling: str
    action: str


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
    anchor_actions: list[AnchorAction] = field(default_factory=list)


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
    source_anchors: list[SourceAnchor] = field(default_factory=list)
    narrative_signals: list[NarrativeSignal] = field(default_factory=list)
    plot_recommendations: list[PlotRecommendation] = field(default_factory=list)

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
        context_window: int = 1_000_000,
        max_output_tokens: int = 32768,
        revision_mode: str = "full",
    ):
        if revision_mode not in {"full", "local", "none"}:
            raise ValueError("revision_mode must be full, local, or none")
        self.revision_mode = revision_mode
        self.trace: list[dict] = []
        self.llm = BudgetedLLMProvider(llm, context_window, max_output_tokens)
        self.segment_index = segment_index
        self._retriever = EvidenceRetriever(segment_index)
        self._limits = rendering_limits or RenderingLimits()
        self._hint_interpreter = HintInterpreter(self.llm, rendering_limits=self._limits)
        self._ending_validator = EndingValidator(self.llm)
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
        self.scene_template = load_template("ending_scene_plan")
        self.local_style_template = load_template("ending_style_revision")

    def continue_story(
        self,
        text: str,
        context: StoryContext,
        hint: str = "",
        *,
        max_chapters: int = 6,
        checkpoint_path: Path | None = None,
        resume: bool = False,
    ) -> str:
        if resume and checkpoint_path is None:
            raise ValueError("Resuming requires a checkpoint path.")
        if checkpoint_path is None:
            return self._continue_story(text, context, hint, max_chapters=max_chapters)
        canonical_context = context.to_dict()
        canonical_context["metadata"].pop("_fingerprint", None)
        provider = self.llm.provider
        defaults = getattr(provider, "options", {})
        signature = digest_json({
            "source": hashlib.sha256(text.encode()).hexdigest(), "context": canonical_context,
            "segments": [asdict(segment) for segment in self.segment_index.segments],
            "hint": hint, "max_units": max_chapters, "model": getattr(provider, "model", type(provider).__name__),
            "provider_options": {key: defaults[key] for key in (
                "seed", "temperature", "top_p", "top_k", "repeat_penalty", "max_tokens",
            ) if key in defaults},
            "temperatures": self._phase_llm_options, "rendering": asdict(self._limits),
            "window": self.llm.context_window, "output": self.llm.max_output_tokens,
            "revision": self.revision_mode, "implementation": implementation_digest(),
        })
        with ContinuationCheckpoint(provider, checkpoint_path, signature, resume=resume) as checkpoint:
            self.llm.provider = checkpoint
            try:
                result = self._continue_story(text, context, hint, max_chapters=max_chapters)
                checkpoint.finish(result, self.last_context.to_dict())
                return result
            except BaseException:
                checkpoint.discard_pending()
                raise
            finally:
                self.llm.provider = provider

    def _continue_story(
        self, text: str, context: StoryContext, hint: str = "", *, max_chapters: int = 6,
    ) -> str:
        """Plan and write an adaptive continuation."""
        if max_chapters < 1:
            raise ValueError("max_chapters must be positive")
        self.trace = []
        style = context.writing_style
        directives = self._hint_interpreter.interpret(hint, context)
        hint_payload = directives.for_prompt()
        context_block = self._build_continuation_context(
            text, context, focus_characters=directives.focus_characters, retrieval_hint=hint_payload,
        )

        subtext_signals = self._phase_subtext(context_block, hint_payload, text)
        if subtext_signals:
            self.trace.append({"stage": "subtext", "signals": [asdict(item) for item in subtext_signals]})
            context_block += "\n\n## 末章伏笔与隐喻预分析（解释假设，不是事实）\n" + self._format_narrative_signals(subtext_signals)
        plan = self._phase_plan(
            context_block, hint_payload, max_chapters, source_text=text,
            prior_signals=subtext_signals,
        )
        if directives.continuation_intent:
            plan.continuation_intent = directives.continuation_intent
        if directives.closure != "unspecified":
            plan.closure_level = directives.closure
        self.trace.append({"stage": "plan", "plan": asdict(plan)})

        memory = ContinuationMemory(context)
        generated: list[str] = []
        unit_states: list[dict] = []
        for chapter in plan.chapters:
            current_context = memory.snapshot()
            unit_states.append({
                "unit": chapter.number,
                "before": {
                    name: asdict(current_context.characters[name])
                    for name in chapter.focus_characters if name in current_context.characters
                },
            })
            chapter_context = self._build_continuation_context(
                text, current_context, focus_characters=chapter.focus_characters,
                retrieval_hint=json.dumps(asdict(chapter), ensure_ascii=False) + "\n" + "\n".join(generated)[-1600:],
                source_anchors=plan.source_anchors,
            ) + "\n\n" + self._format_plan_readings(plan) + "\n\n" + memory.render(
                query=json.dumps(asdict(chapter), ensure_ascii=False), max_chars=self._limits.evidence_chars,
            )
            chapter_context += "\n\n## 跨单元计划（未来安排，不是已发生事实）\n" + self._format_continuation_plan(plan)
            if generated:
                prior = "\n\n".join(generated)[-8000:]
                chapter_context += "\n\n## 已生成的前文（只用于承接，不要重复）\n" + prior
            outline = self._outline_from_chapter(plan, chapter)
            instruction = self._chapter_instruction(plan, chapter, directives.closure)
            instruction += "\n" + self._phase_scene_plan(chapter_context, instruction, hint_payload)
            style_anchors = self._build_style_anchors(current_context, chapter.focus_characters)
            anchor_actions = (
                self._format_anchor_actions(plan.source_anchors, chapter.anchor_actions)
                + "\n"
                + self._format_plan_readings(plan)
            )
            draft = self._phase_draft(
                current_context, outline, style, chapter_context, hint_payload, style_anchors, instruction,
                focus_characters=chapter.focus_characters, source_anchors=anchor_actions,
            )
            unit_trace = {"unit": chapter.number, "instruction": instruction, "draft": draft}
            self.trace.append(unit_trace)
            polished = self._phase_polish(
                draft, style, outline, chapter_context, hint_payload, style_anchors, instruction,
                focus_characters=chapter.focus_characters, source_anchors=anchor_actions,
            )
            unit_trace["revised"] = polished
            update = memory.review(
                self.llm, polished, chapter_context + "\n" + instruction, chapter.number,
                source_anchors=anchor_actions,
            )
            if not update.validation.passed:
                polished = self._phase_final_repair(
                    polished, chapter_context + "\n" + instruction, hint_payload, style,
                    update.validation.issues, update.validation.repair_instructions,
                    "只修复当前单元；遵守本单元计划，不提前完成后续单元的收束。",
                )
                update = memory.review(
                    self.llm, polished, chapter_context + "\n" + instruction, chapter.number,
                    source_anchors=anchor_actions,
                )
                if not update.validation.passed:
                    raise GenerationError("Unit validation failed: " + "; ".join(update.validation.issues))
            memory.commit(update, chapter.number)
            unit_trace["validated"] = polished
            generated.append(f"第{chapter.number}{plan.unit_type} {chapter.title}\n\n{polished}")

        final = "\n\n".join(generated).strip()
        validation_context = self._build_validation_context(
            text, context, plan, directives.focus_characters, unit_states,
        )
        final = self._phase_resolution_review(
            final, memory.snapshot(), validation_context, style, hint_payload, plan.closure_level,
            source_anchors=plan.source_anchors,
            narrative_signals=plan.narrative_signals,
            plot_recommendations=plan.plot_recommendations,
        )
        headings = [f"第{chapter.number}{plan.unit_type} {chapter.title}" for chapter in plan.chapters]
        self._validate_unit_headings(final, headings)

        enforcer = StyleEnforcer(style)
        final = enforcer.post_process(final)
        final = self._validate_or_repair_final(
            final,
            directives,
            validation_context,
            style,
            hint_payload,
            enforcer,
            expected_headings=headings,
        )
        # Final edits may change earlier facts. Re-extract from the final prose
        # before exporting state for another continuation run.
        if final != "\n\n".join(generated).strip():
            memory = self._rebuild_final_memory(text, context, final, plan)
        self.last_context = memory.snapshot()
        self.last_context.metadata["continuation_source_sha256"] = hashlib.sha256((text + "\n\n" + final).encode()).hexdigest()
        return final

    def _rebuild_final_memory(
        self, source_text: str, source: StoryContext, final: str, plan: ContinuationPlan,
    ) -> ContinuationMemory:
        memory = ContinuationMemory(source)
        headings = [f"第{chapter.number}{plan.unit_type} {chapter.title}" for chapter in plan.chapters]
        by_line = {}
        offset = 0
        for line in final.splitlines(keepends=True):
            if line.rstrip("\r\n") in headings:
                by_line[line.rstrip("\r\n")] = offset
            offset += len(line)
        offsets = [by_line[heading] for heading in headings]
        prior = []
        for index, chapter in enumerate(plan.chapters):
            body = final[offsets[index] + len(headings[index]):offsets[index + 1] if index + 1 < len(offsets) else len(final)].strip()
            block = self._build_continuation_context(
                source_text, memory.snapshot(), focus_characters=chapter.focus_characters,
                retrieval_hint=json.dumps(asdict(chapter), ensure_ascii=False),
                source_anchors=plan.source_anchors,
            ) + "\n" + memory.render(query=body[-1000:], max_chars=self._limits.evidence_chars)
            block += "\n前一单元正文：\n" + "\n\n".join(prior)[-8000:]
            block += "\n" + self._chapter_instruction(plan, chapter, "unspecified")
            update = memory.review(
                self.llm, body, block, chapter.number,
                source_anchors=(
                    self._format_anchor_actions(plan.source_anchors, chapter.anchor_actions)
                    + "\n"
                    + self._format_plan_readings(plan)
                ),
            )
            if not update.validation.passed:
                raise GenerationError("Revised continuation state failed review: " + "; ".join(update.validation.issues))
            memory.commit(update, chapter.number)
            prior.append(body)
        return memory

    def _build_validation_context(
        self, text: str, source: StoryContext, plan: ContinuationPlan,
        directive_cast: list[str], unit_states: list[dict],
    ) -> str:
        cast = list(dict.fromkeys([
            *directive_cast,
            *[name for chapter in plan.chapters for name in chapter.focus_characters],
        ]))
        return (
            "## 原作结束时的证据（续写开始之前）\n"
            + self._build_continuation_context(
                text, source, focus_characters=cast, retrieval_hint=self._format_continuation_plan(plan),
                source_anchors=plan.source_anchors,
            )
            + "\n\n## Continuation Plan（计划不是事实）\n"
            + self._format_continuation_plan(plan)
            + "\n\n## 各单元开始状态（修订前记录；若正文修改，须按新正文重新核验）\n"
            + json.dumps(unit_states, ensure_ascii=False)
        )

    def _validate_unit_headings(self, text: str, headings: list[str]) -> None:
        lines = text.splitlines()
        if any(lines.count(heading) != 1 for heading in headings):
            raise GenerationError("Continuation revision removed or duplicated a planned unit heading.")
        positions = [lines.index(heading) for heading in headings]
        if positions != sorted(positions):
            raise GenerationError("Continuation revision changed the planned unit order.")

    def _phase_subtext(
        self, context_block: str, hint: str, source_text: str,
    ) -> list[NarrativeSignal]:
        source_scene = strip_trailing_promotion(source_text or "").strip()[-_SOURCE_TAIL_CHARS:]
        cue_block = self._extract_subtext_cues(source_scene)
        if cue_block.startswith("(") or cue_block.count("\n") < 1:
            return []
        prompt = build_cacheable_prompt(
            context_block or "(无上下文)",
            load_template("ending_subtext"),
            prefix_placeholder="context_block",
            hint=hint,
        )
        for attempt in range(2):
            response = self.llm.generate(
                prompt,
                options=telemetry_options(
                    self._phase_options("outline"), phase="continue", step=": analyze subtext",
                    attempt=attempt + 1, max_attempts=2,
                ),
            )
            try:
                payload = load_json_response(response.content)
                if not isinstance(payload, dict):
                    raise LLMResponseError("Subtext analysis must return an object.")
                return self._parse_narrative_signals(payload.get("signals", []), source_scene)
            except LLMResponseError as exc:
                if attempt:
                    raise
                prompt = cache_prompt(
                    prompt.task + "\n\n上次伏笔分析未通过证据或格式校验：" + str(exc)
                    + "\n只返回完整 JSON；每条 evidence 必须逐字摘自原文，不要把解释写成事实。\n上次输出：\n"
                    + response.content,
                    prefix=prompt.prefix,
                )
        raise AssertionError("Unreachable subtext analysis retry state")

    def _phase_plan(
        self, context_block: str, hint: str, max_chapters: int, *, source_text: str = "",
        prior_signals: list[NarrativeSignal] | None = None,
    ) -> ContinuationPlan:
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
        source_anchors = self._parse_source_anchors(
            payload.get("source_anchors", []), source_text or context_block,
        )
        anchor_ids = {anchor.anchor_id for anchor in source_anchors}
        parsed_signals = self._parse_narrative_signals(
            payload.get("narrative_signals", []), source_text or context_block,
        )
        signal_by_id = {signal.signal_id: signal for signal in (prior_signals or [])}
        signal_by_id.update({signal.signal_id: signal for signal in parsed_signals})
        narrative_signals = list(signal_by_id.values())
        signal_ids = {signal.signal_id for signal in narrative_signals}
        plot_recommendations = self._parse_plot_recommendations(
            payload.get("plot_recommendations", []), anchor_ids | signal_ids,
        )
        if narrative_signals and not plot_recommendations:
            raise LLMResponseError(
                "Continuation plan must include plot_recommendations when narrative_signals are present."
            )
        chapters: list[EndingChapterPlan] = []
        assigned_anchor_ids: set[str] = set()
        for index, item in enumerate(chapters_payload, start=1):
            if not isinstance(item, dict):
                raise LLMResponseError("Ending plan chapter must be an object.")
            for field_name in ("title", "purpose", "viewpoint", "setting", "beats", "focus_characters"):
                if field_name not in item:
                    raise LLMResponseError(f"Continuation plan unit missing required field: {field_name}")
            resolutions = item.get("resolved_threads", item.get("resolutions", []))
            carry_forward = item.get("carry_forward", [])
            anchor_actions = self._parse_anchor_actions(
                item.get("anchor_actions", []), anchor_ids, index,
            )
            assigned_anchor_ids.update(action.anchor_id for action in anchor_actions)
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
                    anchor_actions=anchor_actions,
                )
            )
        missing_anchor_actions = [
            anchor.anchor_id for anchor in source_anchors if anchor.anchor_id not in assigned_anchor_ids
        ]
        if missing_anchor_actions:
            raise LLMResponseError(
                "Continuation plan must assign every source anchor to at least one unit: "
                + ", ".join(missing_anchor_actions)
            )
        return ContinuationPlan(
            unit_count=count,
            rationale=self._required_str(payload.get("rationale"), "rationale"),
            continuation_intent=intent,
            narrative_horizon=horizon,
            closure_level=closure,
            unit_type=unit_type,
            chapters=chapters,
            source_anchors=source_anchors,
            narrative_signals=narrative_signals,
            plot_recommendations=plot_recommendations,
        )

    def _parse_source_anchors(self, value, source_text: str) -> list[SourceAnchor]:
        if not isinstance(value, list):
            raise LLMResponseError("Continuation plan source_anchors must be a list.")
        anchors: list[SourceAnchor] = []
        seen: set[str] = set()
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                raise LLMResponseError(f"source_anchors[{index}] must be an object.")
            for field_name in ("anchor_id", "kind", "label", "evidence", "significance", "handling"):
                if field_name not in item:
                    raise LLMResponseError(f"source_anchors[{index}] missing required field: {field_name}")
            anchor_id = self._required_str(item.get("anchor_id"), f"source_anchors[{index}].anchor_id")
            if anchor_id in seen:
                raise LLMResponseError(f"Duplicate source anchor id: {anchor_id}")
            seen.add(anchor_id)
            kind = self._required_str(item.get("kind"), f"source_anchors[{index}].kind")
            if kind not in _SOURCE_ANCHOR_KINDS:
                raise LLMResponseError(
                    f"source_anchors[{index}].kind must be one of {sorted(_SOURCE_ANCHOR_KINDS)}"
                )
            handling = self._required_str(item.get("handling"), f"source_anchors[{index}].handling")
            if handling not in _SOURCE_ANCHOR_HANDLINGS:
                raise LLMResponseError(
                    f"source_anchors[{index}].handling must be one of {sorted(_SOURCE_ANCHOR_HANDLINGS)}"
                )
            evidence = self._required_str(item.get("evidence"), f"source_anchors[{index}].evidence")
            evidence = self._canonical_source_quote(evidence, source_text) or evidence
            if not self._quote_in_source(evidence, source_text):
                raise LLMResponseError(
                    f"source_anchors[{index}].evidence must quote the supplied source scene: {evidence}"
                )
            anchors.append(
                SourceAnchor(
                    anchor_id=anchor_id,
                    kind=kind,
                    label=self._required_str(item.get("label"), f"source_anchors[{index}].label"),
                    evidence=evidence,
                    significance=self._required_str(
                        item.get("significance"), f"source_anchors[{index}].significance"
                    ),
                    handling=handling,
                    interpretation=self._optional_required_str(
                        item.get("interpretation", ""), f"source_anchors[{index}].interpretation"
                    ),
                    candidate_consequences=self._optional_str_list(
                        item.get("candidate_consequences", []),
                        f"source_anchors[{index}].candidate_consequences",
                    ),
                )
            )
        return anchors

    def _parse_narrative_signals(self, value, source_text: str) -> list[NarrativeSignal]:
        if not isinstance(value, list):
            raise LLMResponseError("Continuation plan narrative_signals must be a list.")
        signals: list[NarrativeSignal] = []
        seen: set[str] = set()
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                raise LLMResponseError(f"narrative_signals[{index}] must be an object.")
            for field_name in (
                "signal_id", "kind", "label", "evidence", "reading", "confidence", "candidate_consequences",
            ):
                if field_name not in item:
                    raise LLMResponseError(f"narrative_signals[{index}] missing required field: {field_name}")
            signal_id = self._required_str(item.get("signal_id"), f"narrative_signals[{index}].signal_id")
            if signal_id in seen:
                raise LLMResponseError(f"Duplicate narrative signal id: {signal_id}")
            seen.add(signal_id)
            kind = self._required_str(item.get("kind"), f"narrative_signals[{index}].kind")
            if kind not in _NARRATIVE_SIGNAL_KINDS:
                raise LLMResponseError(
                    f"narrative_signals[{index}].kind must be one of {sorted(_NARRATIVE_SIGNAL_KINDS)}"
                )
            confidence = self._required_str(
                item.get("confidence"), f"narrative_signals[{index}].confidence"
            )
            if confidence not in _NARRATIVE_CONFIDENCE:
                raise LLMResponseError(
                    f"narrative_signals[{index}].confidence must be one of {sorted(_NARRATIVE_CONFIDENCE)}"
                )
            evidence = self._required_str(item.get("evidence"), f"narrative_signals[{index}].evidence")
            evidence = self._canonical_source_quote(evidence, source_text) or evidence
            if not self._quote_in_source(evidence, source_text):
                raise LLMResponseError(
                    f"narrative_signals[{index}].evidence must quote the supplied source scene: {evidence}"
                )
            signals.append(
                NarrativeSignal(
                    signal_id=signal_id,
                    kind=kind,
                    label=self._required_str(item.get("label"), f"narrative_signals[{index}].label"),
                    evidence=evidence,
                    reading=self._required_str(item.get("reading"), f"narrative_signals[{index}].reading"),
                    confidence=confidence,
                    candidate_referents=self._optional_str_list(
                        item.get("candidate_referents", []),
                        f"narrative_signals[{index}].candidate_referents",
                    ),
                    candidate_consequences=self._required_str_list(
                        item.get("candidate_consequences"),
                        f"narrative_signals[{index}].candidate_consequences",
                    ),
                )
            )
        return signals

    def _parse_plot_recommendations(
        self, value, known_ids: set[str]
    ) -> list[PlotRecommendation]:
        if not isinstance(value, list):
            raise LLMResponseError("Continuation plan plot_recommendations must be a list.")
        recommendations: list[PlotRecommendation] = []
        seen: set[str] = set()
        recommended_count = 0
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                raise LLMResponseError(f"plot_recommendations[{index}] must be an object.")
            for field_name in (
                "recommendation_id", "title", "premise", "based_on", "beats", "tradeoff", "recommended",
            ):
                if field_name not in item:
                    raise LLMResponseError(f"plot_recommendations[{index}] missing required field: {field_name}")
            recommendation_id = self._required_str(
                item.get("recommendation_id"), f"plot_recommendations[{index}].recommendation_id"
            )
            if recommendation_id in seen:
                raise LLMResponseError(f"Duplicate plot recommendation id: {recommendation_id}")
            seen.add(recommendation_id)
            based_on = self._required_str_list(
                item.get("based_on"), f"plot_recommendations[{index}].based_on"
            )
            unknown = [item_id for item_id in based_on if item_id not in known_ids]
            if unknown:
                raise LLMResponseError(
                    f"plot_recommendations[{index}] references unknown evidence ids: {', '.join(unknown)}"
                )
            recommended = item.get("recommended")
            if not isinstance(recommended, bool):
                raise LLMResponseError(f"plot_recommendations[{index}].recommended must be a boolean.")
            recommended_count += int(recommended)
            recommendations.append(
                PlotRecommendation(
                    recommendation_id=recommendation_id,
                    title=self._required_str(item.get("title"), f"plot_recommendations[{index}].title"),
                    premise=self._required_str(item.get("premise"), f"plot_recommendations[{index}].premise"),
                    based_on=based_on,
                    beats=self._required_str_list(item.get("beats"), f"plot_recommendations[{index}].beats"),
                    tradeoff=self._required_str(item.get("tradeoff"), f"plot_recommendations[{index}].tradeoff"),
                    recommended=recommended,
                )
            )
        if recommended_count > 1:
            raise LLMResponseError("Continuation plan may mark at most one plot recommendation as recommended.")
        return recommendations

    def _parse_anchor_actions(
        self, value, anchor_ids: set[str], unit_number: int
    ) -> list[AnchorAction]:
        if not isinstance(value, list):
            raise LLMResponseError(f"chapters[{unit_number - 1}].anchor_actions must be a list.")
        actions: list[AnchorAction] = []
        seen: set[str] = set()
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                raise LLMResponseError(
                    f"chapters[{unit_number - 1}].anchor_actions[{index}] must be an object."
                )
            for field_name in ("anchor_id", "handling", "action"):
                if field_name not in item:
                    raise LLMResponseError(
                        f"chapters[{unit_number - 1}].anchor_actions[{index}] missing required field: {field_name}"
                    )
            anchor_id = self._required_str(
                item.get("anchor_id"), f"chapters[{unit_number - 1}].anchor_actions[{index}].anchor_id"
            )
            if anchor_id not in anchor_ids:
                raise LLMResponseError(
                    f"chapters[{unit_number - 1}].anchor_actions references unknown source anchor: {anchor_id}"
                )
            if anchor_id in seen:
                raise LLMResponseError(
                    f"chapters[{unit_number - 1}].anchor_actions duplicates source anchor: {anchor_id}"
                )
            seen.add(anchor_id)
            handling = self._required_str(
                item.get("handling"), f"chapters[{unit_number - 1}].anchor_actions[{index}].handling"
            )
            if handling not in _SOURCE_ANCHOR_HANDLINGS:
                raise LLMResponseError(
                    f"chapters[{unit_number - 1}].anchor_actions[{index}].handling must be one of "
                    f"{sorted(_SOURCE_ANCHOR_HANDLINGS)}"
                )
            actions.append(
                AnchorAction(
                    anchor_id=anchor_id,
                    handling=handling,
                    action=self._required_str(
                        item.get("action"),
                        f"chapters[{unit_number - 1}].anchor_actions[{index}].action",
                    ),
                )
            )
        return actions

    def _quote_in_source(self, quote: str, source_text: str) -> bool:
        source = "".join(str(source_text or "").split())
        return bool(quote.strip()) and "".join(quote.split()) in source

    def _canonical_source_quote(self, quote: str, source_text: str) -> str | None:
        """Snap a near-exact model quote back to a contiguous source passage.

        Structured models often change only the final Chinese quote mark or
        collapse an ellipsis while copying a long passage.  We can safely
        repair that narrow transcription error when a long exact prefix and a
        high similarity score identify one source window.  Short or genuinely
        invented evidence remains rejected by ``_quote_in_source``.
        """
        compact_quote = "".join(quote.split())
        if len(compact_quote) < 18:
            return None
        source_chars: list[str] = []
        positions: list[int] = []
        for position, char in enumerate(str(source_text or "")):
            if char.isspace():
                continue
            source_chars.append(char)
            positions.append(position)
        compact_source = "".join(source_chars)
        starts: list[int] = []
        initial_prefix_length = min(32, max(12, len(compact_quote) // 3))
        for prefix_length in range(initial_prefix_length, 11, -1):
            prefix = compact_quote[:prefix_length]
            cursor = compact_source.find(prefix)
            while cursor >= 0:
                starts.append(cursor)
                cursor = compact_source.find(prefix, cursor + 1)
            if starts:
                break
        if not starts:
            return None

        max_delta = min(40, max(8, len(compact_quote) // 4))
        best_score = 0.0
        best_window: tuple[int, int] | None = None
        best_closed_score = 0.0
        best_closed_window: tuple[int, int] | None = None
        closing_quotes = set("”’\"'」』）》)】")
        for start in starts:
            for delta in range(-max_delta, max_delta + 1):
                end = start + len(compact_quote) + delta
                if start < 0 or end > len(compact_source) or end <= start:
                    continue
                candidate = compact_source[start:end]
                score = SequenceMatcher(None, compact_quote, candidate).ratio()
                if score > best_score:
                    best_score = score
                    best_window = (start, end)
                if compact_quote[-1] in closing_quotes and candidate[-1] in closing_quotes:
                    if score > best_closed_score:
                        best_closed_score = score
                        best_closed_window = (start, end)
        if compact_quote[-1] in closing_quotes and best_closed_window is not None and best_closed_score >= 0.92:
            best_window, best_score = best_closed_window, best_closed_score
        if best_window is not None and best_score >= 0.92:
            start, end = best_window
            return str(source_text)[positions[start]:positions[end - 1] + 1].strip()

        # Models sometimes join two source paragraphs with an ellipsis while
        # calling the result one piece of evidence.  Keep the longest
        # contiguous, source-backed clause instead of accepting the bridge.
        clauses = [part for part in re.split(r"(?:…+|\.{3,})", compact_quote) if len(part) >= 18]
        matches = [part for part in clauses if part in compact_source]
        if matches:
            return max(matches, key=len)
        return None

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
        anchor_actions = self._format_anchor_actions(plan.source_anchors, chapter.anchor_actions)
        return (
            f"本次续写计划共 {plan.chapter_count} 个{plan.unit_type}，本单元是第 {chapter.number} 个。"
            f"目的：{chapter.purpose}；视角：{chapter.viewpoint}；场景：{chapter.setting}。"
            f"重点人物：{'、'.join(chapter.focus_characters) or '以当前场景人物为准'}。"
            f"本单元结果：{chapter.end_state or '由行动和场景自然形成'}。{close}。"
            f"读者收束要求为 {closure}；不要擅自改变它。\n"
            "本单元原文高显著锚点处理：\n" + anchor_actions
            + "\n"
            + self._format_plan_readings(plan)
        )

    def _format_continuation_plan(self, plan: ContinuationPlan) -> str:
        return json.dumps(asdict(plan), ensure_ascii=False, indent=2)

    def _phase_scene_plan(self, context_block: str, instruction: str, hint: str) -> str:
        prompt = build_cacheable_prompt(
            context_block, self.scene_template, prefix_placeholder="context_block",
            chapter_instruction=instruction, hint=hint,
        )
        for attempt in range(2):
            response = self.llm.generate(prompt, options=telemetry_options(
                self._phase_options("outline"), phase="continue", step=": plan unit scenes",
                attempt=attempt + 1, max_attempts=2,
            ))
            try:
                return self._parse_scene_plan(response.content)
            except LLMResponseError as exc:
                if attempt:
                    raise
                prompt = cache_prompt(
                    prompt.task + "\n\n上次输出未通过校验：" + str(exc)
                    + "\n保留有依据的场景内容，补齐每一场的全部字段。严格返回完整 JSON 对象。\n上次输出：\n"
                    + response.content,
                    prefix=prompt.prefix,
                )
        raise AssertionError("Unreachable scene planning retry state")

    def _parse_scene_plan(self, content: str) -> str:
        payload = load_json_response(content)
        scenes = payload.get("scenes") if isinstance(payload, dict) else None
        if not isinstance(scenes, list) or not scenes:
            raise LLMResponseError("Scene plan requires a non-empty scenes list.")
        lines = ["本单元场景计划（只执行当前单元，不提前兑现后续结果）："]
        fields = ("viewpoint", "setting", "goal", "obstacle", "action", "turn", "cost", "end_state", "knowledge_boundary")
        for number, scene in enumerate(scenes, start=1):
            if not isinstance(scene, dict):
                raise LLMResponseError("Scene plan item must be an object.")
            details = [f"{key}={self._required_str(scene.get(key), f'scenes[{number - 1}].{key}')}" for key in fields]
            lines.append(f"- 场景{number}: " + "; ".join(details))
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
        *, focus_characters: list[str] | None = None,
        source_anchors: str = "",
    ) -> str:
        """阶段2: 按大纲写作初稿，应用风格指南。"""
        outline_text = self._format_outline(outline)
        style_guide = format_style_guide(style)
        style_samples = format_style_samples(style, focus_characters=focus_characters, scene_query=chapter_instruction)
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
            source_anchors=source_anchors or "(本单元没有需要特别处理的原文高显著锚点)",
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
        *, focus_characters: list[str] | None = None,
        source_anchors: str = "",
    ) -> str:
        """阶段3: 修订并润色初稿（合并原 revision + polish）。"""
        if self.revision_mode == "none":
            return draft
        style_guide = format_style_guide(style)
        style_samples = format_style_samples(style, focus_characters=focus_characters, scene_query=chapter_instruction)
        style_constraints = format_style_constraints(style)
        characteristic_words = ", ".join(style.characteristic_words) if style and style.characteristic_words else ""
        tone_markers = ", ".join(style.tone_markers) if style and style.tone_markers else ""
        checklist = self._revision_checklist(style)

        prompt = build_cacheable_prompt(
            context_block or "(无上下文)",
            self.local_style_template if self.revision_mode == "local" else self.polish_template,
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
            source_anchors=source_anchors or "(本单元没有需要特别处理的原文高显著锚点)",
            banned_expressions=BANNED_EXPRESSIONS_PROMPT,
            chapter_instruction=chapter_instruction,
        )

        response = self.llm.generate(
            prompt=prompt,
            options=telemetry_options(
                {**self._phase_options("polish"), **({"no_think": True} if self.revision_mode == "local" else {})},
                phase="continue",
                step=": polish",
            ),
        )
        content = response.content.strip()
        if not content:
            raise LLMResponseError("Polish phase returned empty content.")
        if self.revision_mode == "local":
            payload = load_json_response(content)
            if not isinstance(payload, dict) or not isinstance(payload.get("edits"), list):
                raise LLMResponseError("Local polish must return an edits list.")
            return apply_edits(draft, [TextEdit.from_dict(item) for item in payload["edits"]], allow_noop=True)
        return content

    def _phase_resolution_review(
        self,
        polished: str,
        context: StoryContext,
        context_block: str,
        style: WritingStyle | None,
        hint: str,
        closure_level: str = "closed",
        source_anchors: list[SourceAnchor] | None = None,
        narrative_signals: list[NarrativeSignal] | None = None,
        plot_recommendations: list[PlotRecommendation] | None = None,
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
            source_anchors=self._format_source_anchors(source_anchors or []),
            narrative_signals=self._format_narrative_signals(narrative_signals or []),
            plot_recommendations=self._format_plot_recommendations(plot_recommendations or []),
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
        for field_name in ("status", "missing_threads", "notes"):
            if field_name not in payload:
                raise LLMResponseError(f"Resolution phase missing required field: {field_name}")
        status = self._required_str(payload.get("status"), "status").lower()
        if status not in _VALID_RESOLUTION_STATUSES:
            raise LLMResponseError(f"Invalid resolution status: {status!r}")
        missing_threads = self._required_str_list(payload.get("missing_threads"), "missing_threads")
        edits_payload = payload.get("edits", [])
        if not isinstance(edits_payload, list):
            raise LLMResponseError("Resolution edits must be a list.")
        edits = [TextEdit.from_dict(item) for item in edits_payload]
        notes = self._required_str(payload.get("notes"), "notes", allow_empty=True)
        if status == "ok":
            if missing_threads or edits or payload.get("bridges"):
                raise LLMResponseError("Resolution status ok requires empty missing_threads and edits.")
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
        return apply_edits(polished, edits)

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
        return any(name in event.deceased_characters for event in context.events)

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
            return "依本单元行动与因果展开；段落、对白比例和篇幅遵循原作当前场景，不设统一字数。"

        lines = [f"- 叙事节点{idx}: {beat}" for idx, beat in enumerate(beats, start=1)]
        lines.append("- 节点不是段落配额；按原作节奏安排段落、对白、停顿与转场，不要求每段等长或逐句接尾。")
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

    def _build_continuation_context(
        self, text: str, context: StoryContext, *, focus_characters: list[str] | None = None,
        retrieval_hint: str = "", source_anchors: list[SourceAnchor] | None = None,
    ) -> str:
        """Build one evidence block reused by outline, draft, polish and review."""
        alias_owners = unambiguous_alias_owners(context)
        requested = focus_characters or [
            name for segment in self.segment_index.segments[-_MAX_RECENT_SCENES:]
            for name in segment.characters
        ]
        focus = list(dict.fromkeys(
            alias_owners.get(name.casefold(), name) for name in requested
        ))
        sections = context.for_prompt(limits=self._limits, focus_characters=focus)
        base = format_context_sections(sections)
        hard_facts = [
            event for event in context.events if event.is_irreversible
        ]
        fact_lines = [
            f"- [{event.event_id}] {event.type}: {event.summary} "
            f"({', '.join(event.participants) or 'unknown'})"
            + (f" | 原文证据: {event.evidence}" if event.evidence else "")
            for event in hard_facts
        ]
        source_tail = strip_trailing_promotion(text or "").strip()[-_SOURCE_TAIL_CHARS:]
        relevant_events, passages = self._retriever.retrieve(
            retrieval_hint + "\n" + source_tail[-2000:] + "\n" + " ".join(focus),
            [event for event in context.events if not event.is_irreversible], focus,
            max_events=self._limits.max_history_events,
            max_passages=self._limits.max_evidence_passages,
            max_chars=self._limits.evidence_chars,
        )
        history = "\n".join(
            f"- [{event.event_id}, chapter={event.chapter}, unit={event.source_unit}] {event.summary}"
            + (f" | 原文证据: {event.evidence}" if event.evidence else "")
            for event in relevant_events
        )
        excerpts = []
        for name in focus:
            segment = self._latest_segment_for_character(name)
            if segment:
                variants = [name] + [alias for alias, owner in alias_owners.items() if owner == name]
                position = max(segment.content.casefold().rfind(variant.casefold()) for variant in variants)
                if position >= 0:
                    excerpt = segment.content[max(0, position - 500):position + 700]
                    excerpts.append(f"- {name} [Segment {segment.segment_id}]: {excerpt}")
        evidence = [
            "原作片段是历史证据；已有续写时从最新生成单元承接，不要跳回原作末尾。未来计划不是已发生事实。",
            base,
            "## Hard Facts\n" + ("\n".join(fact_lines) if fact_lines else "(无已记录不可逆事实)"),
            "## Recent Source Scene\n" + (source_tail or "(无原文尾部)"),
            "## Candidate Subtext Cues（程序只召回原文片段，不代表任何解释成立）\n"
            + self._extract_subtext_cues(source_tail),
            "## Source Anchors（原文高显著锚点；不是普通背景物件）\n"
            + self._format_source_anchors(source_anchors or []),
            "## Indexed Scene Tails\n" + self._recent_segment_digest(context),
            "## Active Cast History\n" + (history or "(无相关历史事件)"),
            "## Active Cast Source Evidence\n" + ("\n".join(excerpts) or "(无相关原文片段)"),
            "## Retrieved Scene Evidence（历史证据，不代表当前角色已知；以各人物知识状态为准）\n" + (
                "\n".join(
                    f"- [Segment {p.segment.segment_id}, chapter={p.segment.chapter}, offset={p.offset}]: {p.text}"
                    for p in passages
                ) or "(无额外相关片段)"
            ),
            "## Open World Threads\n" + (
                "\n".join(context.story_state.unresolved_events) if context.story_state else "(无世界未决线索)"
            ),
        ]
        return "\n\n".join(item for item in evidence if item)

    def _extract_subtext_cues(self, source_text: str) -> str:
        """Recall suspicious source paragraphs before asking the model to interpret them."""
        paragraphs = [item.strip() for item in source_text.splitlines() if item.strip()]
        selected: list[str] = []
        for paragraph in paragraphs:
            if not any(term in paragraph for term in _SUBTEXT_CUE_TERMS):
                continue
            if paragraph in selected:
                continue
            selected.append(paragraph)
        if not selected:
            return "(未召回到明显的反常对白或意象片段)"
        # Keep the cue list bounded; each item remains a contiguous source quote.
        lines: list[str] = []
        remaining = 9000
        for index, paragraph in enumerate(selected, start=1):
            if len(paragraph) + 4 > remaining:
                break
            lines.append(f"- cue-{index}: {paragraph}")
            remaining -= len(paragraph) + 4
        return "\n".join(lines) or "(候选片段超过召回预算)"

    def _format_source_anchors(self, anchors: list[SourceAnchor]) -> str:
        if not anchors:
            return "(暂无已识别的原文高显著锚点)"
        lines = []
        for anchor in anchors:
            line = (
                f"- [{anchor.anchor_id}] {anchor.kind}/{anchor.label}；处理建议={anchor.handling}；"
                f"意义={anchor.significance}；原文证据={anchor.evidence}"
            )
            if anchor.interpretation:
                line += f"；可能解读={anchor.interpretation}"
            if anchor.candidate_consequences:
                line += "；候选后果=" + " / ".join(anchor.candidate_consequences)
            lines.append(line)
        return "\n".join(lines)

    def _format_narrative_signals(self, signals: list[NarrativeSignal]) -> str:
        if not signals:
            return "(暂无已识别的对白深意、隐喻或伏笔解释)"
        lines = []
        for signal in signals:
            lines.append(
                f"- [{signal.signal_id}] {signal.kind}/{signal.label}；置信度={signal.confidence}；"
                f"原文证据={signal.evidence}；可能解读={signal.reading}；"
                f"候选对应={' / '.join(signal.candidate_referents) or '未确定'}；"
                f"候选后果={' / '.join(signal.candidate_consequences)}"
            )
        return "\n".join(lines)

    def _format_plot_recommendations(self, recommendations: list[PlotRecommendation]) -> str:
        if not recommendations:
            return "(暂无基于末章证据的情节建议)"
        lines = []
        for recommendation in recommendations:
            marker = "优先" if recommendation.recommended else "备选"
            lines.append(
                f"- [{recommendation.recommendation_id}] {marker}：{recommendation.title}；"
                f"前提={recommendation.premise}；依据={','.join(recommendation.based_on)}；"
                f"节点={' / '.join(recommendation.beats)}；代价/风险={recommendation.tradeoff}"
            )
        return "\n".join(lines)

    def _format_plan_readings(self, plan: ContinuationPlan) -> str:
        return (
            "## 末章对白与意象的解释假设（不是已发生事实）\n"
            + self._format_narrative_signals(plan.narrative_signals)
            + "\n\n## 基于证据的候选情节（优先项可作为本轮方向，备选项不得擅自同时兑现）\n"
            + self._format_plot_recommendations(plan.plot_recommendations)
        )

    def _format_anchor_actions(
        self, anchors: list[SourceAnchor], actions: list[AnchorAction]
    ) -> str:
        if not anchors:
            return "(本单元没有需要特别处理的原文高显著锚点)"
        by_id = {anchor.anchor_id: anchor for anchor in anchors}
        lines = []
        for action in actions:
            anchor = by_id.get(action.anchor_id)
            if anchor is None:
                continue
            lines.append(
                f"- [{anchor.anchor_id}] {anchor.kind}/{anchor.label}；本单元={action.handling}；"
                f"具体处理={action.action}；原文证据={anchor.evidence}"
            )
        if not lines:
            return "(本单元没有分配锚点；若正文提及相关物件或消息，不得无故改变其意义)"
        lines.append(
            "- 处理要求：advance 必须通过人物行动、信息或后果推进；hold 必须在正文中保留可追踪的悬置；"
            "background 只能自然带过，不能把有特写或递交动作的锚点写成无意义背景。"
        )
        return "\n".join(lines)

    def _latest_segment_for_character(self, name: str) -> Segment | None:
        segment_ids = self.segment_index.char_index.get(name)
        if not segment_ids:
            return None
        for segment in reversed(self.segment_index.segments):
            if segment.segment_id in segment_ids:
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

    def _build_style_anchors(self, context: StoryContext, focus_characters: list[str] | None = None) -> str:
        anchors: list[str] = []
        style = context.writing_style
        if style and style.characteristic_words:
            keywords = "、".join(style.characteristic_words[:self._limits.max_characteristic_words])
            anchors.append(f"原作词汇参考：{keywords}（仅在语境合适时使用，无使用数量要求）")

        cast = focus_characters if focus_characters is not None else [
            char.name
            for char in context.characters.values()
            if char.role in ("main", "support")
        ]
        if cast:
            anchors.append("本单元人物参考：" + "、".join(cast))

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

    def _optional_required_str(self, value, field_name: str) -> str:
        if not isinstance(value, str):
            raise LLMResponseError(f"Resolution {field_name} must be a string.")
        return value.strip()

    def _optional_str_list(self, value, field_name: str) -> list[str]:
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
        *, expected_headings: list[str] | None = None,
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
            if expected_headings:
                self._validate_unit_headings(current, expected_headings)
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
        payload = load_json_response(content)
        if not isinstance(payload, dict) or not isinstance(payload.get("edits"), list):
            raise LLMResponseError("Final repair must return a JSON edits list.")
        return apply_edits(final_text, [TextEdit.from_dict(item) for item in payload["edits"]])

    def _closure_instruction(self, directives) -> str:
        closure = getattr(directives, "closure", "unspecified")
        if closure == "open":
            return "保持开放或留白的本轮落点；不要把尚未到时的问题强行解释完，也不要无故改成确定终止。"
        if closure == "closed":
            return "保持本轮要求的闭合程度；明确交代要求的结果，但不要加入原文没有依据的新人物或复盘。"
        return "保持当前稿件自然的落点；不得擅自把尚未到终止点的续写改成全书结局。"


# Generic public name; keep EndingWriter for source compatibility.
ContinuationWriter = EndingWriter
