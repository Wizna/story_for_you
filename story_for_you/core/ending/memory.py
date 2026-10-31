from __future__ import annotations

import json
import hashlib
from copy import deepcopy
from dataclasses import dataclass, field

from story_for_you.analysis.context import (
    CharacterState,
    PlotEvent,
    Relationship,
    StoryContext,
    StoryState,
)
from story_for_you.analysis.extractors.characters import CharacterExtractor
from story_for_you.analysis.layers.state_store import StateStore
from story_for_you.analysis.threads import ThreadUpdate
from story_for_you.core.ending.validator import (
    ContinuationValidationResult,
    ContinuationValidator,
)
from story_for_you.core.exceptions import LLMResponseError
from story_for_you.core.prompting import load_template
from story_for_you.core.ending.retrieval import rank_texts
from story_for_you.llm.base import LLMProvider
from story_for_you.llm.telemetry import telemetry_options
from story_for_you.utils.json_utils import load_json_response
from story_for_you.utils.prompting import build_cacheable_prompt, cache_prompt


@dataclass
class UnitUpdate:
    validation: ContinuationValidationResult
    summary: str = ""
    characters: list[CharacterState] = field(default_factory=list)
    events: list[PlotEvent] = field(default_factory=list)
    story_state: StoryState | None = None
    relationships: list[Relationship] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    unit_number: int | None = None


class ContinuationMemory:
    """Keep generated observations separate from the immutable source analysis."""

    def __init__(self, source: StoryContext):
        self.source = deepcopy(source)
        identity = source.to_dict()
        identity["metadata"].pop("_fingerprint", None)
        self.namespace = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
        self.store = StateStore()
        self.store.update(source.characters.values(), [], source.events)
        if source.story_state:
            self.store.set_story_state(deepcopy(source.story_state))
        # ``source_unit`` identifies a physical source unit. Generated
        # continuation units have their own user-facing number (1, 2, ...),
        # but their provenance must sort after the original novel rather than
        # appearing before source unit 1 on the next run.
        self._generated_source_unit_base = self._next_source_unit(source)
        self.summaries: list[str] = []
        self.events: list[PlotEvent] = []

    def _next_source_unit(self, source: StoryContext) -> int:
        """Return the first provenance unit after all source observations."""
        observed: list[int] = []
        # A source unit can contain no extracted event, relationship, or
        # thread at all.  In that case the observations below do not reveal
        # the end of the physical source.  Analyzer metadata still records the
        # number of processed units, so use it as a lower bound for generated
        # provenance and keep chronology stable across sparse analyses.
        metadata = source.metadata if isinstance(source.metadata, dict) else {}
        for key in ("source_unit_count",):
            value = metadata.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                observed.append(value)
        coverage = metadata.get("chapter_coverage")
        if isinstance(coverage, dict):
            value = coverage.get("count")
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                observed.append(value)
        for event in source.events:
            value = event.source_unit or event.chapter
            if value is not None:
                observed.append(value)
        for character in source.characters.values():
            for relationship in [
                *character.relationships,
                *character.relationship_history,
                *character.incoming_relationship_history,
            ]:
                if relationship.source_unit is not None:
                    observed.append(relationship.source_unit)
                elif relationship.chapter is not None:
                    observed.append(relationship.chapter)
            for thread in character.threads:
                for observation in thread.history:
                    value = observation.source_unit or observation.chapter
                    if value is not None:
                        observed.append(value)
        if source.story_state:
            for thread in source.story_state.threads:
                for observation in thread.history:
                    value = observation.source_unit or observation.chapter
                    if value is not None:
                        observed.append(value)
        return max(observed, default=0) + 1

    def _generated_source_unit(self, unit_number: int) -> int:
        return self._generated_source_unit_base + unit_number - 1

    def _compact_thread_records(self, threads) -> list[dict]:
        """Expose stable IDs without replaying every historical quotation."""
        ordered = sorted(
            threads,
            key=lambda thread: (
                thread.status in {"open", "dormant"},
                thread.history[-1].source_unit if thread.history and thread.history[-1].source_unit else 0,
            ),
            reverse=True,
        )
        return [
            {
                "thread_id": thread.thread_id,
                "summary": thread.summary,
                "status": thread.status,
                "kind": thread.kind,
                "confidence": thread.confidence,
                "candidate_outcomes": thread.candidate_outcomes[-3:],
            }
            for thread in ordered[:12]
        ]

    def snapshot(self) -> StoryContext:
        context = deepcopy(self.source)
        context.characters = self.store.characters_snapshot()
        context.story_state = self.store.story_snapshot()
        context.events.extend(deepcopy(self.events))
        return context

    def render(self, *, query: str = "", max_chars: int | None = None) -> str:
        summaries = self.summaries
        if max_chars is not None:
            recent = list(range(max(0, len(summaries) - 2), len(summaries)))
            ranked = [index for _, index in rank_texts(query, summaries)]
            selected = []
            remaining = max_chars
            for index in dict.fromkeys([*recent, *ranked]):
                if len(summaries[index]) <= remaining:
                    selected.append(index)
                    remaining -= len(summaries[index])
            summaries = [summaries[index] for index in sorted(selected)]
        return "\n".join(
            [
                f"## 已通过逐单元审查的续写记忆（展示 {len(summaries)}/{len(self.summaries)} 个单元，完整历史保留；不是未来计划）",
                *summaries,
                *([f"- [{event.event_id}] {event.summary}" for event in self.events] if max_chars is None else []),
            ]
        )

    def review(
        self, llm: LLMProvider, text: str, context_block: str, unit_number: int,
        *, source_anchors: str = "",
    ) -> UnitUpdate:
        world = self.store.story_snapshot()
        characters = self.store.characters_snapshot()
        text_folded = text.casefold()
        known_states = {}
        for name, char in characters.items():
            labels = [name, *char.aliases]
            if not any(
                isinstance(label, str) and label.strip() and label.casefold() in text_folded
                for label in labels
            ):
                continue
            known_states[name] = {
                "status": char.status,
                "status_evidence": char.status_evidence,
                "goal": char.goal,
                "location": char.location,
                "realm": char.realm,
                "knowledge": char.knowledge,
                "relationships": [
                    {
                        "targets": relation.targets,
                        "relation_type": relation.relation_type,
                        "sentiment": relation.sentiment,
                    }
                    for relation in char.relationships
                ],
            }
        known_character_threads = {
            name: self._compact_thread_records(char.threads)
            for name, char in characters.items()
            if char.threads and any(
                isinstance(label, str) and label.strip() and label.casefold() in text_folded
                for label in [name, *char.aliases]
            )
        }
        known_world_threads = self._compact_thread_records(world.threads) if world else []
        prompt = build_cacheable_prompt(
            context_block,
            load_template("ending_memory"),
            prefix_placeholder="context_block",
            unit_text=text,
            unit_number=str(unit_number),
            known_states=json.dumps(known_states, ensure_ascii=False),
            known_threads=json.dumps(
                {
                    "characters": known_character_threads,
                    "world": known_world_threads,
                },
                ensure_ascii=False,
            ),
            source_anchors=source_anchors or "(本单元没有需要特别处理的原文高显著锚点)",
        )
        for attempt in range(2):
            response = llm.generate(
                prompt,
                options=telemetry_options(
                    {"no_think": True, "temperature": 0.1},
                    phase="continue",
                    step=": review and remember unit",
                    attempt=attempt + 1,
                    max_attempts=2,
                ),
            )
            try:
                update = self._parse_update(llm, response.content, text, unit_number)
                return update
            except LLMResponseError as exc:
                if attempt:
                    raise
                prompt = cache_prompt(
                    prompt.task + "\n\n上次记忆输出未通过格式或证据校验：" + str(exc)
                    + "\n对照上述实际成稿和字段格式修正完整 JSON，逐字复制引文；不得修改成稿或绕过事实审查。"
                    + "\n上次输出：\n" + response.content,
                    prefix=prompt.prefix,
                )
        raise AssertionError("Unreachable memory review retry state")

    def _parse_update(self, llm: LLMProvider, content: str, text: str, unit_number: int) -> UnitUpdate:
        payload = load_json_response(content)
        if not isinstance(payload, dict):
            raise LLMResponseError("Unit memory review must return an object.")
        validation = ContinuationValidator(llm)._from_payload(payload)
        update = UnitUpdate(validation, unit_number=unit_number)
        if not validation.passed:
            return update
        summary = payload.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            raise LLMResponseError("Unit memory requires a non-empty summary.")
        for key in ("characters", "events"):
            if not isinstance(payload.get(key), list):
                raise LLMResponseError(f"Unit memory {key} must be a list.")
        extractor = CharacterExtractor(llm)
        characters = []
        for item in payload["characters"]:
            if not isinstance(item, dict):
                raise LLMResponseError("Unit memory character must be an object.")
            evidence = item.get("evidence")
            char = extractor._to_character(item)
            characters.append(char)
            update.evidence.append(evidence.strip() if isinstance(evidence, str) else "")
        events = []
        for index, item in enumerate(payload["events"], start=1):
            if not isinstance(item, dict):
                raise LLMResponseError("Unit memory event must be an object.")
            evidence = item.get("evidence")
            event = PlotEvent.from_dict(item)
            event.event_id = f"GEN-{self.namespace}-U{unit_number}-E{index}"
            event.chapter = unit_number
            event.source_unit = self._generated_source_unit(unit_number)
            if set(event.deceased_characters) - set(event.participants):
                raise LLMResponseError(
                    "Deceased characters must be event participants."
                )
            events.append(event)
            update.evidence.append(evidence.strip() if isinstance(evidence, str) else "")
        world = payload.get("story_state")
        relationships = payload.get("relationships", [])
        if not isinstance(relationships, list):
            raise LLMResponseError("Unit memory relationships must be a list.")
        for item in relationships:
            relation = Relationship.from_dict(item)
            update.relationships.append(relation)
            update.evidence.append(relation.evidence.strip())
        update.summary = summary.strip()
        update.characters = characters
        update.events = events
        update.story_state = StoryState.from_dict(world) if world is not None else None
        if update.story_state:
            quote = payload.get("world_evidence")
            update.evidence.append(quote.strip() if isinstance(quote, str) else "")
        errors = []
        for check in (lambda: self._validate_quotes(payload, text), lambda: self._candidate_store(update)):
            try:
                check()
            except LLMResponseError as exc:
                errors.append(str(exc))
        if errors:
            # A model sometimes invents an ID while correctly identifying a
            # new observation (for example, it hashes the rewritten summary
            # instead of copying the existing ID).  Do not let that discard
            # the whole validated unit.  Recover only this narrow class of
            # state-store error by recording the observation as a new open
            # thread; the StateStore itself remains strict for all callers.
            if all(error.startswith("Unknown thread ID for ") for error in errors):
                recovered = self._recover_unknown_thread_ids(update)
                if recovered:
                    try:
                        self._candidate_store(update)
                        return update
                    except LLMResponseError as exc:
                        errors.append(str(exc))
            raise LLMResponseError("\n".join(errors))
        return update

    def _recover_unknown_thread_ids(self, update: UnitUpdate) -> bool:
        """Keep evidence-backed observations when an LLM fabricates a thread ID.

        ``StateStore`` intentionally rejects unknown IDs so direct callers
        cannot silently resolve the wrong promise.  The continuation review
        has a safer fallback: an unknown ``update``/``resolve`` is treated as
        an ``add`` with a deterministic ID generated by ``StateStore``.  This
        preserves the new clue as an open thread and avoids claiming that an
        existing thread was resolved.  Exact known IDs are left untouched.
        """
        recovered = False
        characters = self.store.characters_snapshot()
        for character in update.characters:
            existing = characters.get(character.name)
            if existing is None:
                # Alias resolution is handled by StateStore when committing;
                # for recovery, only canonical names are safe to inspect.
                continue
            known_ids = {thread.thread_id for thread in existing.threads}
            replacements: list[ThreadUpdate] = []
            for change in character.thread_updates:
                if change.operation in {"update", "resolve"} and change.thread_id not in known_ids:
                    replacements.append(ThreadUpdate(
                        "add", "", change.summary, change.evidence,
                        kind=change.kind, confidence=change.confidence,
                        candidate_outcomes=list(change.candidate_outcomes),
                    ))
                    recovered = True
                else:
                    replacements.append(change)
            character.thread_updates = replacements

        if update.story_state is not None:
            existing_world = self.store.story_snapshot()
            known_ids = {
                thread.thread_id for thread in (existing_world.threads if existing_world else [])
            }
            replacements = []
            for change in update.story_state.thread_updates:
                if change.operation in {"update", "resolve"} and change.thread_id not in known_ids:
                    replacements.append(ThreadUpdate(
                        "add", "", change.summary, change.evidence,
                        kind=change.kind, confidence=change.confidence,
                        candidate_outcomes=list(change.candidate_outcomes),
                    ))
                    recovered = True
                else:
                    replacements.append(change)
            update.story_state.thread_updates = replacements

        return recovered

    def _validate_quotes(self, payload: dict, text: str) -> None:
        references = []
        for field_name in ("characters", "events", "relationships"):
            rows = payload.get(field_name, [])
            if not isinstance(rows, list):
                continue  # Shape validation reports malformed containers separately.
            for index, item in enumerate(rows):
                if not isinstance(item, dict):
                    continue
                label = f"{field_name}[{index}]"
                references.append((label + ".evidence", item.get("evidence")))
                if item.get("status") == "dead":
                    references.append((label + ".status_evidence", item.get("status_evidence")))
                for changes_name in ("state_changes", "thread_updates"):
                    changes = item.get(changes_name, [])
                    if isinstance(changes, list):
                        references.extend(
                            (f"{label}.{changes_name}[{i}].evidence", change.get("evidence"))
                            for i, change in enumerate(changes) if isinstance(change, dict)
                        )
        world = payload.get("story_state")
        if isinstance(world, dict):
            references.append(("world_evidence", payload.get("world_evidence")))
            changes = world.get("thread_updates", [])
            if isinstance(changes, list):
                references.extend(
                    (f"story_state.thread_updates[{i}].evidence", change.get("evidence"))
                    for i, change in enumerate(changes) if isinstance(change, dict)
                )
        source = "".join(text.split())
        invalid = []
        paragraphs = [paragraph for paragraph in text.split("\n\n") if paragraph.strip()]
        for label, quote in references:
            if isinstance(quote, str) and quote.strip() and "".join(quote.split()) in source:
                continue
            detail = label + "=" + json.dumps(quote, ensure_ascii=False)
            matches = rank_texts(quote, paragraphs) if isinstance(quote, str) else []
            if matches and matches[0][0] > 0:
                # Retrieval aids a new model decision; it never substitutes a
                # near match for the invalid quotation automatically.
                detail += "\n可核对的实际成稿片段（需自行确认支持该事实）：" + paragraphs[matches[0][1]][:800]
            invalid.append(detail)
        if invalid:
            raise LLMResponseError("Memory evidence must quote the generated unit. Invalid quotes: " + "; ".join(invalid))

    def _candidate_store(self, update: UnitUpdate) -> StateStore:
        if not update.validation.passed:
            raise LLMResponseError("Cannot remember an unvalidated unit.")
        candidate = StateStore.from_dict(self.store.to_dict())
        candidate.update(
            update.characters,
            update.relationships,
            update.events,
            chapter=update.unit_number,
            source_unit=self._generated_source_unit(update.unit_number or 1),
        )
        if update.story_state:
            candidate.set_story_state(
                update.story_state,
                chapter=update.unit_number,
                source_unit=self._generated_source_unit(update.unit_number or 1),
            )
        return candidate

    def commit(self, update: UnitUpdate, unit_number: int) -> None:
        self.store = self._candidate_store(update)
        for event in update.events:
            record = deepcopy(event)
            if record.chapter <= 0:
                record.chapter = unit_number
            if record.source_unit is None:
                record.source_unit = self._generated_source_unit(unit_number)
            self.events.append(record)
        references = "\n".join(
            "  原文证据: " + quote for quote in dict.fromkeys(update.evidence)
        )
        self.summaries.append(
            f"- 单元{unit_number}: {update.summary}"
            + ("\n" + references if references else "")
        )
