from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field

from story_for_you.analysis.context import (
    CharacterState,
    PlotEvent,
    Relationship,
    StoryContext,
    StoryState,
)
from story_for_you.analysis.extractors.characters import CharacterExtractor
from story_for_you.analysis.layers.state_store import StateStore
from story_for_you.core.ending.validator import (
    ContinuationValidationResult,
    ContinuationValidator,
)
from story_for_you.core.exceptions import LLMResponseError
from story_for_you.core.prompting import load_template
from story_for_you.llm.base import LLMProvider
from story_for_you.llm.telemetry import telemetry_options
from story_for_you.utils.json_utils import load_json_response
from story_for_you.utils.prompting import build_cacheable_prompt


@dataclass
class UnitUpdate:
    validation: ContinuationValidationResult
    summary: str = ""
    characters: list[CharacterState] = field(default_factory=list)
    events: list[PlotEvent] = field(default_factory=list)
    story_state: StoryState | None = None
    relationships: list[Relationship] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)


class ContinuationMemory:
    """Keep generated observations separate from the immutable source analysis."""

    def __init__(self, source: StoryContext):
        self.source = deepcopy(source)
        self.store = StateStore()
        self.store.update(source.characters.values(), [], source.events)
        if source.story_state:
            self.store.set_story_state(deepcopy(source.story_state))
        self.summaries: list[str] = []
        self.events: list[PlotEvent] = []

    def snapshot(self) -> StoryContext:
        context = deepcopy(self.source)
        context.characters = self.store.characters_snapshot()
        context.story_state = self.store.story_snapshot()
        context.events.extend(deepcopy(self.events))
        return context

    def render(self) -> str:
        return "\n".join(
            [
                "## 已通过逐单元审查的续写记忆（保持全部历史摘要，不是未来计划）",
                *self.summaries,
                *[f"- [{event.event_id}] {event.summary}" for event in self.events],
            ]
        )

    def review(
        self, llm: LLMProvider, text: str, context_block: str, unit_number: int
    ) -> UnitUpdate:
        world = self.store.story_snapshot()
        prompt = build_cacheable_prompt(
            context_block,
            load_template("ending_memory"),
            prefix_placeholder="context_block",
            unit_text=text,
            unit_number=str(unit_number),
            known_threads=json.dumps(
                {
                    "characters": {
                        name: [asdict(thread) for thread in char.threads]
                        for name, char in self.store.characters_snapshot().items()
                        if char.threads
                    },
                    "world": [asdict(thread) for thread in world.threads]
                    if world
                    else [],
                },
                ensure_ascii=False,
            ),
        )
        response = llm.generate(
            prompt,
            options=telemetry_options(
                {"no_think": True, "temperature": 0.1},
                phase="continue",
                step=": review and remember unit",
            ),
        )
        payload = load_json_response(response.content)
        if not isinstance(payload, dict):
            raise LLMResponseError("Unit memory review must return an object.")
        validation = ContinuationValidator(llm)._from_payload(payload)
        update = UnitUpdate(validation)
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
        source_text = "".join(text.split())
        for item in payload["characters"]:
            if not isinstance(item, dict):
                raise LLMResponseError("Unit memory character must be an object.")
            evidence = item.get("evidence")
            if (
                not isinstance(evidence, str)
                or not evidence.strip()
                or "".join(evidence.split()) not in source_text
            ):
                raise LLMResponseError(
                    "Character state changes require evidence from the generated unit."
                )
            char = extractor._to_character(item)
            for change in char.thread_updates:
                if "".join(change.evidence.split()) not in source_text:
                    raise LLMResponseError(
                        "Thread change evidence must quote the generated unit."
                    )
            characters.append(char)
            update.evidence.append(evidence.strip())
        events = []
        for index, item in enumerate(payload["events"], start=1):
            if not isinstance(item, dict):
                raise LLMResponseError("Unit memory event must be an object.")
            evidence = item.get("evidence")
            if (
                not isinstance(evidence, str)
                or not evidence.strip()
                or "".join(evidence.split()) not in source_text
            ):
                raise LLMResponseError(
                    "New facts require evidence from the generated unit."
                )
            event = PlotEvent.from_dict(item)
            event.event_id = f"GEN-U{unit_number}-E{index}"
            if set(event.deceased_characters) - set(event.participants):
                raise LLMResponseError(
                    "Deceased characters must be event participants."
                )
            events.append(event)
            update.evidence.append(evidence.strip())
        world = payload.get("story_state")
        relationships = payload.get("relationships", [])
        if not isinstance(relationships, list):
            raise LLMResponseError("Unit memory relationships must be a list.")
        for item in relationships:
            relation = Relationship.from_dict(item)
            if (
                not relation.evidence.strip()
                or "".join(relation.evidence.split()) not in source_text
            ):
                raise LLMResponseError(
                    "Relationship change evidence must quote the generated unit."
                )
            update.relationships.append(relation)
            update.evidence.append(relation.evidence.strip())
        update.summary = summary.strip()
        update.characters = characters
        update.events = events
        update.story_state = StoryState.from_dict(world) if world is not None else None
        if update.story_state:
            quote = payload.get("world_evidence")
            if (
                not isinstance(quote, str)
                or not quote.strip()
                or "".join(quote.split()) not in source_text
            ):
                raise LLMResponseError(
                    "World state changes require evidence from the generated unit."
                )
            for change in update.story_state.thread_updates:
                if "".join(change.evidence.split()) not in source_text:
                    raise LLMResponseError(
                        "World thread changes must quote the generated unit."
                    )
            update.evidence.append(quote.strip())
        return update

    def commit(self, update: UnitUpdate, unit_number: int) -> None:
        if not update.validation.passed:
            raise LLMResponseError("Cannot remember an unvalidated unit.")
        candidate = StateStore.from_dict(self.store.to_dict())
        candidate.update(update.characters, update.relationships, update.events)
        if update.story_state:
            candidate.set_story_state(update.story_state)
        self.store = candidate
        self.events.extend(update.events)
        references = "\n".join(
            "  原文证据: " + quote for quote in dict.fromkeys(update.evidence)
        )
        self.summaries.append(
            f"- 单元{unit_number}: {update.summary}"
            + ("\n" + references if references else "")
        )
