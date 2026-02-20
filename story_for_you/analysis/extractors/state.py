from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Any

from story_for_you.analysis.context import PlotEvent, StoryState
from story_for_you.analysis.prompting import fill_template, load_template
from story_for_you.llm.base import LLMProvider
from story_for_you.utils.json_utils import load_json_response

logger = logging.getLogger(__name__)


class StateSynthesizer:
    """Produces an updated StoryState from plot events."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm
        self.template = load_template("state_update")

    def update(
        self,
        story_state: StoryState | None,
        events: list[PlotEvent],
        recent_context: str,
    ) -> StoryState:
        """Synthesize the long-term story state via the LLM."""
        prior_state_payload: Any = asdict(story_state) if story_state else None
        events_payload = [asdict(event) for event in events]
        prompt = fill_template(
            self.template,
            prior_state=(
                "null" if prior_state_payload is None else json.dumps(prior_state_payload, ensure_ascii=False)
            ),
            events=json.dumps(events_payload, ensure_ascii=False),
            recent_context=recent_context.strip() or "暂无历史上下文。",
        )
        response = self.llm.generate(prompt=prompt, options={"no_think": True})
        try:
            data = load_json_response(response.content)
            if not isinstance(data, dict):
                raise ValueError("Story state response is not a JSON object.")
            state = StoryState(
                current_arc=str(data.get("current_arc") or (story_state.current_arc if story_state else "setup")),
                world_tension=str(data.get("world_tension") or (story_state.world_tension if story_state else "low")),
                major_conflicts=[str(item) for item in data.get("major_conflicts", [])][:5],
                time_constraints=[str(item) for item in data.get("time_constraints", [])][:3],
                unresolved_events=[str(item) for item in data.get("unresolved_events", [])][:5],
            )
            return self._stabilize_arc(state, events)
        except (TypeError, ValueError) as exc:
            logger.warning("Failed to synthesize story state via LLM: %s", exc)
            state = self._fallback_update(story_state, events)
            return self._stabilize_arc(state, events)

    # Fallback heuristics ----------------------------------------------------------
    def _fallback_update(self, story_state: StoryState | None, events: list[PlotEvent]) -> StoryState:
        if story_state is None:
            story_state = StoryState(
                current_arc="setup",
                world_tension="low",
                major_conflicts=[],
                time_constraints=[],
                unresolved_events=[],
            )
        for event in events:
            if event.type in {"conflict", "setback"}:
                story_state.world_tension = "high"
                story_state.major_conflicts.append(event.summary)
            elif event.type == "reveal":
                if story_state.world_tension == "low":
                    story_state.world_tension = "medium"
                story_state.major_conflicts.append(f"Revelation: {event.summary}")
            if event.is_irreversible:
                story_state.unresolved_events.append(event.summary)
        story_state.major_conflicts = story_state.major_conflicts[-5:]
        story_state.unresolved_events = list(dict.fromkeys(story_state.unresolved_events))[-5:]
        return story_state

    def _stabilize_arc(self, story_state: StoryState, events: list[PlotEvent]) -> StoryState:
        if not events:
            return story_state
        irreversible_seen = any(event.is_irreversible for event in events)
        conflict_seen = any(event.type in {"conflict", "setback"} for event in events)
        reveal_seen = any(event.type == "reveal" for event in events)

        if story_state.current_arc == "setup":
            if conflict_seen:
                story_state.current_arc = "journey"
            if irreversible_seen:
                story_state.current_arc = "dark-night" if conflict_seen else "twist"
            elif reveal_seen:
                story_state.current_arc = "twist"
        elif story_state.current_arc in {"journey", "twist"} and irreversible_seen:
            story_state.current_arc = "dark-night"
        return story_state
