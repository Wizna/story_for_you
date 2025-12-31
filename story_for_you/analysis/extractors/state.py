from __future__ import annotations

from story_for_you.analysis.context import PlotEvent, StoryState
from story_for_you.llm.base import LLMProvider


class StateSynthesizer:
    """Produces an updated StoryState from plot events."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def update(self, story_state: StoryState | None, events: list[PlotEvent]) -> StoryState:
        """Synthesize the long-term story state."""
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
        story_state.current_arc = self._infer_arc(story_state, events)
        return story_state

    def _infer_arc(self, story_state: StoryState, events: list[PlotEvent]) -> str:
        if not events:
            return story_state.current_arc
        latest = events[-1].type
        mapping = {
            "conflict": "climax",
            "reveal": "twist",
            "progress": "journey",
            "setback": "dark-night",
        }
        return mapping.get(latest, story_state.current_arc)
