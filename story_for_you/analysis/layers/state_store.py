from typing import Iterable

from story_for_you.analysis.context import (
    CharacterState,
    PlotEvent,
    Relationship,
    StoryState,
)


class StateStore:
    """Aggregates character and world state over time."""

    def __init__(self) -> None:
        self._characters: dict[str, CharacterState] = {}
        self._story_state: StoryState | None = None

    def update(
        self,
        characters: Iterable[CharacterState],
        relationships: Iterable[Relationship],
        events: Iterable[PlotEvent],
    ) -> None:
        """Placeholder for state merge logic."""
        for character in characters:
            self._characters[character.name] = character
        if self._story_state is None:
            self._story_state = StoryState(
                current_arc="unknown",
                world_tension="medium",
                major_conflicts=[],
                time_constraints=[],
                unresolved_events=[],
            )
        # TODO: merge relationships and events into the stored state.

    def characters_snapshot(self) -> dict[str, CharacterState]:
        """Return a shallow copy of known characters."""
        return dict(self._characters)

    def story_snapshot(self) -> StoryState | None:
        """Return the latest synthesized story state."""
        return self._story_state

    def clear(self) -> None:
        """Reset the state store."""
        self._characters.clear()
        self._story_state = None
