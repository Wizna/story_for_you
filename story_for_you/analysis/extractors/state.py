from story_for_you.analysis.context import PlotEvent, StoryState
from story_for_you.llm.base import LLMProvider


class StateSynthesizer:
    """Produces an updated StoryState from plot events."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def update(self, story_state: StoryState | None, events: list[PlotEvent]) -> StoryState:
        """Synthesize the long-term story state."""
        raise NotImplementedError
