from story_for_you.analysis.context import PlotEvent
from story_for_you.llm.base import LLMProvider


class EventExtractor:
    """LLM-backed event extractor placeholder."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def extract(self, chapter_text: str, participants: list[str]) -> list[PlotEvent]:
        """Extract lasting events from the chapter text."""
        raise NotImplementedError
