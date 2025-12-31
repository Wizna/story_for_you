from story_for_you.analysis.context import Relationship
from story_for_you.llm.base import LLMProvider


class RelationshipMapper:
    """Maps relationship deltas between characters."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def map(self, chapter_text: str, characters: list[str] | None = None) -> list[Relationship]:
        """Return relationship changes observed in the text."""
        raise NotImplementedError
