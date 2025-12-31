from story_for_you.analysis.context import ChapterSummary
from story_for_you.llm.base import LLMProvider


class ChapterSummarizer:
    """LLM-backed chapter summarizer placeholder."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def summarize(self, chapter_text: str, chapter_no: int) -> ChapterSummary:
        """Summarize the given chapter text."""
        raise NotImplementedError
