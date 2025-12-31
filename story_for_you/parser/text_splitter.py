from dataclasses import dataclass


@dataclass
class TextChunk:
    content: str
    start_pos: int
    end_pos: int
    chapter: str | None = None


class TextSplitter:
    """Splits text into manageable chunks."""

    def __init__(self, chunk_size: int = 4000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str) -> list[TextChunk]:
        """Split the text into chunks."""
        raise NotImplementedError

    def merge(self, chunks: list[str]) -> str:
        """Merge processed chunks back into a single string."""
        raise NotImplementedError
