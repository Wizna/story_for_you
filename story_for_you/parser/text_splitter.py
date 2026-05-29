from __future__ import annotations

from dataclasses import dataclass

import re


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
        if not text:
            return []
        chunks: list[TextChunk] = []
        cursor = 0
        length = len(text)
        while cursor < length:
            upper = min(cursor + self.chunk_size, length)
            boundary = self._find_boundary(text, cursor, upper)
            content = text[cursor:boundary].strip()
            if not content:
                cursor = boundary
                continue
            chapter = self._detect_chapter(content)
            chunks.append(
                TextChunk(
                    content=content,
                    start_pos=cursor,
                    end_pos=boundary,
                    chapter=chapter,
                )
            )
            if boundary >= length:
                break
            next_cursor = boundary - self.overlap
            cursor = next_cursor if next_cursor > cursor else boundary
        return chunks

    def merge(self, chunks: list[str]) -> str:
        """Merge processed chunks back into a single string."""
        filtered = [chunk.strip() for chunk in chunks if chunk.strip()]
        return "\n\n".join(filtered)

    def _find_boundary(self, text: str, start: int, tentative_end: int) -> int:
        """Prefer splitting on blank lines or sentence boundaries."""
        if tentative_end >= len(text):
            return len(text)
        window = text[start:tentative_end]
        double_newline = window.rfind("\n\n")
        if double_newline != -1 and start + double_newline > start:
            return start + double_newline
        sentence_match = max(window.rfind("。"), window.rfind("."), window.rfind("!"), window.rfind("?"))
        if sentence_match != -1 and start + sentence_match > start:
            return start + sentence_match + 1
        return tentative_end

    def _detect_chapter(self, content: str) -> str | None:
        """Attempt to detect a chapter label from the chunk."""
        first_line = content.splitlines()[0].strip()
        patterns = [
            r"^chapter\s+\d+",
            r"^第.+章",
        ]
        for pattern in patterns:
            if re.match(pattern, first_line, flags=re.IGNORECASE):
                return first_line
        return None
