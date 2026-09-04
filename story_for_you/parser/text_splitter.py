from __future__ import annotations

from dataclasses import dataclass

import re


_VOLUME_HEADING = re.compile(
    r"^第[ \t]*(?P<number>[0-9零〇一二三四五六七八九十百千]+)[ \t]*卷(?:[ \t]+.*)?$",
    flags=re.IGNORECASE,
)
_CHAPTER_WITHIN_VOLUME = re.compile(
    r"第[ \t]*(?P<number>[0-9零〇一二三四五六七八九十百千]+)[ \t]*[章节回](?:[ \t]+.*)?$",
    flags=re.IGNORECASE,
)
_TRAILER_PROMOTION = re.compile(
    r"(?im)^[ \t]*={8,}[ \t]*\r?$\r?\n"
    r"(?=[^\r\n]*(?:更多.*(?:小说|书籍)|下载|备用网址|www\.|https?://))[^\r\n]*"
    r"(?:\r?\n[^\r\n]*){0,3}?\r?\n^[ \t]*={8,}[ \t]*\r?$\s*\Z"
)


@dataclass
class TextChunk:
    content: str
    start_pos: int
    end_pos: int
    chapter: str | None = None


class TextSplitter:
    """Splits text into manageable chunks."""

    def __init__(
        self,
        chunk_size: int = 4000,
        overlap: int = 200,
        *,
        preserve_chapter_boundaries: bool = False,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.preserve_chapter_boundaries = preserve_chapter_boundaries

    def split(self, text: str) -> list[TextChunk]:
        """Split the text into chunks."""
        if not text:
            return []
        content_end = self._content_end(text)
        if self.preserve_chapter_boundaries:
            chapter_ranges = self._chapter_ranges(text, content_end=content_end)
            if chapter_ranges:
                return self._split_chapter_ranges(text, chapter_ranges)
        return self._split_range(text, 0, content_end, chapter=None, overlap=self.overlap)

    def _split_chapter_ranges(self, text: str, ranges: list[tuple[int, int, str]]) -> list[TextChunk]:
        """Split recognized chapters without carrying overlap into another chapter."""
        chunks: list[TextChunk] = []
        for start, end, chapter in ranges:
            # Repeating text across analysis units creates duplicate events and
            # unstable state updates.  Natural chapter boundaries are sufficient
            # context; very long chapters are split at local boundaries only.
            chunks.extend(self._split_range(text, start, end, chapter=chapter, overlap=0))
        return chunks

    def _split_range(
        self,
        text: str,
        start: int,
        end: int,
        *,
        chapter: str | None,
        overlap: int,
    ) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        cursor = start
        while cursor < end:
            upper = min(cursor + self.chunk_size, end)
            boundary = self._find_boundary(text, cursor, upper)
            raw_content = text[cursor:boundary]
            content = raw_content.strip()
            if not content:
                cursor = boundary
                continue
            leading_trim = len(raw_content) - len(raw_content.lstrip())
            trailing_trim = len(raw_content) - len(raw_content.rstrip())
            chunks.append(
                TextChunk(
                    content=content,
                    start_pos=cursor + leading_trim,
                    end_pos=boundary - trailing_trim,
                    chapter=chapter or self._detect_chapter(content),
                )
            )
            if boundary >= end:
                break
            next_cursor = boundary - overlap
            cursor = next_cursor if next_cursor > cursor else boundary
        return chunks

    def _chapter_ranges(
        self, text: str, *, content_end: int | None = None
    ) -> list[tuple[int, int, str]]:
        """Return source spans beginning at real chapter headings.

        Content before the first recognized chapter is treated as front matter.
        This deliberately avoids applying heuristic title-page stripping to books
        without chapter headings, where the opening text may be narrative.
        """
        end = len(text) if content_end is None else content_end
        headings: list[tuple[int, int, str]] = []
        for match in re.finditer(r"(?m)^[ \t]*(?P<label>[^\n\r]+?)[ \t]*\r?$", text[:end]):
            heading_kind = self._heading_kind(match.group("label"))
            if heading_kind is not None:
                headings.append((match.start(), match.end(), heading_kind))

        unit_starts: list[int] = []
        for index, (start, heading_end, kind) in enumerate(headings):
            if kind == "chapter":
                unit_start = start
                if index:
                    previous_start, previous_end, previous_kind = headings[index - 1]
                    # Preserve a bare volume title as context for its first
                    # chapter, rather than analyzing it alone or appending it
                    # to the preceding chapter.
                    if (
                        previous_kind == "volume"
                        and not text[previous_end:start].strip()
                    ):
                        unit_start = previous_start
                unit_starts.append(unit_start)
                continue
            next_start = headings[index + 1][0] if index + 1 < len(headings) else end
            # A bare volume label before its first chapter is hierarchy only.
            # A volume label followed by an actual prologue/foreword is a
            # legitimate analysis unit and must be retained.
            if text[heading_end:next_start].strip():
                unit_starts.append(start)
        return [
            (start, unit_starts[index + 1] if index + 1 < len(unit_starts) else end, str(index + 1))
            for index, start in enumerate(unit_starts)
        ]

    def _content_end(self, text: str) -> int:
        """Exclude a clearly delimited download-site promotion at EOF.

        The marker requires both separator lines and a promotion/URL signal, so
        ordinary story endings and author notes remain untouched.
        """
        match = _TRAILER_PROMOTION.search(text)
        return match.start() if match else len(text)

    def _heading_kind(self, label: str) -> str | None:
        """Classify a heading as a narrative chapter or a volume label."""
        stripped = label.strip()
        volume_match = _VOLUME_HEADING.match(stripped)
        if volume_match:
            remainder = stripped[volume_match.end("number") :]
            if _CHAPTER_WITHIN_VOLUME.search(remainder):
                return "chapter"
            return "volume"
        return "chapter" if self._chapter_number(stripped) is not None else None

    def _chapter_number(self, label: str) -> int | None:
        stripped = label.strip()
        patterns = (
            r"^第[ \t]*(?P<number>[0-9零〇一二三四五六七八九十百千]+)[ \t]*[章节回卷篇](?:[ \t]+.*)?$",
            r"^chapter[ \t]+(?P<number>\d+)(?:[ \t:：.-]+.*)?$",
            r"^(?P<number>[零〇一二三四五六七八九十百千]+)$",
        )
        for pattern in patterns:
            match = re.match(pattern, stripped, flags=re.IGNORECASE)
            if not match:
                continue
            return self._parse_chapter_number(match.group("number"))
        return None

    def _parse_chapter_number(self, value: str) -> int | None:
        if value.isdecimal():
            return int(value)
        digits = {"零": 0, "〇": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
        if any(char not in {*digits, "十", "百", "千"} for char in value):
            return None
        total = 0
        current = 0
        units = {"十": 10, "百": 100, "千": 1000}
        for char in value:
            if char in digits:
                current = digits[char]
                continue
            if char in units:
                total += (current or 1) * units[char]
                current = 0
        return total + current or None

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
        chapter = self._chapter_number(first_line)
        return str(chapter) if chapter is not None else None
