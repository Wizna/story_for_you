from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from story_for_you.analysis.context import PlotEvent
from story_for_you.indexer.segment import Segment, SegmentIndex


def terms(text: str) -> set[str]:
    """Lexical retrieval only: words and Chinese n-grams, never inferred facts."""
    result = set(re.findall(r"[a-z0-9_]{2,}", text.casefold()))
    for run in re.findall(r"[\u3400-\u9fff]+", text):
        for size in (2, 3):
            result.update(run[i:i + size] for i in range(len(run) - size + 1))
    return result


def rank_texts(query: str, texts: list[str]) -> list[tuple[float, int]]:
    wanted = terms(query)
    features = [terms(text) & wanted for text in texts]
    frequency = Counter(term for row in features for term in row if term in wanted)
    weights = {term: math.log1p(len(texts) / count) for term, count in frequency.items()}
    scores = [
        (sum(weights[term] for term in row) / (1 + math.log1p(len(texts[index]))), index)
        for index, row in enumerate(features)
    ]
    return sorted(scores, key=lambda item: (item[0], item[1]), reverse=True)


@dataclass
class Passage:
    segment: Segment
    offset: int
    text: str


class EvidenceRetriever:
    def __init__(self, index: SegmentIndex) -> None:
        self.passages = [
            Passage(segment, start, segment.content[start:start + 1200])
            for segment in index.segments
            for start in range(0, len(segment.content), 1000)
        ]

    def retrieve(
        self, query: str, events: list[PlotEvent], focus: list[str],
        *, max_events: int = 24, max_passages: int = 8, max_chars: int = 12000,
    ) -> tuple[list[PlotEvent], list[Passage]]:
        scores = dict((index, score) for score, index in rank_texts(
            query, [event.summary + "\n" + event.evidence for event in events],
        ))
        ranked = sorted(range(len(events)), key=lambda i: (
            scores[i] + bool(set(events[i].participants).intersection(focus)), i,
        ), reverse=True)
        chosen = []
        remaining = max_chars
        for i in ranked:
            event = events[i]
            if not scores[i] and not set(event.participants).intersection(focus):
                continue
            size = len(event.summary) + len(event.evidence)
            if size > remaining or len(chosen) >= max_events:
                continue
            chosen.append(i)
            remaining -= size
        selected = self.retrieve_passages(
            query,
            max_passages=max_passages,
            max_chars=remaining,
        )
        # Story chronology remains explicit even though selection uses relevance.
        return [events[i] for i in sorted(chosen)], selected

    def retrieve_passages(
        self,
        query: str,
        *,
        max_passages: int = 8,
        max_chars: int = 12000,
        max_per_chapter: int = 2,
        exclude_segment_ids: set[int] | None = None,
    ) -> list[Passage]:
        """Retrieve diverse source passages for long-range continuity checks.

        Overlapping index windows can otherwise spend the whole evidence budget
        repeating one scene.  We cap each segment and source chapter, then
        restore source order so the model can see whether a motif recurs or
        changes over time.  This remains lexical retrieval; interpretation stays
        with the LLM and is explicitly labelled as historical evidence.
        """
        if max_passages <= 0 or max_chars <= 0 or not self.passages:
            return []
        excluded = exclude_segment_ids or set()
        ranked = rank_texts(query, [passage.text for passage in self.passages])
        chosen: list[tuple[float, int, Passage]] = []
        used_segments: set[int] = set()
        chapter_counts: Counter[int | None] = Counter()
        remaining = max_chars
        for score, index in ranked:
            if score <= 0 or len(chosen) >= max_passages:
                break
            passage = self.passages[index]
            segment_id = passage.segment.segment_id
            chapter = passage.segment.chapter
            # Imported indexes may not carry chapter labels.  Treat each
            # unlabelled segment as its own diversity bucket; otherwise all
            # early evidence would share ``chapter=None`` and the cap would
            # discard nearly the entire novel.
            chapter_key = chapter if chapter is not None else segment_id
            if segment_id in excluded or segment_id in used_segments:
                continue
            if chapter_counts[chapter_key] >= max_per_chapter:
                continue
            if len(passage.text) > remaining:
                continue
            normalized = "".join(passage.text.split())
            if any(normalized == "".join(item.text.split()) for _, _, item in chosen):
                continue
            chosen.append((score, index, passage))
            used_segments.add(segment_id)
            chapter_counts[chapter_key] += 1
            remaining -= len(passage.text)
        chosen.sort(key=lambda item: item[2].segment.segment_id)
        return [item[2] for item in chosen]
