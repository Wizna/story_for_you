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
        selected = []
        for score, i in rank_texts(query, [passage.text for passage in self.passages]):
            passage = self.passages[i]
            if score <= 0 or len(selected) >= max_passages:
                break
            if len(passage.text) <= remaining:
                selected.append(passage)
                remaining -= len(passage.text)
        # Story chronology remains explicit even though selection uses relevance.
        return [events[i] for i in sorted(chosen)], selected
