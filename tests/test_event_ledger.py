from __future__ import annotations

from story_for_you.analysis.context import EventImpact, PlotEvent
from story_for_you.analysis.layers.event_ledger import EventLedger


def _event(
    event_id: str,
    *,
    participants: list[str] | None = None,
    summary: str = "翠翠在渡口遇见傩送。",
) -> PlotEvent:
    return PlotEvent(
        event_id=event_id,
        chapter=1,
        type="progress",
        participants=participants or ["翠翠", "傩送"],
        summary=summary,
        impact=EventImpact(),
    )


def test_event_ledger_discards_exact_duplicate_events():
    ledger = EventLedger()

    retained = ledger.record([_event("CH1-E01"), _event("CH1-E01")])

    assert [event.event_id for event in retained] == ["CH1-E01"]
    assert [event.event_id for event in ledger.timeline()] == ["CH1-E01"]


def test_event_ledger_keeps_same_summary_with_a_different_cast():
    ledger = EventLedger()

    retained = ledger.record(
        [
            _event("CH1-E01"),
            _event("CH2-E01", participants=["翠翠", "天保"]),
        ]
    )

    assert [event.event_id for event in retained] == ["CH1-E01", "CH2-E01"]


def test_repeated_transfer_after_return_is_a_new_occurrence():
    ledger = EventLedger()
    rows = [
        _event("E1", summary="甲把钥匙交给乙"),
        _event("E2", summary="乙把钥匙还给甲"),
        _event("E1", summary="甲把钥匙交给乙"),
    ]
    for chapter, event in zip((1, 2, 10), rows):
        event.chapter = chapter
        assert ledger.record([event]) == [event]
    restored = EventLedger.from_dict(ledger.to_dict())
    assert restored.timeline()[-1].summary == "甲把钥匙交给乙"
    assert len(restored.timeline()) == 3
    assert restored.record([rows[-1]]) == []


def test_split_chapter_keeps_occurrences_with_same_model_event_id():
    rows = [_event("CH1-E01"), _event("CH1-E01")]
    rows[0].source_unit = 1
    rows[1].source_unit = 2
    assert len(EventLedger().record(rows)) == 2
