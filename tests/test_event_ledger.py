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

    retained = ledger.record([_event("CH1-E01"), _event("CH2-E01")])

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
