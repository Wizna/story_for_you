from dataclasses import asdict

import pytest

from story_for_you.analysis.context import CharacterState, PlotEvent, StoryState, EventImpact
from story_for_you.analysis.layers.state_store import StateStore
from story_for_you.analysis.threads import ThreadUpdate
from story_for_you.exceptions import LLMResponseError


def test_thread_is_resolved_only_with_explicit_identified_update():
    store = StateStore()
    store.update(
        [CharacterState(name="甲", unresolved=["寻找妹妹", "归还宝剑"])], [], []
    )
    original = store.characters_snapshot()["甲"]
    key = original.threads[0].thread_id
    store.update(
        [
            CharacterState(
                name="甲",
                thread_updates=[
                    ThreadUpdate("update", key, "已得知妹妹在城中", "妹妹在城中。"),
                ],
            )
        ],
        [],
        [],
    )
    assert store.characters_snapshot()["甲"].unresolved == [
        "已得知妹妹在城中",
        "归还宝剑",
    ]
    store.update(
        [
            CharacterState(
                name="甲",
                thread_updates=[
                    ThreadUpdate("resolve", key, "兄妹已重逢", "兄妹终于重逢。"),
                ],
            )
        ],
        [],
        [],
    )
    snapshot = store.characters_snapshot()["甲"]
    assert snapshot.unresolved == ["归还宝剑"]
    assert snapshot.threads[0].status == "resolved"
    assert snapshot.threads[0].thread_id == key
    restored = StateStore.from_dict(store.to_dict()).characters_snapshot()["甲"]
    assert asdict(restored) == asdict(snapshot)


def test_thread_update_rejects_missing_evidence_and_unknown_id():
    with pytest.raises(LLMResponseError, match="evidence"):
        ThreadUpdate.from_dict(
            {
                "operation": "resolve",
                "thread_id": "T1",
                "summary": "找到妹妹",
                "evidence": "",
            }
        )
    store = StateStore()
    store.update([CharacterState(name="甲")], [], [])
    with pytest.raises(LLMResponseError, match="Unknown thread ID"):
        store.update(
            [
                CharacterState(
                    name="甲",
                    thread_updates=[
                        ThreadUpdate(
                            "resolve", "missing", "找到妹妹", "妹妹推门而入。"
                        ),
                    ],
                )
            ],
            [],
            [],
        )


def test_old_state_snapshot_migrates_thread_ids_without_losing_promises():
    character = asdict(CharacterState(name="甲", unresolved=["寻找妹妹"]))
    character.pop("threads")
    character.pop("thread_updates")
    restored = StateStore.from_dict(
        {"characters": {"甲": character}, "story_state": None, "event_log": []}
    )
    snapshot = restored.characters_snapshot()["甲"]
    assert snapshot.unresolved == ["寻找妹妹"]
    assert snapshot.threads[0].thread_id.startswith("T-")


def test_world_threads_survive_unmentioned_chapters_and_resolve_by_id():
    store = StateStore()
    store.set_story_state(StoryState("journey", "medium", [], [], ["城门密信的去向"]))
    first = store.story_snapshot()
    key = first.threads[0].thread_id
    store.set_story_state(StoryState("journey", "medium", [], [], []))
    assert store.story_snapshot().unresolved_events == ["城门密信的去向"]
    store.set_story_state(
        StoryState(
            "journey",
            "medium",
            [],
            [],
            [],
            thread_updates=[
                ThreadUpdate("resolve", key, "密信已送达", "密信交到了城主手中。"),
            ],
        )
    )
    assert store.story_snapshot().unresolved_events == []
    assert store.story_snapshot().threads[0].status == "resolved"


def test_explicit_resolution_and_new_legacy_observation_can_coexist():
    store = StateStore()
    store.update([CharacterState(name="甲", unresolved=["寻找妹妹"])], [], [])
    key = store.characters_snapshot()["甲"].threads[0].thread_id
    store.update([CharacterState(
        name="甲", unresolved=["寻找妹妹", "归还宝剑"],
        thread_updates=[ThreadUpdate("resolve", key, "兄妹重逢", "妹妹推门而入。")],
    )], [], [])
    assert store.characters_snapshot()["甲"].unresolved == ["归还宝剑"]


def test_thread_history_keeps_chapter_evidence_and_does_not_reopen_resolution():
    store = StateStore()
    store.update(
        [CharacterState(name="甲", thread_updates=[
            ThreadUpdate("add", "", "密信下落不明", "他把密信藏在袖中。", kind="mystery"),
        ])],
        [],
        [],
        chapter=4,
        source_unit=9,
    )
    thread = store.characters_snapshot()["甲"].threads[0]
    key = thread.thread_id
    assert thread.history[0].chapter == 4
    assert thread.history[0].source_unit == 9

    store.update(
        [CharacterState(name="甲", thread_updates=[
            ThreadUpdate("resolve", key, "密信已经送达", "他亲手把信交给城主。"),
        ])],
        [],
        [],
        chapter=5,
        source_unit=10,
    )
    store.update(
        [CharacterState(name="甲", thread_updates=[
            ThreadUpdate("update", key, "城主是否回信仍未知", "城主只说稍后答复。"),
        ])],
        [],
        [],
        chapter=6,
        source_unit=11,
    )
    thread = store.characters_snapshot()["甲"].threads[0]
    assert thread.status == "resolved"
    assert [item.chapter for item in thread.history] == [4, 5, 6]


def test_initial_thread_record_evidence_enters_history_for_long_range_recall():
    from story_for_you.analysis.threads import StoryThread

    store = StateStore()
    store.update(
        [CharacterState(
            name="甲",
            threads=[StoryThread("T-old", "旧匣下落", evidence="他把旧匣藏入佛龛。")],
        )],
        [],
        [],
        chapter=2,
        source_unit=8,
    )

    thread = store.characters_snapshot()["甲"].threads[0]
    assert len(thread.history) == 1
    assert thread.history[0].evidence == "他把旧匣藏入佛龛。"
    assert thread.history[0].source_unit == 8


def test_non_irreversible_death_mention_does_not_mark_character_dead():
    store = StateStore()
    store.update([CharacterState(name="甲", status="alive")], [], [])
    store.update(
        [],
        [],
        [PlotEvent(
            "rumour", 3, "reveal", ["甲"], "传言甲已死", EventImpact(),
            is_irreversible=False, deceased_characters=["甲"], evidence="有人说甲已经死了。",
        )],
    )

    assert store.characters_snapshot()["甲"].status == "alive"
