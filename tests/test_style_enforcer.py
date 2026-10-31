"""Tests for mechanical StyleEnforcer post-processing."""

from __future__ import annotations

from story_for_you.core.ending.style_enforcer import StyleEnforcer


def test_exact_repetition_is_left_for_contextual_review():
    enforcer = StyleEnforcer()
    text = "翠翠坐在船头，望着对岸的山。\n\n翠翠坐在船头，望着对岸的山。"

    result = enforcer.post_process(text)

    assert result == text


def test_internal_whitespace_and_repetition_are_preserved():
    enforcer = StyleEnforcer()
    text = "翠翠坐在 船头 望着对岸。\n\n翠翠坐在  船头  望着对岸。"

    result = enforcer.post_process(text)

    assert result == text


def test_repeated_answers_in_different_scenes_are_preserved():
    text = '第1章 夜审\n\n“你认罪吗？”\n\n“不。”\n\n第2章 再审\n\n“你在场吗？”\n\n“不。”'
    assert StyleEnforcer().post_process(text).count('“不。”') == 2


def test_duplicate_bridge_filtered():
    enforcer = StyleEnforcer()
    polished = "翠翠坐在船头。远处传来歌声，她将虎耳草系在芦管上。"
    bridges = ["翠翠坐在船头。远处传来歌声，她将虎耳草系在芦管上。"]

    result = enforcer.filter_duplicate_bridges(polished, bridges)

    assert len(result) == 0


def test_shared_opening_with_new_plot_is_preserved():
    enforcer = StyleEnforcer()
    text = (
        "林凡再度运转周身的灵力。石门依旧纹丝不动。\n\n"
        "林凡再度运转周身的灵力。这一次石门轰然洞开，他终于逃出了地牢。"
    )

    result = enforcer.post_process(text)

    assert "终于逃出了地牢" in result
