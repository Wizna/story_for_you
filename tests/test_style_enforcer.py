"""Tests for mechanical StyleEnforcer post-processing."""

from __future__ import annotations

from story_for_you.core.ending.style_enforcer import StyleEnforcer


def test_exact_duplicate_removed():
    enforcer = StyleEnforcer()
    text = "翠翠坐在船头，望着对岸的山。\n\n翠翠坐在船头，望着对岸的山。"

    result = enforcer.post_process(text)

    assert result.count("翠翠坐在船头") == 1


def test_whitespace_normalized_duplicate_removed():
    enforcer = StyleEnforcer()
    text = "翠翠坐在 船头 望着对岸。\n\n翠翠坐在  船头  望着对岸。"

    result = enforcer.post_process(text)

    assert result.count("翠翠坐在") == 1


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
