"""Tests for StyleEnforcer post-processing and meta-content filtering."""

from __future__ import annotations

from story_for_you.core.ending.style_enforcer import StyleEnforcer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_enforcer(register: str = "mixed") -> StyleEnforcer:
    """Build a StyleEnforcer with a minimal mock style."""

    class _FakeStyle:
        register = "mixed"
        characteristic_words: list[str] = []
        tone_markers: list[str] = []
        metaphor_style = ""

    style = _FakeStyle()
    style.register = register
    return StyleEnforcer(style)


# ---------------------------------------------------------------------------
# Meta-content stripping
# ---------------------------------------------------------------------------

class TestStripMetaContent:
    """Verify that _strip_meta_content removes editorial annotations."""

    def test_removes_editorial_hint(self):
        enforcer = _make_enforcer()
        text = "翠翠坐在船头。\n\n【编辑提示】通过虎耳草与白塔意象呼应身世伏笔。"
        result = enforcer._strip_meta_content(text)
        assert "【编辑提示】" not in result
        assert "翠翠坐在船头。" in result

    def test_removes_reader_custom_marker(self):
        enforcer = _make_enforcer()
        text = "河面漂过月光碎片。\n\n（读者定制版本）"
        result = enforcer._strip_meta_content(text)
        assert "（读者定制版本）" not in result
        assert "河面漂过月光碎片。" in result

    def test_removes_multiple_meta_patterns(self):
        enforcer = _make_enforcer()
        text = (
            "翠翠在白塔下拾起一撮虎耳草。\n\n"
            "【编辑提示】通过虎耳草意象呼应伏笔\n\n"
            "（读者定制版本）"
        )
        result = enforcer._strip_meta_content(text)
        assert "【编辑提示】" not in result
        assert "（读者定制版本）" not in result
        assert "翠翠在白塔下拾起一撮虎耳草。" in result

    def test_preserves_bracket_in_dialogue(self):
        """Full-width brackets in dialogue should not be stripped."""
        enforcer = _make_enforcer()
        text = '她说:\u201c这里的水很清。\u201d'
        result = enforcer._strip_meta_content(text)
        assert result == text

    def test_removes_half_width_variant(self):
        enforcer = _make_enforcer()
        text = "结局。\n\n(读者定制版本)"
        result = enforcer._strip_meta_content(text)
        assert "(读者定制版本)" not in result

    def test_removes_note_prefix(self):
        enforcer = _make_enforcer()
        text = "故事正文。\n\n注：此处为续写片段"
        result = enforcer._strip_meta_content(text)
        assert "注：" not in result

    def test_collapses_excess_blank_lines(self):
        enforcer = _make_enforcer()
        text = "段落一。\n\n\n\n\n段落二。"
        result = enforcer._strip_meta_content(text)
        assert "\n\n\n" not in result
        assert "段落一。" in result
        assert "段落二。" in result


# ---------------------------------------------------------------------------
# Full post_process pipeline
# ---------------------------------------------------------------------------

class TestPostProcess:
    """Verify end-to-end post_process behavior."""

    def test_strips_meta_and_deduplicates(self):
        enforcer = _make_enforcer()
        text = (
            "翠翠坐在船头，望着对岸的山。\n\n"
            "翠翠坐在船头，望着对岸的山。\n\n"
            "【编辑提示】通过虎耳草意象呼应。\n\n"
            "（读者定制版本）"
        )
        result = enforcer.post_process(text)
        assert "【编辑提示】" not in result
        assert "（读者定制版本）" not in result
        # Only one copy of the duplicated paragraph
        assert result.count("翠翠坐在船头") == 1

    def test_empty_text_returns_empty(self):
        enforcer = _make_enforcer()
        assert enforcer.post_process("") == ""

    def test_pure_meta_returns_original(self):
        """If stripping meta leaves nothing, return original text."""
        enforcer = _make_enforcer()
        text = "（读者定制版本）"
        result = enforcer.post_process(text)
        # The meta is stripped, leaving empty → fallback returns original
        assert result == text

    def test_preserves_clean_content(self):
        enforcer = _make_enforcer()
        text = "翠翠在河边等待。\n\n傩送从远处走来。"
        result = enforcer.post_process(text)
        assert "翠翠在河边等待。" in result
        assert "傩送从远处走来。" in result


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class TestDeduplication:
    """Verify paragraph deduplication catches similar content."""

    def test_exact_duplicate_removed(self):
        enforcer = _make_enforcer()
        text = "翠翠坐在船头，望着对岸的山。\n\n翠翠坐在船头，望着对岸的山。"
        result = enforcer.post_process(text)
        assert result.count("翠翠坐在船头") == 1

    def test_whitespace_normalized_duplicate_removed(self):
        enforcer = _make_enforcer()
        # Paragraphs identical after whitespace normalization
        text = "翠翠坐在 船头 望着对岸。\n\n翠翠坐在  船头  望着对岸。"
        result = enforcer.post_process(text)
        assert result.count("翠翠坐在") == 1


# ---------------------------------------------------------------------------
# Bridge filtering
# ---------------------------------------------------------------------------

class TestFilterDuplicateBridges:
    """Verify bridge deduplication against existing polished content."""

    def test_duplicate_bridge_filtered(self):
        enforcer = _make_enforcer()
        polished = "翠翠坐在船头。远处传来歌声，她将虎耳草系在芦管上。"
        bridges = ["翠翠坐在船头。远处传来歌声，她将虎耳草系在芦管上。"]
        result = enforcer.filter_duplicate_bridges(polished, bridges)
        assert len(result) == 0

    def test_unique_bridge_kept(self):
        enforcer = _make_enforcer()
        polished = "翠翠坐在船头，望着对岸的山。"
        bridges = ["顺顺立在渡口，看着远方的白鹭掠过水面。"]
        result = enforcer.filter_duplicate_bridges(polished, bridges)
        assert len(result) == 1
