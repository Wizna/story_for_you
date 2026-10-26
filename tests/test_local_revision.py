import json

import pytest

from story_for_you.analysis.context import CharacterState, StoryContext
from story_for_you.core.ending.revision import TextEdit, apply_edits
from story_for_you.core.ending_writer import EndingWriter
from story_for_you.exceptions import GenerationError, LLMResponseError
from story_for_you.indexer.segment import SegmentIndex
from tests.test_review_fixes import _ResponseLLM


def test_revision_preserves_all_unaffected_prose_and_whitespace():
    text = "第一章\n\n甲持剑入城。\n\n保留的好段落。\n\n第二章\n\n乙空手离去。"
    result = apply_edits(
        text, [TextEdit("乙空手离去。", "乙携剑离去。", "前章甲已经交出宝剑")]
    )
    assert result == text.replace("乙空手离去。", "乙携剑离去。")


@pytest.mark.parametrize(
    "text,edits",
    [
        ("甲乙甲乙", [TextEdit("甲乙", "修改", "定位不唯一")]),
        ("aaa", [TextEdit("aa", "b", "重叠的重复引用")]),
        ("abcdef", [TextEdit("abc", "x", "原因"), TextEdit("bcde", "y", "原因")]),
        ("正文", [TextEdit("不存在", "修改", "引用失配")]),
    ],
)
def test_invalid_revision_is_rejected_without_partial_application(text, edits):
    with pytest.raises(LLMResponseError):
        apply_edits(text, edits)


def test_resolution_replaces_the_problem_scene_instead_of_appending_bridge():
    response = {
        "status": "needs_changes",
        "missing_threads": ["归还宝剑"],
        "notes": "应在离别场景归还",
        "edits": [
            {
                "old_text": "甲转身离开。",
                "new_text": "甲还剑后转身离开。",
                "reason": "偿还承诺",
            }
        ],
    }
    writer = EndingWriter(
        _ResponseLLM(json.dumps(response)), SegmentIndex([], {}, {}, {})
    )
    context = StoryContext(
        characters={"甲": CharacterState(name="甲", unresolved=["归还宝剑"])}
    )
    result = writer._phase_resolution_review(
        "甲转身离开。\n\n城门缓缓合上。", context, "归还宝剑的承诺", None, "{}"
    )
    assert result == "甲还剑后转身离开。\n\n城门缓缓合上。"


def test_revision_cannot_remove_or_reorder_planned_units():
    writer = EndingWriter(_ResponseLLM(""), SegmentIndex([], {}, {}, {}))
    with pytest.raises(GenerationError, match="removed or duplicated"):
        writer._validate_unit_headings(
            "第1章 起点\n\n正文", ["第1章 起点", "第2章 转折"]
        )
    with pytest.raises(GenerationError, match="unit order"):
        writer._validate_unit_headings(
            "第2章 转折\n\n第1章 起点", ["第1章 起点", "第2章 转折"]
        )
