import json

import pytest

from story_for_you.core.ending_writer import EndingOutline, EndingWriter
from story_for_you.indexer.segment import SegmentIndex
from story_for_you.exceptions import LLMResponseError
from tests.test_cacheable_prompting import _RecordingLLM


def _polish(llm, mode, draft):
    return EndingWriter(llm, SegmentIndex([], {}, {}, {}), revision_mode=mode)._phase_polish(
        draft, None, EndingOutline(), "原作上下文", "只续写一场", "", "继续当前场景",
    )

def test_draft_only_skips_polish_call():
    draft = "“不。”\n\n她仍然不肯走。\n\n“不。”"
    llm = _RecordingLLM([])
    assert _polish(llm, "none", draft) == draft
    assert llm.prompts == []


def test_local_polish_changes_only_the_quoted_passage():
    draft = "“不。”\n\n她仍然不肯走。\n\n“不。”"
    llm = _RecordingLLM([json.dumps({"edits": [{
        "old_text": "她仍然不肯走。", "new_text": "她站着没动。", "reason": "减少解释",
    }]})])
    assert _polish(llm, "local", draft) == "“不。”\n\n她站着没动。\n\n“不。”"
    assert len(llm.prompts) == 1


def test_local_polish_can_keep_a_draft_without_changes():
    draft = "她站着没动。"
    assert _polish(_RecordingLLM(['{"edits": []}']), "local", draft) == draft


def test_local_polish_can_explicitly_keep_a_verified_quote():
    draft = "她站着没动。"
    payload = json.dumps({"edits": [{"old_text": draft, "new_text": draft, "reason": "保持原样"}]})
    assert _polish(_RecordingLLM([payload]), "local", draft) == draft
    with pytest.raises(LLMResponseError, match="exactly one"):
        _polish(_RecordingLLM([payload]), "local", "不同的正文")
