import json

import pytest
from typer.testing import CliRunner

from story_for_you.analysis.context import CharacterState, StoryContext
from story_for_you.core.ending_writer import EndingWriter
from story_for_you.exceptions import LLMResponseError, GenerationError
from story_for_you.indexer.segment import SegmentIndex
from story_for_you.llm.base import LLMResponse
from tests.test_continuation_pipeline import PipelineLLM


def _source():
    return StoryContext(characters={"甲": CharacterState("甲", role="main", location="城门")})


def _writer(llm):
    return EndingWriter(llm, SegmentIndex([], {}, {}, {}))


def test_resume_replays_completed_steps_without_new_model_calls(tmp_path):
    class Interrupted(PipelineLLM):
        failed = False

        def generate(self, prompt, system="", options=None):
            if options["_sfy_step"] == ": draft" and self.unit == 1 and not self.failed:
                self.failed = True
                raise ConnectionError("network interrupted")
            return super().generate(prompt, system, options)

    llm = Interrupted()
    path = tmp_path / "run.json"
    with pytest.raises(ConnectionError):
        _writer(llm).continue_story("甲在城门。", _source(), checkpoint_path=path)
    assert len(json.loads(path.read_text())["calls"]) == 7
    writer = _writer(llm)
    result = writer.continue_story("甲在城门。", _source(), checkpoint_path=path, resume=True)
    assert "第3章" in result
    assert len(llm.draft_prompts) == 3
    assert writer.last_context.characters["甲"].location == "客栈"
    assert json.loads(path.read_text())["completed"]
    assert _writer(llm).continue_story("甲在城门。", _source(), checkpoint_path=path, resume=True) == result
    assert len(llm.draft_prompts) == 3
    with pytest.raises(LLMResponseError, match="does not match"):
        _writer(llm).continue_story("不同原文", _source(), checkpoint_path=path, resume=True)


def test_invalid_repair_response_is_not_reused_after_resume(tmp_path):
    class BadReview(PipelineLLM):
        failures = 0

        def generate(self, prompt, system="", options=None):
            if options["_sfy_step"] == ": review and remember unit" and self.failures < 2:
                self.failures += 1
                return LLMResponse(json.dumps({"passed": True, "issues": [], "repair_instructions": []}), 0)
            return super().generate(prompt, system, options)

    llm = BadReview()
    path = tmp_path / "run.json"
    with pytest.raises(LLMResponseError, match="summary"):
        _writer(llm).continue_story("甲在城门。", _source(), checkpoint_path=path)
    assert len(json.loads(path.read_text())["calls"]) == 6
    _writer(llm).continue_story("甲在城门。", _source(), checkpoint_path=path, resume=True)
    assert len(llm.draft_prompts) == 3


def test_final_repair_rebuilds_exported_state_from_final_prose():
    class FinalRepair(PipelineLLM):
        validated = False

        def generate(self, prompt, system="", options=None):
            step = options["_sfy_step"]
            if step == ": validate continuation":
                first = not self.validated
                self.validated = True
                return LLMResponse(json.dumps({"passed": not first, "issues": ["末尾应在桥头"] if first else [],
                                               "repair_instructions": ["修复位置"] if first else []}), 0)
            if step == ": final repair":
                old = self.texts[0]
                self.texts[0] = old.replace("客栈", "桥头")
                return LLMResponse(json.dumps({"edits": [{"old_text": old, "new_text": self.texts[0], "reason": "修复位置"}]}), 0)
            response = super().generate(prompt, system, options)
            if step == ": continuation plan":
                payload = json.loads(response.content)
                payload["chapters"] = payload["chapters"][:1]
                payload["unit_count"] = 1
                return LLMResponse(json.dumps(payload), 0)
            if step == ": review and remember unit" and "桥头" in self.texts[0]:
                payload = json.loads(response.content)
                payload["characters"][0]["location"] = "桥头"
                payload["summary"] = "甲把剑交给乙，前往桥头。"
                return LLMResponse(json.dumps(payload), 0)
            return response

    writer = _writer(FinalRepair())
    result = writer.continue_story("甲在城门。", _source())
    assert "桥头" in result
    assert writer.last_context.characters["甲"].location == "桥头"


def test_cli_exports_matching_state_and_combined_input(tmp_path, monkeypatch):
    from story_for_you.cli import main
    from story_for_you.config.settings import Settings
    from story_for_you.utils.file_io import read_text_file

    source = tmp_path / "source.txt"
    source.write_text("甲在城门。")
    settings = Settings()
    settings.cache.directory = str(tmp_path / "cache")
    cc = main._CommandContext(settings, source.read_text(), PipelineLLM(), _source(), [], SegmentIndex([], {}, {}, {}))
    monkeypatch.setattr(main, "_prepare", lambda *_: cc)
    checkpoint = tmp_path / "run.json"
    result = CliRunner().invoke(main.app, ["continue", str(source), "--checkpoint", str(checkpoint)])
    assert result.exit_code == 0, result.output
    state = checkpoint.with_suffix(".context.json")
    combined = state.with_suffix(".source.txt")
    restored, _, _ = main._load_artifacts(input_file=combined, text=read_text_file(combined), settings=settings,
                                         llm=None, context_path=state, no_cache=True)
    assert restored.characters["甲"].location == "客栈"
    with pytest.raises(Exception, match="does not match"):
        main._load_artifacts(input_file=source, text=source.read_text(), settings=settings,
                             llm=None, context_path=state, no_cache=True)


def test_resume_retries_the_edit_that_removed_a_heading(tmp_path):
    class BadHeading(PipelineLLM):
        validations = 0
        repairs = 0

        def generate(self, prompt, system="", options=None):
            step = options["_sfy_step"]
            if step == ": validate continuation":
                self.validations += 1
                return LLMResponse(json.dumps({"passed": self.validations > 1,
                    "issues": ["澄清动作"] if self.validations == 1 else [], "repair_instructions": []}), 0)
            if step == ": final repair":
                self.repairs += 1
                old, new = ("第1章 单元1", "") if self.repairs == 1 else ("乙拔剑守在门外。", "乙持剑守在门外。")
                return LLMResponse(json.dumps({"edits": [{"old_text": old, "new_text": new, "reason": "澄清动作"}]}), 0)
            return super().generate(prompt, system, options)

    llm = BadHeading()
    path = tmp_path / "run.json"
    with pytest.raises(GenerationError, match="heading"):
        _writer(llm).continue_story("甲在城门。", _source(), checkpoint_path=path)
    result = _writer(llm).continue_story("甲在城门。", _source(), checkpoint_path=path, resume=True)
    assert "第1章 单元1" in result
    assert "乙持剑守在门外。" in result
    assert llm.repairs == 2
    assert len(llm.draft_prompts) == 3


def test_checkpoint_refuses_concurrent_access_and_modified_response(tmp_path):
    from story_for_you.core.ending.checkpoint import ContinuationCheckpoint
    from story_for_you.utils.prompting import cache_prompt
    from tests.test_review_fixes import _ResponseLLM

    path = tmp_path / "run.json"
    provider = _ResponseLLM("original")
    with ContinuationCheckpoint(provider, path, "source", resume=False) as checkpoint:
        checkpoint.generate(cache_prompt("task"))
        with pytest.raises(BlockingIOError):
            with ContinuationCheckpoint(provider, path, "source", resume=True):
                pass
    payload = json.loads(path.read_text())
    payload["calls"][0]["response"]["content"] = "changed"
    path.write_text(json.dumps(payload))
    with ContinuationCheckpoint(provider, path, "source", resume=True) as checkpoint:
        with pytest.raises(LLMResponseError, match="hash mismatch"):
            checkpoint.generate(cache_prompt("task"))
