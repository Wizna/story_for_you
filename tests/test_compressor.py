from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from story_for_you.analysis.context import StoryContext
from story_for_you.cli import main as cli_main
from story_for_you.core.compressor import StoryCompressor
from story_for_you.core.exceptions import LLMResponseError
from story_for_you.indexer.segment import Segment, SegmentIndex
from story_for_you.llm.base import LLMProvider, LLMResponse
from story_for_you.utils.prompting import CacheablePrompt


class _RecordingLLM(LLMProvider):
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(
        self, prompt: CacheablePrompt, system: str = "", options: dict | None = None
    ) -> LLMResponse:
        self.prompts.append(prompt.render())
        return LLMResponse(content=f"压缩片段{len(self.prompts)}", tokens_used=0)

    def generate_stream(
        self, prompt: CacheablePrompt, system: str = "", options: dict | None = None
    ):
        yield from ()


def _index(segments: list[Segment]) -> SegmentIndex:
    return SegmentIndex(segments, {}, {}, {})


def test_compress_rewrites_selected_segments_without_attribute_error():
    llm = _RecordingLLM()
    compressor = StoryCompressor(llm, _index([Segment(1, "甲走入城门。")]))

    assert compressor.compress(StoryContext()) == "压缩片段1"
    assert "甲走入城门" in llm.prompts[0]


def test_compress_batches_long_input_in_source_order():
    llm = _RecordingLLM()
    segments = [Segment(1, "甲" * 90), Segment(2, "乙" * 90)]
    compressor = StoryCompressor(
        llm,
        _index(segments),
        levels={"light": 1.0, "medium": 1.0, "heavy": 1.0},
        context_window=5000,
        max_output_tokens=70,
    )

    result = compressor.compress(StoryContext())

    assert len(llm.prompts) > 1
    assert result == "\n\n".join(f"压缩片段{i}" for i in range(1, len(llm.prompts) + 1))
    assert all(len(prompt.encode("utf-8")) <= 5000 - 70 for prompt in llm.prompts)
    assert "甲" in llm.prompts[0]
    assert "乙" in llm.prompts[-1]
    assert all("甲" not in prompt for prompt in llm.prompts[next(i for i, p in enumerate(llm.prompts) if "乙" in p):])
    assert sum(prompt.count("甲") for prompt in llm.prompts) == 90
    assert sum(prompt.count("乙") for prompt in llm.prompts) == 90


def test_compress_rejects_insufficient_context_before_llm_call():
    llm = _RecordingLLM()
    compressor = StoryCompressor(
        llm, _index([Segment(1, "正文")]), context_window=100, max_output_tokens=50
    )

    with pytest.raises(LLMResponseError, match="context window is too small"):
        compressor.compress(StoryContext())
    assert llm.prompts == []


def test_compress_command_writes_output_from_existing_context(tmp_path, monkeypatch):
    source = tmp_path / "story.txt"
    context_path = tmp_path / "context.json"
    output = tmp_path / "compressed.txt"
    source.write_text("甲走入城门。", encoding="utf-8")
    context_path.write_text(
        json.dumps(StoryContext().to_dict(), ensure_ascii=False), encoding="utf-8"
    )
    llm = _RecordingLLM()
    monkeypatch.setattr(cli_main, "_build_cli_llm", lambda settings: llm)

    result = CliRunner().invoke(cli_main.app, [
        "compress", str(source), "--context", str(context_path), "--output", str(output),
    ])

    assert result.exit_code == 0, result.output
    assert output.read_text(encoding="utf-8") == "压缩片段1"
