from __future__ import annotations

from story_for_you.analysis.context import StoryContext
from story_for_you.cli import main as cli_main
from story_for_you.config.settings import Settings
from story_for_you.indexer.segment import Segment
from story_for_you.indexer.service import SegmentIndexService
from typer.testing import CliRunner


runner = CliRunner()


def _fake_reanalyze(text: str, settings: Settings, llm):
    context = StoryContext()
    segments = [
        Segment(
            segment_id=1,
            content=text,
            chapter=1,
            characters=[],
            metadata={"start": 0, "end": len(text)},
        )
    ]
    segment_index = SegmentIndexService().build(context, segments)
    return context, segments, segment_index


def test_load_artifacts_uses_configured_cache_directory(tmp_path, monkeypatch):
    input_file = tmp_path / "novel.txt"
    input_file.write_text("第一章\n故事开始。", encoding="utf-8")
    cache_dir = tmp_path / "custom-cache"
    settings = Settings()
    settings.cache.directory = str(cache_dir)
    monkeypatch.setattr(cli_main, "_reanalyze", _fake_reanalyze)

    cli_main._load_artifacts(
        input_file=input_file,
        text=input_file.read_text(encoding="utf-8"),
        settings=settings,
        llm=object(),
    )

    assert any(cache_dir.glob("*/context.json"))
    assert not (tmp_path / ".story_cache").exists()


def test_load_artifacts_respects_disabled_cache(tmp_path, monkeypatch):
    input_file = tmp_path / "novel.txt"
    input_file.write_text("第一章\n故事开始。", encoding="utf-8")
    cache_dir = tmp_path / "disabled-cache"
    settings = Settings()
    settings.cache.enabled = False
    settings.cache.directory = str(cache_dir)
    monkeypatch.setattr(cli_main, "_reanalyze", _fake_reanalyze)

    cli_main._load_artifacts(
        input_file=input_file,
        text=input_file.read_text(encoding="utf-8"),
        settings=settings,
        llm=object(),
    )

    assert not cache_dir.exists()


def test_cache_status_does_not_create_missing_configured_directory(tmp_path):
    cache_dir = tmp_path / "missing-cache"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
cache:
  directory: {cache_dir}
""",
        encoding="utf-8",
    )

    result = runner.invoke(cli_main.app, ["cache", "status", "--config", str(config_path)])

    assert result.exit_code == 0
    assert "Cache directory does not exist." in result.output
    assert not cache_dir.exists()
