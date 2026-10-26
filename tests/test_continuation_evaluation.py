import json

import pytest
from typer.testing import CliRunner

from story_for_you.analysis.context import StoryContext
from story_for_you.cli.main import app
from story_for_you.config.settings import Settings
from story_for_you.evaluation.continuation import (
    generate_cases,
    prepare_blind_comparison,
    prepare_cases,
    read_json,
    summarize_ratings,
    write_json,
)
from story_for_you.indexer.segment import SegmentIndex
from tests.test_review_fixes import _ResponseLLM


def _prepared(tmp_path):
    cases = []
    for name, prefix in (
        ("family", "阿云推开旧宅的门。"),
        ("mystery", "侦探看见门锁上的划痕。"),
    ):
        source = tmp_path / f"{name}.txt"
        source.write_text(prefix + "未来秘密：凶手身份尚未揭示。", encoding="utf-8")
        cases.append(
            {
                "id": name,
                "source": source.name,
                "cutoff_chars": len(prefix),
                "hint": "承接当前场景",
                "max_units": 3,
            }
        )
    manifest = tmp_path / "manifest.json"
    write_json(manifest, cases)
    return prepare_cases(manifest, tmp_path / "prepared")


def test_multibook_generation_receives_only_verified_prefixes(tmp_path):
    prepared = _prepared(tmp_path)
    seen = []

    def generate(prefix, hint, units):
        seen.append(prefix)
        assert "未来秘密" not in prefix
        assert units == 3
        return prefix + "续写内容。"

    result = generate_cases(prepared, tmp_path / "run", generate)
    assert len(seen) == len(read_json(result)) == 2
    case = read_json(prepared)[0]
    (prepared.parent / case["prefix"]).write_text("被污染的全文", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        generate_cases(prepared, tmp_path / "invalid", generate)
    assert not (tmp_path / "invalid").exists()


def test_blind_package_and_report_validate_provenance_and_ratings(tmp_path):
    prepared = _prepared(tmp_path)
    baseline = generate_cases(
        prepared, tmp_path / "baseline", lambda *_: "基线续写内容。"
    )
    candidate = generate_cases(
        prepared, tmp_path / "candidate", lambda *_: "候选续写内容。"
    )
    blind = prepare_blind_comparison(
        prepared, baseline, candidate, tmp_path / "comparison", seed=7
    )
    assert not (blind / "private_key.json").exists()
    assert not list(blind.rglob("results.json"))
    assert "未来秘密" not in (blind / "mystery" / "prefix.txt").read_text()
    key_path = blind.parent / "private_key.json"
    key = read_json(key_path)
    ratings = read_json(blind / "ratings.json")
    for case in ratings:
        case["winner"] = next(
            letter
            for letter in ("A", "B")
            if key[case["id"]][letter]["label"] == "candidate"
        )
        for letter in ("A", "B"):
            case[letter] = {
                "contradictions": 0,
                "motivation": 4,
                "knowledge_boundary": 5,
                "plot_progress": 4,
                "style": 3,
                "evidence": [],
            }
    write_json(blind / "ratings.json", ratings)
    report = summarize_ratings(blind / "ratings.json", key_path)
    assert report["results"]["candidate"]["wins"] == 2
    assert report["results"]["candidate"]["mean_scores"]["knowledge_boundary"] == 5
    ratings[0]["A"]["contradictions"] = 1
    write_json(blind / "ratings.json", ratings)
    with pytest.raises(ValueError, match="quoted evidence"):
        summarize_ratings(blind / "ratings.json", key_path)
    rows = read_json(candidate)
    rows[0]["hint"] = "另一种续写要求"
    write_json(candidate, rows)
    with pytest.raises(ValueError, match="matching prefixes"):
        prepare_blind_comparison(
            prepared, baseline, candidate, tmp_path / "invalid-comparison"
        )


def test_eval_cli_generate_uses_fresh_prefix_analysis_without_loading_artifacts(
    tmp_path, monkeypatch
):
    from story_for_you.cli import main

    prepared = _prepared(tmp_path)
    analyzed = []

    def analyze(text, settings, llm):
        analyzed.append(text)
        return StoryContext(), [], SegmentIndex([], {}, {}, {})

    class Writer:
        def __init__(self, *args, **kwargs):
            pass

        def continue_story(self, text, context, hint, max_chapters):
            assert "未来秘密" not in text
            return "承接后的正文。"

    monkeypatch.setattr(main, "_load_settings", lambda _: Settings())
    monkeypatch.setattr(main, "_build_cli_llm", lambda _: _ResponseLLM(""))
    monkeypatch.setattr(main, "_reanalyze", analyze)
    monkeypatch.setattr(
        main,
        "_load_artifacts",
        lambda **_: pytest.fail("Evaluation loaded a story cache"),
    )
    monkeypatch.setattr("story_for_you.core.ending_writer.EndingWriter", Writer)
    result = CliRunner().invoke(
        app,
        ["eval", "generate", str(prepared), "--output-dir", str(tmp_path / "cli-run")],
    )
    assert result.exit_code == 0, result.output
    assert len(analyzed) == 2
    assert len(json.loads((tmp_path / "cli-run" / "results.json").read_text())) == 2
