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
from story_for_you.evaluation.experiment import freeze_analysis, load_frozen_analysis
from story_for_you.evaluation.continuation import GeneratedCase


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


def test_unfinished_book_end_and_repetitions_preserve_input_scope(tmp_path):
    source = tmp_path / "unfinished.txt"
    source.write_text("较早章节。最后一章仍在展开。")
    manifest = tmp_path / "input.json"
    write_json(manifest, [{"id": "ending", "source": source.name, "start_chars": 5,
                           "cutoff_chars": len(source.read_text()), "at_end": True}])
    prepared = prepare_cases(manifest, tmp_path / "prepared", repetitions=2)
    rows = read_json(prepared)
    assert [row["id"] for row in rows] == ["ending-r01", "ending-r02"]
    assert rows[0]["prefix_sha256"] == rows[1]["prefix_sha256"]
    assert rows[0]["source_range"]["start"] == 5
    assert (prepared.parent / rows[0]["prefix"]).read_text() == "最后一章仍在展开。"


def test_frozen_analysis_is_prefix_bound_and_comparison_rejects_mixed_analysis(tmp_path):
    prepared = _prepared(tmp_path)
    seen = []

    def analyze(prefix):
        seen.append(prefix)
        return StoryContext(), [], SegmentIndex([], {}, {}, {})

    settings = Settings()
    settings.llm.base_url = "https://user:secret@example.test"
    frozen = freeze_analysis(prepared, tmp_path / "analysis", analyze, settings)
    case = read_json(prepared)[0]
    prefix = (prepared.parent / case["prefix"]).read_text()
    _, _, _, fingerprint = load_frozen_analysis(frozen, prefix)
    assert len(seen) == 2
    assert all("未来秘密" not in text for text in seen)
    bundle = frozen.parent / read_json(frozen)[0]["bundle"]
    assert "secret" not in bundle.read_text()
    baseline = generate_cases(prepared, tmp_path / "a", lambda *_: GeneratedCase("正文甲", {"analysis_sha256": fingerprint}))
    candidate = generate_cases(prepared, tmp_path / "b", lambda *_: GeneratedCase("正文乙", {"analysis_sha256": "different"}))
    with pytest.raises(ValueError, match="same frozen analysis"):
        prepare_blind_comparison(prepared, baseline, candidate, tmp_path / "blind")
    data = read_json(bundle)
    data["context"]["metadata"]["future"] = "污染"
    write_json(bundle, data)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_frozen_analysis(frozen, prefix)


def test_failed_generation_is_recorded_without_exposing_exception_text(tmp_path):
    prepared = _prepared(tmp_path)

    def fail(*_):
        raise RuntimeError("private provider diagnostic")

    result = generate_cases(prepared, tmp_path / "run", fail)
    rows = read_json(result)
    assert len(rows) == 2
    assert all(row["status"] == "failed" for row in rows)
    assert "private provider diagnostic" not in result.read_text()
    with pytest.raises(ValueError, match="failed generation"):
        prepare_blind_comparison(prepared, result, result, tmp_path / "blind")


def test_crlf_source_and_output_hashes_survive_disk_roundtrip(tmp_path):
    source = tmp_path / "book.txt"
    source.write_bytes("第一章\r\n甲走进门。\r\n未来内容".encode())
    manifest = tmp_path / "input.json"
    write_json(manifest, [{"id": "crlf", "source": source.name, "cutoff_chars": len("第一章\r\n甲走进门。\r\n")}])
    prepared = prepare_cases(manifest, tmp_path / "prepared")
    def generate(prefix, *_):
        assert "\r\n" in prefix
        return "第二章\r\n门外传来脚步声。"
    result = generate_cases(prepared, tmp_path / "run", generate)
    assert read_json(result)[0]["status"] == "completed"
    prepare_blind_comparison(prepared, result, result, tmp_path / "blind")


def test_eval_analysis_refuses_to_overwrite_existing_metrics(tmp_path):
    prepared = _prepared(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    metrics = output / "metrics.json"
    metrics.write_text('{"status": "completed"}')
    result = CliRunner().invoke(app, ["eval", "analyze", str(prepared), "--output-dir", str(output)])
    assert result.exit_code != 0
    assert metrics.read_text() == '{"status": "completed"}'
