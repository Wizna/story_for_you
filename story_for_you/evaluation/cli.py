from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from story_for_you.evaluation.continuation import (
    GeneratedCase,
    generate_cases,
    prepare_blind_comparison,
    prepare_cases,
    summarize_ratings,
    write_json,
    read_json,
)
from story_for_you.evaluation.experiment import MeasuredLLM, freeze_analysis, load_frozen_analysis
from story_for_you.utils.artifacts import implementation_digest, public_settings

app = typer.Typer(
    help="Prepare isolated cutoffs, generate continuations, and compare them blindly."
)


@app.command("prepare")
def prepare(
    manifest: Path, output_dir: Path = typer.Option(..., "--output-dir"),
    repetitions: int = typer.Option(1, "--repetitions", min=1),
) -> None:
    """Create prefix-only evaluation inputs without calling an LLM."""
    typer.echo(str(prepare_cases(manifest.resolve(), output_dir, repetitions)))


@app.command("analyze")
def analyze(
    prepared: Path, output_dir: Path = typer.Option(..., "--output-dir"),
    config: Optional[Path] = typer.Option(None, "--config"),
) -> None:
    """Freeze one fresh prefix-only analysis for all writer variants."""
    from story_for_you.cli.main import _build_cli_llm, _load_settings, _reanalyze

    if output_dir.exists():
        raise typer.BadParameter("Analysis output directory must be new.")
    settings = _load_settings(config)
    with _build_cli_llm(settings) as provider:
        measured = MeasuredLLM(provider, settings.llm.seed, output_dir / "responses")
        status = {"status": "failed"}
        try:
            result = freeze_analysis(
                prepared.resolve(), output_dir, lambda prefix: _reanalyze(prefix, settings, measured), settings,
            )
            status = {"status": "completed"}
        except Exception as exc:
            status["error_type"] = type(exc).__name__
            raise
        finally:
            if output_dir.exists():
                write_json(output_dir / "metrics.json", {**measured.metrics(), **status})
    typer.echo(str(result))


@app.command("generate")
def generate(
    prepared: Path,
    output_dir: Path = typer.Option(..., "--output-dir"),
    config: Optional[Path] = typer.Option(None, "--config"),
    analysis: Optional[Path] = typer.Option(None, "--analysis"),
    revision_mode: Optional[str] = typer.Option(None, "--revision-mode"),
) -> None:
    """Generate from a frozen prefix analysis, or analyze afresh when none is provided."""
    from story_for_you.cli.main import _build_cli_llm, _load_settings, _reanalyze
    from story_for_you.core.ending_writer import EndingWriter

    settings = _load_settings(config)
    mode = revision_mode or settings.ending.revision_mode
    if mode not in {"none", "full", "local"}:
        raise typer.BadParameter("revision-mode must be none, full, or local")
    last = {}
    run_index = 0
    provenance = {"implementation_sha256": implementation_digest(), "settings": public_settings(settings),
                  "revision_mode": mode}
    with _build_cli_llm(settings) as provider:
        llm = MeasuredLLM(provider, settings.llm.seed)

        def run(prefix: str, hint: str, max_units: int) -> GeneratedCase:
            nonlocal run_index
            llm.calls.clear()
            llm.seed = settings.llm.seed + run_index
            run_index += 1
            llm.response_dir = output_dir / f"responses-{run_index:03d}"
            last.clear()
            if max_units > settings.ending.max_chapters:
                raise ValueError(
                    "Evaluation max_units exceeds ending.max_chapters in this configuration."
                )
            if analysis:
                context, _, index, analysis_hash = load_frozen_analysis(analysis.resolve(), prefix)
                last["analysis_sha256"] = analysis_hash
            else:
                context, _, index = _reanalyze(prefix, settings, llm)
            writer = EndingWriter(
                llm,
                index,
                temperatures=settings.ending.temperatures,
                rendering_limits=settings.rendering,
                context_window=settings.llm.context_window,
                max_output_tokens=settings.llm.max_tokens,
                revision_mode=mode,
            )
            try:
                text = writer.continue_story(prefix, context, hint, max_chapters=max_units)
                return GeneratedCase(text, {**last, "metrics": llm.metrics()}, getattr(writer, "trace", []))
            finally:
                last["trace"] = getattr(writer, "trace", [])

        result = generate_cases(
            prepared.resolve(), output_dir, run, model=settings.llm.model, provenance=provenance,
            failure_metadata=lambda: {**last, "metrics": llm.metrics()},
        )
    typer.echo(str(result))
    if any(row.get("status") == "failed" for row in read_json(result)):
        raise typer.Exit(1)


@app.command("blind")
def blind(
    prepared: Path,
    baseline: Path,
    candidate: Path,
    output_dir: Path = typer.Option(..., "--output-dir"),
    seed: Optional[int] = typer.Option(
        None,
        "--seed",
        help="Optional reproducible assignment; keep private from reviewers.",
    ),
) -> None:
    """Package anonymized A/B texts; keep the answer key outside the blind directory."""
    typer.echo(
        str(
            prepare_blind_comparison(
                prepared.resolve(),
                baseline.resolve(),
                candidate.resolve(),
                output_dir,
                seed=seed,
            )
        )
    )


@app.command("report")
def report(
    ratings: Path,
    key: Path,
    output: Path = typer.Option(..., "--output", "-o"),
) -> None:
    """Aggregate completed human ratings into descriptive metrics."""
    result = summarize_ratings(ratings, key)
    write_json(output, result)
    typer.echo(str(output))
