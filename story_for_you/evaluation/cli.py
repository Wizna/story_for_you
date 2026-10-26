from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from story_for_you.evaluation.continuation import (
    generate_cases,
    prepare_blind_comparison,
    prepare_cases,
    summarize_ratings,
    write_json,
)

app = typer.Typer(
    help="Prepare isolated cutoffs, generate continuations, and compare them blindly."
)


@app.command("prepare")
def prepare(
    manifest: Path, output_dir: Path = typer.Option(..., "--output-dir")
) -> None:
    """Create prefix-only evaluation inputs without calling an LLM."""
    typer.echo(str(prepare_cases(manifest.resolve(), output_dir)))


@app.command("generate")
def generate(
    prepared: Path,
    output_dir: Path = typer.Option(..., "--output-dir"),
    config: Optional[Path] = typer.Option(None, "--config"),
) -> None:
    """Analyze each prefix afresh and generate without loading story caches or external context."""
    from story_for_you.cli.main import _build_cli_llm, _load_settings, _reanalyze
    from story_for_you.core.ending_writer import EndingWriter

    settings = _load_settings(config)
    with _build_cli_llm(settings) as llm:

        def run(prefix: str, hint: str, max_units: int) -> str:
            if max_units > settings.ending.max_chapters:
                raise ValueError(
                    "Evaluation max_units exceeds ending.max_chapters in this configuration."
                )
            context, _, index = _reanalyze(prefix, settings, llm)
            writer = EndingWriter(
                llm,
                index,
                temperatures=settings.ending.temperatures,
                rendering_limits=settings.rendering,
                context_window=settings.llm.context_window,
                max_output_tokens=settings.llm.max_tokens,
            )
            return writer.continue_story(prefix, context, hint, max_chapters=max_units)

        result = generate_cases(
            prepared.resolve(), output_dir, run, model=settings.llm.model
        )
    typer.echo(str(result))


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
