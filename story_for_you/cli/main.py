from pathlib import Path
from typing import Optional

import typer

from story_for_you.analysis.story_analyzer import StoryAnalyzer
from story_for_you.cache.store import ContextStore
from story_for_you.config.settings import Settings, SettingsLoader
from story_for_you.llm.ollama import OllamaProvider

app = typer.Typer(help="Story For You command line interface.")
cache_app = typer.Typer(help="Cache management commands.")
app.add_typer(cache_app, name="cache")


def _load_settings(config: Optional[Path]) -> Settings:
    loader = SettingsLoader(config_path=config)
    return loader.load()


def _build_llm(settings: Settings) -> OllamaProvider:
    return OllamaProvider(model=settings.llm.model, base_url=settings.llm.base_url)


@app.command()
def analyze(
    input_file: Path,
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
    format: str = typer.Option("json", "--format", show_default=True),
    config: Optional[Path] = typer.Option(None, "--config"),
) -> None:
    """Analyze the input story and persist a StoryContext artifact."""
    settings = _load_settings(config)
    llm = _build_llm(settings)
    _ = StoryAnalyzer(llm=llm, window_size=settings.analysis.window_size)
    _ = ContextStore()
    typer.echo("Analyze pipeline placeholder - implementation pending.")


@app.command()
def compress(
    input_file: Path,
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
    level: str = typer.Option("medium", "--level", show_default=True),
    config: Optional[Path] = typer.Option(None, "--config"),
) -> None:
    """Compress story beats according to the configured level."""
    _ = _load_settings(config)
    typer.echo(f"Compression pipeline placeholder (level={level}).")


@app.command(name="filter")
def filter_characters(
    input_file: Path,
    characters: str = typer.Option(..., "--characters", "-c"),
    mode: str = typer.Option("soft", "--mode", show_default=True),
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
    config: Optional[Path] = typer.Option(None, "--config"),
) -> None:
    """Filter story content by the provided character list."""
    _ = _load_settings(config)
    targets = [item.strip() for item in characters.split(",") if item.strip()]
    typer.echo("Character filter placeholder for: " + ", ".join(targets))


@app.command()
def remove(
    input_file: Path,
    characters: str = typer.Option(..., "--characters", "-c"),
    mode: str = typer.Option("hard", "--mode", show_default=True),
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
    config: Optional[Path] = typer.Option(None, "--config"),
) -> None:
    """Remove the given characters from the source text."""
    _ = _load_settings(config)
    targets = [item.strip() for item in characters.split(",") if item.strip()]
    typer.echo("Character removal placeholder for: " + ", ".join(targets))


@app.command(name="continue")
def continue_story(
    input_file: Path,
    hint: str = typer.Option("", "--hint"),
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
    config: Optional[Path] = typer.Option(None, "--config"),
) -> None:
    """Continue the story with an optional hint about the desired ending."""
    _ = _load_settings(config)
    typer.echo(f"Ending writer placeholder. Hint: {hint or 'n/a'}")


@cache_app.command()
def clear() -> None:
    """Clear cached analysis artifacts."""
    typer.echo("Cache clearing placeholder.")


@cache_app.command()
def status() -> None:
    """Show cache status summary."""
    typer.echo("Cache status placeholder.")
