from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import typer
import yaml

from story_for_you.analysis.context import StoryContext
from story_for_you.analysis.resumable_analyzer import ResumableStoryAnalyzer
from story_for_you.analysis.story_analyzer import StoryAnalyzer
from story_for_you.cache.progress_store import ProgressStore
from story_for_you.cache.store import CachedArtifacts, ContextStore
from story_for_you.config.settings import Settings, SettingsLoader
from story_for_you.core.character_filter import CharacterFilter
from story_for_you.core.character_remover import CharacterRemover
from story_for_you.core.compressor import StoryCompressor
from story_for_you.core.ending_writer import EndingWriter
from story_for_you.indexer import SegmentIndexService
from story_for_you.indexer.retriever import SegmentRetriever
from story_for_you.indexer.segment import Segment, SegmentIndex
from story_for_you.indexer.serialization import (
    deserialize_index,
    deserialize_segments,
    serialize_index,
    serialize_segments,
)
from story_for_you.llm.base import LLMProvider
from story_for_you.llm.factory import build_llm
from story_for_you.llm.telemetry import TelemetryLLMProvider
from story_for_you.parser.text_splitter import TextChunk, TextSplitter
from story_for_you.utils.file_io import compute_file_hash, read_text_file, write_text_file

app = typer.Typer(help="Story For You command line interface.")
cache_app = typer.Typer(help="Cache management commands.")
app.add_typer(cache_app, name="cache")


def _load_settings(config: Optional[Path]) -> Settings:
    loader = SettingsLoader(config_path=config)
    return loader.load()


def _build_cli_llm(settings: Settings) -> LLMProvider:
    return TelemetryLLMProvider(build_llm(settings), typer.echo)


def _cache_dir(settings: Settings) -> Path:
    return Path(settings.cache.directory or ".story_cache")


def _context_store(settings: Settings) -> ContextStore:
    return ContextStore(_cache_dir(settings))


def _readonly_context_store(settings: Settings) -> ContextStore:
    return ContextStore(_cache_dir(settings), create=False)


def _set_llm_plan(llm: LLMProvider, label: str, total_expected: int | None = None) -> None:
    if isinstance(llm, TelemetryLLMProvider):
        llm.set_plan(label=label, total_expected=total_expected)


def _announce_llm_plan(label: str, total_expected: int | None, phases: list[str]) -> None:
    estimate = f"~{total_expected}" if total_expected is not None else "unknown"
    typer.echo(f"LLM plan for {label}: baseline {estimate} request(s). Repairs/retries are logged as extra requests.")
    for phase in phases:
        typer.echo(f"  - {phase}")


def _analysis_request_estimate(chapter_count: int) -> int:
    return chapter_count * 5 + 1


def _announce_analysis_plan(chapter_count: int) -> None:
    _announce_llm_plan(
        "analyze",
        _analysis_request_estimate(chapter_count),
        [
            f"for each of {chapter_count} chapter chunk(s): characters, relationships, summary, events, story state",
            "final writing style extraction",
        ],
    )


def _has_unresolved_threads(context: StoryContext) -> bool:
    if any(character.unresolved for character in context.characters.values()):
        return True
    return bool(context.story_state and context.story_state.unresolved_events)


def _announce_continue_plan(context: StoryContext) -> int:
    has_resolution_review = _has_unresolved_threads(context)
    total = 5 + (1 if has_resolution_review else 0)
    phases = [
        "interpret user hint",
        "outline ending",
        "draft ending",
        "polish ending",
    ]
    if has_resolution_review:
        phases.append("review unresolved threads")
    phases.extend(
        [
            "validate ending",
            "optional final repair and re-validation if validation fails",
        ]
    )
    _announce_llm_plan("continue", total, phases)
    return total


def _split_text(
    text: str,
    settings: Settings,
    *,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[TextChunk]:
    context_window = settings.llm.context_window or settings.llm.max_tokens
    chunk_budget = max(settings.prompt.min_chunk, context_window - settings.prompt.margin)
    effective_chunk_size = min(chunk_size or settings.parser.chunk_size, chunk_budget)
    effective_overlap = settings.parser.overlap if overlap is None else overlap
    if effective_overlap >= effective_chunk_size:
        effective_overlap = max(0, effective_chunk_size // 4)
    splitter = TextSplitter(chunk_size=effective_chunk_size, overlap=effective_overlap)
    chunks = splitter.split(text)
    if not chunks:
        chunks = [TextChunk(content=text, start_pos=0, end_pos=len(text), chapter="1")]
    return chunks


def _analysis_chunk_size(text: str, settings: Settings) -> int:
    context_window = settings.llm.context_window or settings.llm.max_tokens
    context_budget = max(settings.prompt.min_chunk, context_window - settings.prompt.margin)
    target = min(settings.analysis.target_unit_chars, context_budget)
    text_len = len(text)
    if text_len <= target:
        return max(settings.prompt.min_chunk, text_len)

    estimated_units = max(1, (text_len + target - 1) // target)
    should_enforce_min_units = (
        estimated_units < settings.analysis.min_units
        and text_len >= target * max(2, settings.analysis.min_units // 2)
    )
    if should_enforce_min_units:
        target = max(settings.prompt.min_chunk, text_len // settings.analysis.min_units)
    return max(settings.prompt.min_chunk, min(target, context_budget))


def _split_analysis_text(text: str, settings: Settings) -> list[TextChunk]:
    chunk_size = _analysis_chunk_size(text, settings)
    overlap = min(settings.parser.overlap, max(0, chunk_size // 4))
    return _split_text(text, settings, chunk_size=chunk_size, overlap=overlap)


def _chunks_to_segments(chunks: Iterable[TextChunk]) -> list[Segment]:
    segments: list[Segment] = []
    for idx, chunk in enumerate(chunks, start=1):
        chapter_value = chunk.chapter
        try:
            chapter = int(chapter_value) if chapter_value is not None else None
        except (TypeError, ValueError):
            chapter = None
        segments.append(
            Segment(
                segment_id=idx,
                content=chunk.content,
                chapter=chapter or idx,
                characters=[],
                metadata={"start": chunk.start_pos, "end": chunk.end_pos},
            )
        )
    return segments


def _reanalyze(text: str, settings: Settings, llm):
    chunks = _split_analysis_text(text, settings)
    chapters = [chunk.content for chunk in chunks]
    total_chapters = len(chapters)
    typer.echo(f"Preparing {total_chapters} chapter-sized chunk(s) for analysis...")
    _set_llm_plan(llm, "analyze", _analysis_request_estimate(total_chapters))
    _announce_analysis_plan(total_chapters)
    analyzer = StoryAnalyzer(
        llm=llm,
        window_size=settings.analysis.window_size,
        prompt_budget=settings.llm.context_window,
    )
    progress_label = "Analyzing chapters"
    with typer.progressbar(chapters, length=total_chapters, label=progress_label) as progress_iter:
        context = analyzer.analyze(progress_iter)
    segments = _chunks_to_segments(chunks)
    segment_index = SegmentIndexService().build(context, segments)
    return context, segments, segment_index


def _load_artifacts(
    *,
    input_file: Path,
    text: str,
    settings: Settings,
    llm,
    context_path: Path | None = None,
    segments_path: Path | None = None,
    no_cache: bool = False,
    reanalyze: bool = False,
) -> Tuple[StoryContext, list[Segment], SegmentIndex]:
    use_cache = settings.cache.enabled and not no_cache
    store = _context_store(settings) if use_cache else None
    if context_path:
        payload = json.loads(context_path.read_text(encoding="utf-8"))
        context = StoryContext.from_dict(payload)
        if segments_path and segments_path.exists():
            segments_payload = json.loads(segments_path.read_text(encoding="utf-8"))
            segments = deserialize_segments(segments_payload)
        else:
            segments = _chunks_to_segments(_split_analysis_text(text, settings))
        segment_index = SegmentIndexService().build(context, segments)
        return context, segments, segment_index
    if use_cache and not reanalyze and store is not None:
        cached = store.get(input_file, settings)
        if cached:
            segments = deserialize_segments(cached.segments)
            segment_index = deserialize_index(cached.index, segments)
            return cached.context, segments, segment_index
    context, segments, segment_index = _reanalyze(text, settings, llm)
    if use_cache and settings.cache.auto_save and store is not None:
        artifacts = CachedArtifacts(
            context=context,
            segments=serialize_segments(segments),
            index=serialize_index(segment_index),
            metadata={"input": str(input_file), "segments": len(segments)},
        )
        store.save(input_file, settings, artifacts)
    return context, segments, segment_index


@dataclass
class _CommandContext:
    """四个核心命令共享的准备产物。"""

    settings: Settings
    text: str
    llm: LLMProvider
    context: StoryContext
    segments: list[Segment]
    segment_index: SegmentIndex


def _prepare(
    input_file: Path,
    config: Path | None,
    context_path: Path | None = None,
    segments_path: Path | None = None,
    no_cache: bool = False,
    reanalyze: bool = False,
) -> _CommandContext:
    settings = _load_settings(config)
    text = read_text_file(input_file)
    llm = _build_cli_llm(settings)
    context, segments, segment_index = _load_artifacts(
        input_file=input_file,
        text=text,
        settings=settings,
        llm=llm,
        context_path=context_path,
        segments_path=segments_path,
        no_cache=no_cache,
        reanalyze=reanalyze,
    )
    return _CommandContext(
        settings=settings,
        text=text,
        llm=llm,
        context=context,
        segments=segments,
        segment_index=segment_index,
    )


def _parse_character_names(characters: str) -> list[str]:
    targets = [item.strip() for item in characters.split(",") if item.strip()]
    if not targets:
        raise typer.BadParameter("Provide at least one character name.")
    return targets


def _write_analysis_output(context: StoryContext, output_path: Path, fmt: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "yaml":
        payload = yaml.safe_dump(context.to_dict(), allow_unicode=True, sort_keys=False)
        output_path.write_text(payload, encoding="utf-8")
    else:
        payload = json.dumps(context.to_dict(), ensure_ascii=False, indent=2)
        output_path.write_text(payload, encoding="utf-8")


@app.command()
def analyze(
    input_file: Path,
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
    format: str = typer.Option("json", "--format", show_default=True),
    config: Optional[Path] = typer.Option(None, "--config"),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Resume from saved progress if available"),
) -> None:
    """Analyze the input story and persist a StoryContext artifact."""
    settings = _load_settings(config)
    llm = _build_cli_llm(settings)
    if format not in {"json", "yaml"}:
        raise typer.BadParameter("Format must be 'json' or 'yaml'.")
    text = read_text_file(input_file)

    if resume and settings.cache.enabled:
        file_hash = compute_file_hash(input_file, length=24)
        progress_store = ProgressStore(_cache_dir(settings))

        existing_progress = progress_store.get_progress(file_hash)
        if existing_progress:
            typer.echo(
                f"Found existing progress: {existing_progress.completed_chapters}/{existing_progress.total_chapters} chapters completed"
            )
            typer.echo(f"Resuming from chapter {existing_progress.completed_chapters + 1}...")

        chunks = _split_analysis_text(text, settings)
        chapters = [chunk.content for chunk in chunks]
        total_chapters = len(chapters)
        typer.echo(f"Preparing {total_chapters} chapter-sized chunk(s) for analysis...")
        remaining_chapters = total_chapters
        if existing_progress and existing_progress.total_chapters == total_chapters:
            remaining_chapters = max(total_chapters - existing_progress.completed_chapters, 0)
        _set_llm_plan(llm, "analyze", _analysis_request_estimate(remaining_chapters))
        _announce_analysis_plan(remaining_chapters)

        analyzer = ResumableStoryAnalyzer(
            llm=llm,
            progress_store=progress_store,
            window_size=settings.analysis.window_size,
            prompt_budget=settings.llm.context_window,
        )

        with typer.progressbar(length=total_chapters, label="Analyzing chapters") as progress:
            def update_progress(current: int, total: int) -> None:
                progress.update(1)

            context = analyzer.analyze(chapters, file_hash, progress_callback=update_progress)

        segments = _chunks_to_segments(chunks)
        segment_index = SegmentIndexService().build(context, segments)
    else:
        context, segments, segment_index = _reanalyze(text, settings, llm)

    if settings.cache.enabled and settings.cache.auto_save:
        store = _context_store(settings)
        artifacts = CachedArtifacts(
            context=context,
            segments=serialize_segments(segments),
            index=serialize_index(segment_index),
            metadata={"input": str(input_file), "segments": len(segments)},
        )
        store.save(input_file, settings, artifacts)
    output_path = output
    if output_path is None:
        suffix = "yaml" if format == "yaml" else "json"
        output_path = input_file.with_name(f"{input_file.stem}_analysis.{suffix}")
    _write_analysis_output(context, output_path, format)
    typer.echo(f"Analysis saved to {output_path}")


@app.command()
def compress(
    input_file: Path,
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
    level: str = typer.Option("medium", "--level", show_default=True),
    config: Optional[Path] = typer.Option(None, "--config"),
    context_path: Optional[Path] = typer.Option(None, "--context"),
    segments_path: Optional[Path] = typer.Option(None, "--segments"),
    no_cache: bool = typer.Option(False, "--no-cache"),
    reanalyze: bool = typer.Option(False, "--reanalyze"),
) -> None:
    """Compress story beats according to the configured level."""
    cc = _prepare(input_file, config, context_path, segments_path, no_cache, reanalyze)
    _set_llm_plan(cc.llm, "compress", 1)
    _announce_llm_plan("compress", 1, ["rewrite selected story segments"])
    compressor = StoryCompressor(
        cc.llm, cc.segment_index, level=level, levels=cc.settings.compress.levels.__dict__,
        rendering_limits=cc.settings.rendering,
    )
    compressed = compressor.compress(cc.text, cc.context)
    target = output or input_file.with_name(f"{input_file.stem}_compressed.txt")
    write_text_file(target, compressed)
    typer.echo(f"Compressed story saved to {target}")


@app.command(name="filter")
def filter_characters(
    input_file: Path,
    characters: str = typer.Option(..., "--characters", "-c"),
    mode: str = typer.Option("soft", "--mode", show_default=True),
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
    config: Optional[Path] = typer.Option(None, "--config"),
    context_path: Optional[Path] = typer.Option(None, "--context"),
    segments_path: Optional[Path] = typer.Option(None, "--segments"),
    no_cache: bool = typer.Option(False, "--no-cache"),
    reanalyze: bool = typer.Option(False, "--reanalyze"),
) -> None:
    """Filter story content by the provided character list."""
    targets = _parse_character_names(characters)
    cc = _prepare(input_file, config, context_path, segments_path, no_cache, reanalyze)
    _set_llm_plan(cc.llm, "filter", None)
    _announce_llm_plan("filter", None, ["generate bridge text for each gap between selected segments"])
    retriever = SegmentRetriever(cc.segment_index)
    filterer = CharacterFilter(cc.llm, retriever, rendering_limits=cc.settings.rendering)
    result = filterer.filter(cc.text, targets, cc.context, mode)
    target = output or input_file.with_name(f"{input_file.stem}_filtered.txt")
    write_text_file(target, result.content)
    typer.echo(f"Filtered story saved to {target} (original ratio {result.original_ratio:.2f})")


@app.command()
def remove(
    input_file: Path,
    characters: str = typer.Option(..., "--characters", "-c"),
    mode: str = typer.Option("hard", "--mode", show_default=True),
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
    config: Optional[Path] = typer.Option(None, "--config"),
    context_path: Optional[Path] = typer.Option(None, "--context"),
    segments_path: Optional[Path] = typer.Option(None, "--segments"),
    no_cache: bool = typer.Option(False, "--no-cache"),
    reanalyze: bool = typer.Option(False, "--reanalyze"),
) -> None:
    """Remove the given characters from the source text."""
    targets = _parse_character_names(characters)
    cc = _prepare(input_file, config, context_path, segments_path, no_cache, reanalyze)
    retriever = SegmentRetriever(cc.segment_index)
    affected_count = len(retriever.retrieve_by_characters(include=targets, mode="strict"))
    _set_llm_plan(cc.llm, "remove", affected_count)
    _announce_llm_plan("remove", affected_count, ["decide rewrite/delete for each affected segment"])
    remover = CharacterRemover(cc.llm, retriever, rendering_limits=cc.settings.rendering)
    result = remover.remove(cc.text, targets, cc.context, mode)
    target = output or input_file.with_name(f"{input_file.stem}_rewritten.txt")
    write_text_file(target, result.content)
    typer.echo(
        "Removal saved to "
        + str(target)
        + f" (deleted={result.deleted_segments}, rewritten={result.rewritten_segments}, replaced={result.replaced_segments})"
    )


@app.command(name="continue")
def continue_story(
    input_file: Path,
    hint: str = typer.Option("", "--hint"),
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
    config: Optional[Path] = typer.Option(None, "--config"),
    context_path: Optional[Path] = typer.Option(None, "--context"),
    segments_path: Optional[Path] = typer.Option(None, "--segments"),
    no_cache: bool = typer.Option(False, "--no-cache"),
    reanalyze: bool = typer.Option(False, "--reanalyze"),
) -> None:
    """Continue the story with an optional hint about the desired ending."""
    cc = _prepare(input_file, config, context_path, segments_path, no_cache, reanalyze)
    continue_requests = _announce_continue_plan(cc.context)
    _set_llm_plan(cc.llm, "continue", continue_requests)
    writer = EndingWriter(
        cc.llm, cc.segment_index,
        temperatures=cc.settings.ending.temperatures,
        rendering_limits=cc.settings.rendering,
    )
    continuation = writer.continue_story(cc.text, cc.context, hint)
    target = output or input_file.with_name(f"{input_file.stem}_ending.txt")
    write_text_file(target, continuation)
    typer.echo(f"Continuation saved to {target}")


@cache_app.command()
def clear(config: Optional[Path] = typer.Option(None, "--config")) -> None:
    """Clear cached analysis artifacts."""
    settings = _load_settings(config)
    store = _context_store(settings)
    if store.cache_dir.exists():
        shutil.rmtree(store.cache_dir)
    store.cache_dir.mkdir(parents=True, exist_ok=True)
    typer.echo("Cache cleared.")


@cache_app.command()
def status(config: Optional[Path] = typer.Option(None, "--config")) -> None:
    """Show cache status summary."""
    settings = _load_settings(config)
    store = _readonly_context_store(settings)
    cache_dir = store.cache_dir
    if not cache_dir.exists():
        typer.echo("Cache directory does not exist.")
        return
    entries = [path for path in cache_dir.iterdir() if path.is_dir()]
    total_size = 0
    for file_path in cache_dir.rglob("*"):
        if file_path.is_file():
            total_size += file_path.stat().st_size
    typer.echo(f"{len(entries)} entries, {(total_size / 1024):.1f} KiB total.")
