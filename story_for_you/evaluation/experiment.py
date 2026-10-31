from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Iterator

from story_for_you.analysis.context import StoryContext
from story_for_you.config.settings import Settings
from story_for_you.evaluation.continuation import digest, load_prepared, read_json, read_utf8, write_json
from story_for_you.indexer.serialization import deserialize_index, deserialize_segments, serialize_index, serialize_segments
from story_for_you.indexer.segment import Segment, SegmentIndex
from story_for_you.llm.base import LLMProvider, LLMResponse
from story_for_you.utils.artifacts import digest_json, implementation_digest, public_settings
from story_for_you.utils.prompting import CacheablePrompt


class MeasuredLLM(LLMProvider):
    def __init__(self, provider: LLMProvider, seed: int = 42, response_dir: Path | None = None) -> None:
        self.provider = provider
        self.seed = seed
        self.calls: list[dict] = []
        self.response_dir = response_dir

    def __getattr__(self, name: str) -> Any:
        return getattr(self.provider, name)

    def generate(self, prompt: CacheablePrompt, system: str = "", options: dict | None = None) -> LLMResponse:
        start = time.monotonic()
        row = {"step": (options or {}).get("_sfy_step", ""), "seed": self.seed}
        try:
            response = self.provider.generate(prompt, system, {**(options or {}), "seed": self.seed})
            row.update({key: getattr(response, key) for key in (
                "prompt_tokens", "completion_tokens", "cache_hit_prompt_tokens", "cache_miss_prompt_tokens",
            )})
            if self.response_dir is not None:
                write_json(self.response_dir / f"{len(self.calls) + 1:04d}.json", {
                    **row, "content": response.content, "finish_reason": response.finish_reason,
                })
            return response
        except Exception as exc:
            row["error_type"] = type(exc).__name__
            raise
        finally:
            row["elapsed_seconds"] = time.monotonic() - start
            self.calls.append(row)

    def generate_stream(
        self, prompt: CacheablePrompt, system: str = "", options: dict | None = None,
    ) -> Iterator[str]:
        raise NotImplementedError("Evaluation uses complete, measured responses.")

    def metrics(self) -> dict:
        return {
            "requests": len(self.calls), "failed_requests": sum("error_type" in row for row in self.calls),
            "elapsed_seconds": sum(row["elapsed_seconds"] for row in self.calls),
            "prompt_tokens": sum(row.get("prompt_tokens", 0) for row in self.calls),
            "completion_tokens": sum(row.get("completion_tokens", 0) for row in self.calls),
            "seed": self.seed, "calls": list(self.calls),
        }


def freeze_analysis(
    prepared: Path, output_dir: Path,
    analyze: Callable[[str], tuple[StoryContext, list[Segment], SegmentIndex]], settings: Settings,
) -> Path:
    cases = load_prepared(prepared)
    output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    reused = {}
    provenance = {"implementation_sha256": implementation_digest(), "settings": public_settings(settings)}
    for case in cases:
        key = case["prefix_sha256"]
        if key not in reused:
            prefix = read_utf8(prepared.parent / case["prefix"])
            if digest(prefix) != key:
                raise ValueError("Prefix changed after validation.")
            context, segments, index = analyze(prefix)
            bundle = {"prefix_sha256": key, "context": context.to_dict(), "segments": serialize_segments(segments),
                      "index": serialize_index(index), "provenance": provenance}
            filename = key + ".analysis.json"
            write_json(output_dir / filename, bundle)
            reused[key] = {"bundle": filename, "analysis_sha256": digest_json(bundle)}
        records.append({"id": case["id"], "prefix_sha256": key, **reused[key]})
        write_json(output_dir / "analysis.json", records)
    return output_dir / "analysis.json"


def load_frozen_analysis(manifest: Path, prefix: str) -> tuple[StoryContext, list[Segment], SegmentIndex, str]:
    key = digest(prefix)
    rows = read_json(manifest)
    matching = [row for row in rows if row.get("prefix_sha256") == key]
    if not matching:
        raise ValueError("Frozen analysis does not match this prefix.")
    row = matching[0]
    bundle = read_json(manifest.parent / row["bundle"])
    if digest_json(bundle) != row["analysis_sha256"] or bundle.get("prefix_sha256") != key:
        raise ValueError("Frozen analysis hash mismatch.")
    segments = deserialize_segments(bundle["segments"])
    context = StoryContext.from_dict(bundle["context"])
    return context, segments, deserialize_index(bundle["index"], segments), row["analysis_sha256"]
