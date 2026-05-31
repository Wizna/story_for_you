from __future__ import annotations

from dataclasses import dataclass, is_dataclass
from pathlib import Path
from typing import Any

import hashlib
import json

from story_for_you.analysis.context import StoryContext
from story_for_you.utils.file_io import compute_file_hash


@dataclass
class CachedArtifacts:
    context: StoryContext
    segments: list[dict[str, Any]]
    index: dict[str, Any]
    metadata: dict[str, Any]


class ContextStore:
    """Manages persistence of StoryContext artifacts."""

    def __init__(self, cache_dir: Path | None = None, *, create: bool = True) -> None:
        self.cache_dir = cache_dir or Path(".story_cache")
        if create:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, file_path: Path, settings: dict[str, Any]) -> CachedArtifacts | None:
        """Retrieve cached artifacts for the provided file if available."""
        cache_dir = self._get_cache_dir(file_path, settings)
        ctx_path = cache_dir / "context.json"
        seg_path = cache_dir / "segments.json"
        idx_path = cache_dir / "index.json"
        meta_path = cache_dir / "meta.json"
        if not all(path.exists() for path in (ctx_path, seg_path, idx_path, meta_path)):
            return None
        context_payload = json.loads(ctx_path.read_text(encoding="utf-8"))
        segments_payload = json.loads(seg_path.read_text(encoding="utf-8"))
        index_payload = json.loads(idx_path.read_text(encoding="utf-8"))
        metadata_payload = json.loads(meta_path.read_text(encoding="utf-8"))
        return CachedArtifacts(
            context=StoryContext.from_dict(context_payload),
            segments=segments_payload,
            index=index_payload,
            metadata=metadata_payload,
        )

    def save(
        self, file_path: Path, settings: dict[str, Any], artifacts: CachedArtifacts
    ) -> Path:
        """Persist analysis artifacts to the cache directory."""
        cache_dir = self._get_cache_dir(file_path, settings)
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "context.json").write_text(
            json.dumps(artifacts.context.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (cache_dir / "segments.json").write_text(
            json.dumps(artifacts.segments, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (cache_dir / "index.json").write_text(
            json.dumps(artifacts.index, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (cache_dir / "meta.json").write_text(
            json.dumps(artifacts.metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return cache_dir

    # Internal helpers -------------------------------------------------
    def _normalize_settings(self, settings: dict[str, Any] | Any) -> dict[str, Any]:
        if is_dataclass(settings):
            return self._filter_cache_relevant(settings)
        if isinstance(settings, dict):
            return settings
        return {}

    def _filter_cache_relevant(self, settings: Any) -> dict[str, Any]:
        data = settings.__dict__
        relevant = {}
        for key in ("llm", "parser", "cache"):
            value = data.get(key)
            if value is None:
                continue
            relevant[key] = value.__dict__ if is_dataclass(value) else value
        return relevant

    def _get_cache_dir(self, file_path: Path, settings: dict[str, Any] | Any) -> Path:
        normalized = self._normalize_settings(settings)
        file_hash = self._get_file_hash(file_path)
        config_hash = self._get_config_hash(normalized)
        fingerprint = self._build_fingerprint(file_hash=file_hash, config_hash=config_hash)
        return self.cache_dir / fingerprint

    def _build_fingerprint(self, *, file_hash: str, config_hash: str) -> str:
        raw = f"{file_hash}:{config_hash}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    def _get_file_hash(self, file_path: Path) -> str:
        return compute_file_hash(file_path, length=16)

    def _get_config_hash(self, settings: dict[str, Any]) -> str:
        payload = json.dumps(settings, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
