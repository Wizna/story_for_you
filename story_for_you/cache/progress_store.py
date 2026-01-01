from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AnalysisProgress:
    """Represents the saved progress of an analysis run."""

    file_hash: str
    total_chapters: int
    completed_chapters: int
    chapter_results: list[dict[str, Any]] = field(default_factory=list)
    memory_state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize progress to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AnalysisProgress:
        """Deserialize progress from a dictionary."""
        return cls(
            file_hash=payload.get("file_hash", ""),
            total_chapters=payload.get("total_chapters", 0),
            completed_chapters=payload.get("completed_chapters", 0),
            chapter_results=payload.get("chapter_results", []),
            memory_state=payload.get("memory_state", {}),
        )


class ProgressStore:
    """Manages analysis progress for resumable runs."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = (cache_dir or Path(".story_cache")) / "progress"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_progress_path(self, file_hash: str) -> Path:
        """Get the path to the progress file for a given file hash."""
        return self.cache_dir / f"{file_hash}.json"

    def get_progress(self, file_hash: str) -> AnalysisProgress | None:
        """Load existing progress for a file."""
        path = self._get_progress_path(file_hash)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return AnalysisProgress.from_dict(payload)
        except (json.JSONDecodeError, KeyError):
            return None

    def save_progress(self, progress: AnalysisProgress) -> None:
        """Save current progress (called after each chapter)."""
        path = self._get_progress_path(progress.file_hash)
        path.write_text(
            json.dumps(progress.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def clear_progress(self, file_hash: str) -> None:
        """Clear progress when analysis completes."""
        path = self._get_progress_path(file_hash)
        if path.exists():
            path.unlink()

    def has_progress(self, file_hash: str) -> bool:
        """Check if progress exists for a file."""
        return self._get_progress_path(file_hash).exists()
