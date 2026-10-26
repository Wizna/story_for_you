from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


@dataclass
class AnalysisProgress:
    """Represents the saved progress of an analysis run."""

    analysis_key: str
    total_units: int
    completed_units: int
    memory_state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize progress to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AnalysisProgress:
        """Deserialize progress from a dictionary."""
        if not isinstance(payload, dict):
            raise ValueError("Analysis progress must be an object")
        analysis_key = payload.get("analysis_key")
        total_units = payload.get("total_units")
        completed_units = payload.get("completed_units")
        memory_state = payload.get("memory_state")
        if (
            not isinstance(analysis_key, str)
            or not isinstance(total_units, int)
            or isinstance(total_units, bool)
            or not isinstance(completed_units, int)
            or isinstance(completed_units, bool)
            or not 1 <= completed_units <= total_units
            or not isinstance(memory_state, dict)
        ):
            raise ValueError("Invalid analysis progress")
        chapter_window = memory_state.get("chapter_window")
        if (
            not isinstance(chapter_window, dict)
            or not isinstance(chapter_window.get("history"), list)
            or len(chapter_window["history"]) != completed_units
            or not isinstance(memory_state.get("event_ledger"), dict)
            or not isinstance(memory_state.get("state_store"), dict)
        ):
            raise ValueError("Invalid analysis progress state")
        return cls(
            analysis_key=analysis_key,
            total_units=total_units,
            completed_units=completed_units,
            memory_state=memory_state,
        )


class ProgressStore:
    """Manages analysis progress for resumable runs."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = (cache_dir or Path(".story_cache")) / "progress"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_progress_path(self, analysis_key: str) -> Path:
        """Get the path to the progress file for an analysis run."""
        return self.cache_dir / f"{analysis_key}.json"

    def get_progress(self, analysis_key: str) -> AnalysisProgress | None:
        """Load existing progress for a file."""
        path = self._get_progress_path(analysis_key)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            progress = AnalysisProgress.from_dict(payload)
            return progress if progress.analysis_key == analysis_key else None
        except (json.JSONDecodeError, ValueError):
            return None

    def save_progress(self, progress: AnalysisProgress) -> None:
        """Save current progress (called after each chapter)."""
        path = self._get_progress_path(progress.analysis_key)
        payload = json.dumps(progress.to_dict(), ensure_ascii=False, indent=2)
        temporary: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent, delete=False
            ) as handle:
                temporary = Path(handle.name)
                handle.write(payload)
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def clear_progress(self, analysis_key: str) -> None:
        """Clear progress when analysis completes."""
        path = self._get_progress_path(analysis_key)
        if path.exists():
            path.unlink()

    def has_progress(self, analysis_key: str) -> bool:
        """Check if progress exists for a file."""
        return self._get_progress_path(analysis_key).exists()
