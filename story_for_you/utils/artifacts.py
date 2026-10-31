from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from story_for_you.config.settings import Settings


def digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def implementation_digest() -> str:
    root = Path(__file__).resolve().parents[1]
    return digest_json({
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*")) if path.suffix in {".py", ".txt"}
    })


def public_settings(settings: Settings) -> dict:
    # Provider URLs and environment values can contain credentials. Persist only
    # an explicit allowlist of generation/analysis parameters.
    return {
        "llm": {name: getattr(settings.llm, name) for name in (
            "provider", "model", "temperature", "top_p", "top_k", "repeat_penalty",
            "max_tokens", "context_window", "seed",
        )},
        **{name: asdict(getattr(settings, name)) for name in (
            "analysis", "parser", "prompt", "ending", "rendering",
        )},
    }


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
