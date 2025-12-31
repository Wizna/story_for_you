from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import os

import yaml


@dataclass
class LLMSettings:
    provider: str = "ollama"
    model: str = "qwen2.5:7b-instruct"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.7
    max_tokens: int = 4096
    seed: int = 42


@dataclass
class ParserSettings:
    chunk_size: int = 4000
    overlap: int = 200


@dataclass
class CacheSettings:
    enabled: bool = True
    directory: str = ".story_cache"
    auto_save: bool = True


@dataclass
class CompressLevels:
    light: float = 0.8
    medium: float = 0.5
    heavy: float = 0.3


@dataclass
class CompressSettings:
    default_level: str = "medium"
    levels: CompressLevels = field(default_factory=CompressLevels)


@dataclass
class OutputSettings:
    add_ai_marker: bool = True
    marker_text: str = "【本文经 AI 处理】"


@dataclass
class AnalysisSettings:
    window_size: int = 12


@dataclass
class Settings:
    llm: LLMSettings = field(default_factory=LLMSettings)
    parser: ParserSettings = field(default_factory=ParserSettings)
    cache: CacheSettings = field(default_factory=CacheSettings)
    compress: CompressSettings = field(default_factory=CompressSettings)
    output: OutputSettings = field(default_factory=OutputSettings)
    analysis: AnalysisSettings = field(default_factory=AnalysisSettings)


class SettingsLoader:
    """Load settings from defaults, config files, and environment variables."""

    ENV_PREFIX = "STORY_"

    def __init__(self, config_path: Path | None = None):
        self.config_path = config_path

    def load(self) -> Settings:
        """Load settings with the documented precedence rules."""
        data: dict[str, Any] = {}
        file_data = self._load_from_file()
        if file_data:
            data = file_data
        env_data = self._load_from_env()
        if env_data:
            data = self._merge(data, env_data)
        settings = Settings()
        if data:
            self._apply(settings, data)
        return settings

    def dump_example(self) -> dict[str, Any]:
        """Expose a serializable example configuration."""
        return asdict(Settings())

    def _load_from_file(self) -> dict[str, Any]:
        path = self.config_path
        if path is None:
            candidate = Path("config.yaml")
            path = candidate if candidate.exists() else None
        if path is None or not path.exists():
            return {}
        raw = path.read_text(encoding="utf-8")
        return yaml.safe_load(raw) or {}

    def _load_from_env(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        for key, value in os.environ.items():
            if not key.startswith(self.ENV_PREFIX):
                continue
            path = key.removeprefix(self.ENV_PREFIX).lower().split("__")
            current = data
            for part in path[:-1]:
                current = current.setdefault(part, {})
            current[path[-1]] = self._coerce_env_value(value)
        return data

    def _coerce_env_value(self, value: str) -> Any:
        text = value.strip()
        lowered = text.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        for cast in (int, float):
            try:
                return cast(text)
            except ValueError:
                continue
        if "," in text:
            return [item.strip() for item in text.split(",") if item.strip()]
        return text

    def _merge(self, base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
        result = dict(base)
        for key, value in overrides.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge(result[key], value)
            else:
                result[key] = value
        return result

    def _apply(self, instance: Any, data: dict[str, Any]) -> None:
        for field_info in fields(instance):
            key = field_info.name
            if key not in data:
                continue
            value = getattr(instance, key)
            update = data[key]
            if is_dataclass(value) and isinstance(update, dict):
                self._apply(value, update)
            else:
                setattr(instance, key, update)
