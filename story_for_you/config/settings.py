from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import os

import yaml


@dataclass
class LLMSettings:
    provider: str = "ollama"
    model: str = "qwen3:8b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    max_tokens: int = 4096
    timeout: float = 300.0
    seed: int = 42
    api_key: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature must be between 0.0 and 2.0, got {self.temperature}")
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError(f"top_p must be between 0.0 and 1.0, got {self.top_p}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be non-negative, got {self.top_k}")
        if self.repeat_penalty < 0.0:
            raise ValueError(f"repeat_penalty must be non-negative, got {self.repeat_penalty}")
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")
        if self.timeout <= 0:
            raise ValueError(f"timeout must be positive, got {self.timeout}")


@dataclass
class ParserSettings:
    chunk_size: int = 4000
    overlap: int = 200

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.overlap < 0:
            raise ValueError(f"overlap must be non-negative, got {self.overlap}")
        if self.overlap >= self.chunk_size:
            raise ValueError(f"overlap ({self.overlap}) must be less than chunk_size ({self.chunk_size})")


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

    def __post_init__(self) -> None:
        for level_name in ("light", "medium", "heavy"):
            value = getattr(self, level_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{level_name} level must be between 0.0 and 1.0, got {value}")


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

    def __post_init__(self) -> None:
        if self.window_size <= 0:
            raise ValueError(f"window_size must be positive, got {self.window_size}")


@dataclass
class PromptSettings:
    """Settings for prompt budget calculation."""

    margin: int = 2300  # Reserve space for instructions/context
    min_chunk: int = 800  # Minimum chunk size

    def __post_init__(self) -> None:
        if self.margin < 0:
            raise ValueError(f"margin must be non-negative, got {self.margin}")
        if self.min_chunk <= 0:
            raise ValueError(f"min_chunk must be positive, got {self.min_chunk}")


@dataclass
class EndingPhaseTemperatures:
    """Temperature settings for each ending writer phase."""

    inspiration: float = 0.55
    outline: float = 0.55
    draft: float = 0.65
    revision: float = 0.35
    polish: float = 0.35
    resolution: float = 0.35
    legacy: float = 0.7

    def __post_init__(self) -> None:
        for phase in ("inspiration", "outline", "draft", "revision", "polish", "resolution", "legacy"):
            value = getattr(self, phase)
            if not 0.0 <= value <= 2.0:
                raise ValueError(f"{phase} temperature must be between 0.0 and 2.0, got {value}")


@dataclass
class EndingSettings:
    """Settings for ending writer."""

    temperatures: EndingPhaseTemperatures = field(default_factory=EndingPhaseTemperatures)


@dataclass
class Settings:
    llm: LLMSettings = field(default_factory=LLMSettings)
    parser: ParserSettings = field(default_factory=ParserSettings)
    cache: CacheSettings = field(default_factory=CacheSettings)
    compress: CompressSettings = field(default_factory=CompressSettings)
    output: OutputSettings = field(default_factory=OutputSettings)
    analysis: AnalysisSettings = field(default_factory=AnalysisSettings)
    prompt: PromptSettings = field(default_factory=PromptSettings)
    ending: EndingSettings = field(default_factory=EndingSettings)


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
