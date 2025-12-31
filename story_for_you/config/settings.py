from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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
    """Lightweight settings loader placeholder."""

    def __init__(self, config_path: Path | None = None):
        self.config_path = config_path

    def load(self) -> Settings:
        """Return default settings until YAML parsing is implemented."""
        return Settings()

    def dump_example(self) -> dict[str, Any]:
        """Expose a serializable example configuration."""
        return Settings().__dict__
