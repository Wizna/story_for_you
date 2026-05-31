from __future__ import annotations

import pytest

from story_for_you.config.settings import SettingsLoader


def test_loader_validates_file_overrides_after_apply(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
parser:
  chunk_size: 10
  overlap: 10
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="overlap .* must be less than chunk_size"):
        SettingsLoader(config_path=config_path).load()


def test_loader_validates_environment_overrides(monkeypatch):
    monkeypatch.setenv("STORY_LLM__CONTEXT_WINDOW", "0")

    with pytest.raises(ValueError, match="context_window must be positive"):
        SettingsLoader().load()
