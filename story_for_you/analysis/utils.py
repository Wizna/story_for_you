from __future__ import annotations

from typing import Iterable

from story_for_you.analysis.context import CharacterState


def compute_primary_cast(characters: Iterable[CharacterState], limit: int = 5) -> list[str]:
    """Return a deterministic list of protagonist names."""
    mains = [character.name for character in characters if character.role == "main"]
    if not mains:
        return []
    mains.sort()
    return mains[:limit]
