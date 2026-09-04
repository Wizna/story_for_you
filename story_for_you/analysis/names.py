"""Normalization helpers for character labels produced by LLM extraction."""

from __future__ import annotations

import re

_POSSESSIVE_KINSHIP = re.compile(r"^(?P<owner>.+?)的(?P<kin>母亲|父亲|祖父|祖母|爷爷|奶奶|外祖父|外祖母)$")


def normalize_character_label(label: str) -> str:
    """Return a comparison key while preserving meaningful character names.

    Chinese fiction commonly alternates between forms such as "翠翠母亲" and
    "翠翠的母亲".  Treat these as the same label, while keeping the original
    display name in the structured context.
    """
    compact = "".join(label.split()).casefold()
    match = _POSSESSIVE_KINSHIP.fullmatch(compact)
    if match:
        return f"{match.group('owner')}{match.group('kin')}"
    return compact
