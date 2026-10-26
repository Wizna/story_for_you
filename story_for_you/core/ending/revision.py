from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from story_for_you.exceptions import LLMResponseError


@dataclass
class TextEdit:
    old_text: str
    new_text: str
    reason: str

    @classmethod
    def from_dict(cls, payload: Any) -> TextEdit:
        if not isinstance(payload, dict):
            raise LLMResponseError("Revision edit must be an object.")
        for key in ("old_text", "new_text", "reason"):
            if not isinstance(payload.get(key), str):
                raise LLMResponseError(f"Revision edit {key} must be a string.")
        if not payload["old_text"].strip() or not payload["reason"].strip():
            raise LLMResponseError("Revision requires a non-empty quote and reason.")
        return cls(payload["old_text"], payload["new_text"], payload["reason"])


def apply_edits(text: str, edits: list[TextEdit]) -> str:
    if not edits:
        raise LLMResponseError("Revision requested changes but returned no edits.")
    spans: list[tuple[int, int, str]] = []
    for edit in edits:
        start = text.find(edit.old_text)
        if start < 0 or text.find(edit.old_text, start + 1) >= 0:
            raise LLMResponseError(
                "Revision quote must match exactly one location in the original draft."
            )
        if edit.old_text == edit.new_text:
            raise LLMResponseError("Revision edit must change the quoted passage.")
        spans.append((start, start + len(edit.old_text), edit.new_text))
    spans.sort()
    for previous, following in zip(spans, spans[1:]):
        if previous[1] > following[0]:
            raise LLMResponseError("Revision edits overlap; no changes were applied.")
    result = text
    for start, end, replacement in reversed(spans):
        result = result[:start] + replacement + result[end:]
    if not result.strip():
        raise LLMResponseError("Revision cannot delete the entire story.")
    return result
