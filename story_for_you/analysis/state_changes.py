from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from story_for_you.exceptions import LLMResponseError


@dataclass
class StateChange:
    field: str
    operation: str
    evidence: str
    value: str | None = None
    previous: str | None = None

    @classmethod
    def from_dict(cls, payload: Any) -> StateChange:
        if not isinstance(payload, dict):
            raise LLMResponseError("State change must be an object.")
        values = {}
        for key in ("field", "operation", "evidence", "value", "previous"):
            value = payload.get(key)
            if value is not None and not isinstance(value, str):
                raise LLMResponseError(f"State change {key} must be a string or null.")
            values[key] = value.strip() or None if isinstance(value, str) else None
        allowed = {
            "goal": {"set", "clear"}, "location": {"set", "clear"},
            "realm": {"set", "clear"}, "knowledge": {"add", "remove", "replace"},
        }
        if values["operation"] not in allowed.get(values["field"], set()):
            raise LLMResponseError("Invalid state change field/operation.")
        if not values["evidence"]:
            raise LLMResponseError("State changes require evidence.")
        if values["operation"] in {"clear", "remove", "replace"} and not values["previous"]:
            raise LLMResponseError("Clearing or replacing state requires the exact previous value.")
        if values["operation"] in {"set", "add", "replace"} and not values["value"]:
            raise LLMResponseError("State change requires a new value.")
        if values["operation"] in {"clear", "remove"} and values["value"] is not None:
            raise LLMResponseError("Clearing state must not supply a new value.")
        return cls(**values)
