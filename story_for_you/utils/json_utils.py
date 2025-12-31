from __future__ import annotations

import json
import re
from typing import Any

__all__ = ["load_json_response"]

_CODE_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _strip_code_fence(text: str) -> str:
    """Return the payload inside the first markdown code fence, if present."""
    match = _CODE_FENCE_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _candidate_starts(payload: str) -> list[int]:
    """Return potential JSON start offsets to attempt raw decoding."""
    indices = {0}
    for symbol in ("{", "["):
        idx = payload.find(symbol)
        if idx >= 0:
            indices.add(idx)
    return sorted(indices)


def load_json_response(raw_text: str) -> Any | None:
    """Parse semistructured LLM output into JSON data if possible."""
    if not raw_text:
        return None
    candidate = _strip_code_fence(raw_text)
    if not candidate:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for start in _candidate_starts(candidate):
        try:
            data, _ = decoder.raw_decode(candidate, idx=start)
            return data
        except json.JSONDecodeError:
            continue
    return None
