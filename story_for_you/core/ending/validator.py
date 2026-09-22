"""LLM-backed validation for generated continuations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from story_for_you.core.ending.hint_interpreter import HintDirectives
from story_for_you.core.exceptions import LLMResponseError
from story_for_you.core.prompting import load_template
from story_for_you.llm.base import LLMProvider
from story_for_you.llm.telemetry import telemetry_options
from story_for_you.utils.json_utils import load_json_response
from story_for_you.utils.prompting import build_cacheable_prompt

__all__ = [
    "ContinuationValidationResult",
    "ContinuationValidator",
    "EndingValidationResult",
    "EndingValidator",
]


@dataclass
class ContinuationValidationResult:
    passed: bool
    issues: list[str] = field(default_factory=list)
    repair_instructions: list[str] = field(default_factory=list)


class ContinuationValidator:
    """Ask the model to review intent, continuity, and contradictions."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm
        self.template = load_template("ending_validation")

    def validate(
        self,
        text: str,
        directives: HintDirectives,
        *,
        context_block: str,
    ) -> ContinuationValidationResult:
        prompt = build_cacheable_prompt(
            context_block or "(无上下文)",
            self.template,
            prefix_placeholder="context_block",
            directives=json.dumps(asdict(directives), ensure_ascii=False, indent=2),
            final_text=text.strip(),
        )
        response = self.llm.generate(
            prompt=prompt,
            options=telemetry_options(
                {"no_think": True},
                phase="continue",
                step=": validate continuation",
            ),
        )
        payload = load_json_response(response.content)
        if not isinstance(payload, dict):
            raise LLMResponseError("Continuation validation returned invalid JSON object.")
        return self._from_payload(payload)

    def _from_payload(self, payload: dict[str, Any]) -> ContinuationValidationResult:
        for field_name in ("passed", "issues", "repair_instructions"):
            if field_name not in payload:
                raise LLMResponseError(f"Continuation validation missing required field: {field_name}")
        passed = payload.get("passed")
        if not isinstance(passed, bool):
            raise LLMResponseError("Continuation validation JSON must include boolean 'passed'.")
        return ContinuationValidationResult(
            passed=passed,
            issues=self._str_list(payload.get("issues")),
            repair_instructions=self._str_list(payload.get("repair_instructions")),
        )

    def _str_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            raise LLMResponseError("Continuation validation list fields must be JSON arrays.")
        items: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise LLMResponseError("Continuation validation list items must be strings.")
            text = item.strip()
            if text:
                items.append(text)
        return items


# Compatibility aliases for older callers and serialized integrations.
EndingValidationResult = ContinuationValidationResult
EndingValidator = ContinuationValidator
