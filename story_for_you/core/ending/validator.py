"""LLM-backed validation for generated endings."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from story_for_you.core.ending.hint_interpreter import HintDirectives
from story_for_you.core.exceptions import LLMResponseError
from story_for_you.core.prompting import fill_template, load_template
from story_for_you.llm.base import LLMProvider
from story_for_you.utils.json_utils import load_json_response

__all__ = ["EndingValidationResult", "EndingValidator"]


@dataclass
class EndingValidationResult:
    passed: bool
    issues: list[str] = field(default_factory=list)
    repair_instructions: list[str] = field(default_factory=list)


class EndingValidator:
    """Asks the model to review user intent, continuity, and contradictions."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm
        self.template = load_template("ending_validation")

    def validate(
        self,
        text: str,
        directives: HintDirectives,
        *,
        context_block: str,
    ) -> EndingValidationResult:
        prompt = fill_template(
            self.template,
            directives=json.dumps(asdict(directives), ensure_ascii=False, indent=2),
            context_block=context_block or "(无上下文)",
            final_text=text.strip(),
        )
        response = self.llm.generate(prompt=prompt, options={"no_think": True})
        payload = load_json_response(response.content)
        if not isinstance(payload, dict):
            raise LLMResponseError("Ending validation returned invalid JSON object.")
        return self._from_payload(payload)

    def _from_payload(self, payload: dict[str, Any]) -> EndingValidationResult:
        passed = payload.get("passed")
        if not isinstance(passed, bool):
            raise LLMResponseError("Ending validation JSON must include boolean 'passed'.")
        return EndingValidationResult(
            passed=passed,
            issues=self._str_list(payload.get("issues")),
            repair_instructions=self._str_list(payload.get("repair_instructions")),
        )

    def _str_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise LLMResponseError("Ending validation list fields must be JSON arrays.")
        return [str(item).strip() for item in value if str(item).strip()]
