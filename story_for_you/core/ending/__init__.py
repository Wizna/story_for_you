"""Ending writer submodule.

Provides utilities for multi-stage story continuation:
- Constants for quality filtering
- Hint interpretation for user directives
- Style enforcement for output quality
"""

from story_for_you.core.ending.constants import (
    BANNED_EXPRESSIONS_PROMPT,
)
from story_for_you.core.ending.hint_interpreter import HintDirectives, HintInterpreter
from story_for_you.core.ending.style_enforcer import StyleEnforcer
from story_for_you.core.ending.validator import EndingValidationResult, EndingValidator

__all__ = [
    "BANNED_EXPRESSIONS_PROMPT",
    "HintDirectives",
    "HintInterpreter",
    "StyleEnforcer",
    "EndingValidationResult",
    "EndingValidator",
]
