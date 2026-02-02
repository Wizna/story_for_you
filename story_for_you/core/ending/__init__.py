"""Ending writer submodule.

Provides utilities for multi-stage story continuation:
- Constants for quality filtering
- Hint interpretation for user directives
- Style enforcement for output quality
"""

from story_for_you.core.ending.constants import (
    LOW_QUALITY_PHRASES,
    QUESTION_MARKERS,
    SCENE_KEYWORDS,
)
from story_for_you.core.ending.hint_interpreter import HintDirectives, HintInterpreter
from story_for_you.core.ending.style_enforcer import StyleEnforcer

__all__ = [
    "LOW_QUALITY_PHRASES",
    "QUESTION_MARKERS",
    "SCENE_KEYWORDS",
    "HintDirectives",
    "HintInterpreter",
    "StyleEnforcer",
]
