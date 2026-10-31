from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from story_for_you.exceptions import LLMResponseError


def thread_id(owner: str, summary: str) -> str:
    value = owner + ":" + "".join(summary.split())
    return "T-" + hashlib.sha256(value.encode()).hexdigest()[:16]


@dataclass
class ThreadEvidence:
    """One source-backed observation in a thread's lifetime."""

    summary: str
    evidence: str
    chapter: int | None = None
    source_unit: int | None = None

    @classmethod
    def from_dict(cls, payload: Any) -> "ThreadEvidence":
        if not isinstance(payload, dict):
            raise LLMResponseError("ThreadEvidence must be an object.")
        summary = payload.get("summary")
        evidence = payload.get("evidence")
        if not isinstance(summary, str) or not summary.strip():
            raise LLMResponseError("ThreadEvidence.summary must not be empty.")
        if not isinstance(evidence, str) or not evidence.strip():
            raise LLMResponseError("ThreadEvidence.evidence must not be empty.")
        chapter = payload.get("chapter")
        source_unit = payload.get("source_unit")
        if chapter is not None and (not isinstance(chapter, int) or isinstance(chapter, bool)):
            raise LLMResponseError("ThreadEvidence.chapter must be an integer or null.")
        if source_unit is not None and (
            not isinstance(source_unit, int) or isinstance(source_unit, bool) or source_unit < 1
        ):
            raise LLMResponseError("ThreadEvidence.source_unit must be a positive integer or null.")
        return cls(summary.strip(), evidence.strip(), chapter, source_unit)


@dataclass
class StoryThread:
    thread_id: str
    summary: str
    status: str = "open"
    evidence: str = ""
    kind: str = "unclassified"
    confidence: str = "medium"
    candidate_outcomes: list[str] = field(default_factory=list)
    history: list[ThreadEvidence] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: Any) -> StoryThread:
        if not isinstance(payload, dict):
            raise LLMResponseError("StoryThread must be an object.")
        for key in ("thread_id", "summary", "status", "evidence"):
            if not isinstance(payload.get(key), str):
                raise LLMResponseError(f"StoryThread.{key} must be a string.")
        if payload["status"] not in {"open", "resolved", "dormant", "superseded"}:
            raise LLMResponseError("Invalid thread status.")
        if not payload["thread_id"].strip() or not payload["summary"].strip():
            raise LLMResponseError("Thread ID and summary must not be empty.")
        if payload["status"] in {"resolved", "superseded"} and not payload["evidence"].strip():
            raise LLMResponseError("Resolved thread requires evidence.")
        kind = payload.get("kind", "unclassified")
        confidence = payload.get("confidence", "medium")
        if not isinstance(kind, str) or not kind.strip():
            raise LLMResponseError("StoryThread.kind must be a non-empty string.")
        if not isinstance(confidence, str) or confidence not in {"high", "medium", "low"}:
            raise LLMResponseError("StoryThread.confidence must be high, medium, or low.")
        outcomes = payload.get("candidate_outcomes", [])
        if not isinstance(outcomes, list) or any(not isinstance(item, str) for item in outcomes):
            raise LLMResponseError("StoryThread.candidate_outcomes must be a string list.")
        history_payload = payload.get("history", [])
        if not isinstance(history_payload, list):
            raise LLMResponseError("StoryThread.history must be a list.")
        return cls(
            thread_id=payload["thread_id"].strip(),
            summary=payload["summary"].strip(),
            status=payload["status"].strip(),
            evidence=payload["evidence"].strip(),
            kind=kind.strip(),
            confidence=confidence,
            candidate_outcomes=[item.strip() for item in outcomes if item.strip()],
            history=[ThreadEvidence.from_dict(item) for item in history_payload],
        )


@dataclass
class ThreadUpdate:
    operation: str
    thread_id: str
    summary: str
    evidence: str
    kind: str = "unclassified"
    confidence: str = "medium"
    candidate_outcomes: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: Any) -> ThreadUpdate:
        if not isinstance(payload, dict):
            raise LLMResponseError("ThreadUpdate must be an object.")
        for key in ("operation", "thread_id", "summary", "evidence"):
            if not isinstance(payload.get(key), str):
                raise LLMResponseError(f"ThreadUpdate.{key} must be a string.")
        values = {
            key: payload[key].strip()
            for key in ("operation", "thread_id", "summary", "evidence")
        }
        if values["operation"] not in {"add", "update", "advance", "resolve", "supersede"}:
            raise LLMResponseError("Invalid thread operation.")
        if not values["evidence"] or not values["summary"]:
            raise LLMResponseError("Thread changes require summary and evidence.")
        if values["operation"] != "add" and not values["thread_id"]:
            raise LLMResponseError(
                "Updating or resolving a thread requires its existing ID."
            )
        kind = payload.get("kind", "unclassified")
        confidence = payload.get("confidence", "medium")
        outcomes = payload.get("candidate_outcomes", [])
        if not isinstance(kind, str) or not kind.strip():
            raise LLMResponseError("ThreadUpdate.kind must be a non-empty string.")
        if not isinstance(confidence, str) or confidence not in {"high", "medium", "low"}:
            raise LLMResponseError("ThreadUpdate.confidence must be high, medium, or low.")
        if not isinstance(outcomes, list) or any(not isinstance(item, str) for item in outcomes):
            raise LLMResponseError("ThreadUpdate.candidate_outcomes must be a string list.")
        return cls(
            **values,
            kind=kind.strip(),
            confidence=confidence,
            candidate_outcomes=[item.strip() for item in outcomes if item.strip()],
        )
