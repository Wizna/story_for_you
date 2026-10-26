from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from story_for_you.exceptions import LLMResponseError


def thread_id(owner: str, summary: str) -> str:
    value = owner + ":" + "".join(summary.split())
    return "T-" + hashlib.sha256(value.encode()).hexdigest()[:16]


@dataclass
class StoryThread:
    thread_id: str
    summary: str
    status: str = "open"
    evidence: str = ""

    @classmethod
    def from_dict(cls, payload: Any) -> StoryThread:
        if not isinstance(payload, dict):
            raise LLMResponseError("StoryThread must be an object.")
        for key in ("thread_id", "summary", "status", "evidence"):
            if not isinstance(payload.get(key), str):
                raise LLMResponseError(f"StoryThread.{key} must be a string.")
        if payload["status"] not in {"open", "resolved"}:
            raise LLMResponseError("Invalid thread status.")
        if not payload["thread_id"].strip() or not payload["summary"].strip():
            raise LLMResponseError("Thread ID and summary must not be empty.")
        if payload["status"] == "resolved" and not payload["evidence"].strip():
            raise LLMResponseError("Resolved thread requires evidence.")
        return cls(
            **{
                key: payload[key].strip()
                for key in ("thread_id", "summary", "status", "evidence")
            }
        )


@dataclass
class ThreadUpdate:
    operation: str
    thread_id: str
    summary: str
    evidence: str

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
        if values["operation"] not in {"add", "update", "resolve"}:
            raise LLMResponseError("Invalid thread operation.")
        if not values["evidence"] or not values["summary"]:
            raise LLMResponseError("Thread changes require summary and evidence.")
        if values["operation"] != "add" and not values["thread_id"]:
            raise LLMResponseError(
                "Updating or resolving a thread requires its existing ID."
            )
        return cls(**values)
