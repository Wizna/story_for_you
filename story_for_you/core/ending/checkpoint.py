from __future__ import annotations

import fcntl
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

from story_for_you.exceptions import LLMResponseError
from story_for_you.llm.base import LLMProvider, LLMResponse
from story_for_you.utils.artifacts import atomic_write_json, digest_json
from story_for_you.utils.prompting import CacheablePrompt


class ContinuationCheckpoint(LLMProvider):
    """Replay accepted stage responses to reconstruct exactly the same story state."""

    def __init__(self, provider: LLMProvider, path: Path, signature: str, *, resume: bool) -> None:
        self.provider = provider
        self.path = path
        self.signature = signature
        self.resume = resume
        self.position = 0
        self.pending: int | None = None
        self.payload: dict = {}
        self.lock = None

    def __enter__(self) -> ContinuationCheckpoint:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = self.path.with_suffix(self.path.suffix + ".lock").open("a+")
        try:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            if self.resume:
                self.payload = json.loads(self.path.read_text(encoding="utf-8"))
                if (self.payload.get("version") != 1 or self.payload.get("signature") != self.signature
                        or not isinstance(self.payload.get("calls"), list)):
                    raise LLMResponseError("Continuation checkpoint does not match source, analysis, settings or implementation.")
            else:
                if self.path.exists():
                    raise FileExistsError("Checkpoint exists; use --resume or choose a new checkpoint path.")
                self.payload = {"version": 1, "signature": self.signature, "calls": [], "completed": False}
                self._save()
            return self
        except BaseException:
            self.close()
            raise

    def generate(self, prompt: CacheablePrompt, system: str = "", options: dict | None = None) -> LLMResponse:
        # Reaching the next request acknowledges parsing and application of the
        # preceding response. A network failure must not discard that response.
        self.pending = None
        key = digest_json({"prefix": prompt.prefix, "task": prompt.task, "system": system, "options": options or {}})
        calls = self.payload["calls"]
        if self.position < len(calls):
            record = calls[self.position]
            if (record.get("request_sha256") != key
                    or record.get("response_sha256") != digest_json(record.get("response"))):
                raise LLMResponseError("Continuation checkpoint stage hash mismatch.")
            response = LLMResponse(**record["response"])
            response.ensure_complete()
        else:
            response = self.provider.generate(prompt, system, options)
            response.ensure_complete()
            row = asdict(response)
            calls.append({"request_sha256": key, "response_sha256": digest_json(row), "response": row})
            self._save()
        self.pending = self.position
        self.position += 1
        return response

    def generate_stream(
        self, prompt: CacheablePrompt, system: str = "", options: dict | None = None,
    ) -> Iterator[str]:
        raise NotImplementedError("Continuation checkpoints store complete stage responses.")

    def discard_pending(self) -> None:
        if self.pending is not None:
            del self.payload["calls"][self.pending:]
        self.payload["completed"] = False
        self.payload.pop("result", None)
        self._save()

    def finish(self, text: str, context: dict) -> None:
        self.pending = None
        self.payload["completed"] = True
        self.payload["result"] = {"text": text, "context": context}
        self._save()

    def _save(self) -> None:
        atomic_write_json(self.path, self.payload)

    def close(self) -> None:
        if self.lock is not None:
            self.lock.close()
            self.lock = None
