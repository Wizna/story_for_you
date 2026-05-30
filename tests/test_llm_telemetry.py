from __future__ import annotations

from story_for_you.llm.base import LLMProvider, LLMResponse
from story_for_you.llm.telemetry import TelemetryLLMProvider, telemetry_options


class _FakeLLM(LLMProvider):
    model = "deepseek-v4-pro"

    def __init__(self):
        self.options_seen = None

    def generate(self, prompt: str, system: str = "", options: dict | None = None) -> LLMResponse:
        self.options_seen = options
        return LLMResponse(
            content="ok",
            tokens_used=350,
            prompt_tokens=300,
            completion_tokens=50,
            cache_hit_prompt_tokens=200,
            cache_miss_prompt_tokens=100,
        )

    def generate_stream(self, prompt: str, system: str = "", options: dict | None = None):
        yield from []


def test_telemetry_logs_phase_attempt_usage_and_cost():
    messages: list[str] = []
    fake = _FakeLLM()
    llm = TelemetryLLMProvider(fake, messages.append)
    llm.set_plan("analyze", total_expected=2)

    response = llm.generate(
        "第一行\n第二行",
        options=telemetry_options(
            {"temperature": 0.1},
            phase="analyze chapter 1",
            step=": chapter summary",
            attempt=1,
            max_attempts=2,
        ),
    )

    assert response.content == "ok"
    assert fake.options_seen == {"temperature": 0.1}
    joined = "\n".join(messages)
    assert "[LLM 1/2] analyze chapter 1: chapter summary attempt=1/2" in joined
    assert "prompt: 第一行 / 第二行" in joined
    assert "in=300 out=50 total=350 cache=200" in joined
    assert "cost=$0.000174" in joined
    assert "remaining~1" in joined


def test_telemetry_strips_internal_options():
    payload = telemetry_options({"no_think": True}, phase="continue", step=": draft")

    assert payload["no_think"] is True
    assert payload["_sfy_phase"] == "continue"
    assert payload["_sfy_step"] == ": draft"
