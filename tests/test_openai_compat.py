"""Tests for OpenAICompatibleProvider and factory integration."""

from __future__ import annotations

import json

import httpx
import pytest

from story_for_you.config.settings import Settings
from story_for_you.core.exceptions import (
    ConfigurationError,
    LLMConnectionError,
    LLMResponseError,
    LLMTimeoutError,
)
from story_for_you.llm.factory import build_llm
from story_for_you.llm.openai_compat import OpenAICompatibleProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok_response(content: str = "Hello!", tokens: int = 10) -> dict:
    """Return a minimal OpenAI-style chat completion response."""
    return {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": tokens},
    }


def _deepseek_usage_response(content: str = "Hello!") -> dict:
    return {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 300,
            "completion_tokens": 50,
            "total_tokens": 350,
            "prompt_cache_hit_tokens": 200,
            "prompt_cache_miss_tokens": 100,
        },
    }


def _sse_chunks(texts: list[str]) -> str:
    """Build a raw SSE response body from a list of delta content strings."""
    lines: list[str] = []
    for text in texts:
        chunk = {
            "choices": [{"delta": {"content": text}}],
        }
        lines.append(f"data: {json.dumps(chunk)}")
        lines.append("")  # blank line between events
    lines.append("data: [DONE]")
    lines.append("")
    return "\n".join(lines)


def _make_transport(handler):
    """Create an httpx.MockTransport from a request handler function."""
    return httpx.MockTransport(handler)


def _provider(transport: httpx.MockTransport, **kwargs) -> OpenAICompatibleProvider:
    """Build an OpenAICompatibleProvider wired to a mock transport."""
    provider = OpenAICompatibleProvider(
        model=kwargs.get("model", "test-model"),
        base_url=kwargs.get("base_url", "https://api.example.com"),
        api_key=kwargs.get("api_key", "sk-test-key"),
        timeout=kwargs.get("timeout", 30.0),
        options=kwargs.get("options"),
    )
    # Patch internal client to use our mock transport.
    provider._client = httpx.Client(
        transport=transport,
        base_url=provider.base_url,
        timeout=provider.timeout,
        headers=provider._client.headers,
    )
    return provider


# ---------------------------------------------------------------------------
# generate() tests
# ---------------------------------------------------------------------------

class TestGenerate:
    def test_successful_generate(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/v1/chat/completions"
            body = json.loads(request.content)
            assert body["model"] == "test-model"
            assert body["stream"] is False
            assert body["messages"][-1]["content"] == "Say hi"
            return httpx.Response(200, json=_ok_response("Hi there!", 42))

        p = _provider(_make_transport(handler))
        resp = p.generate("Say hi")
        assert resp.content == "Hi there!"
        assert resp.tokens_used == 42
        assert resp.prompt_tokens == 5
        assert resp.completion_tokens == 5

    def test_generate_extracts_deepseek_cache_usage(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_deepseek_usage_response())

        p = _provider(
            _make_transport(handler),
            base_url="https://api.deepseek.com",
            model="deepseek-v4-pro",
        )

        resp = p.generate("Say hi")

        assert resp.tokens_used == 350
        assert resp.prompt_tokens == 300
        assert resp.completion_tokens == 50
        assert resp.cache_hit_prompt_tokens == 200
        assert resp.cache_miss_prompt_tokens == 100

    def test_deepseek_generate_uses_official_chat_path(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/chat/completions"
            return httpx.Response(200, json=_ok_response())

        p = _provider(
            _make_transport(handler),
            base_url="https://api.deepseek.com",
            model="deepseek-v4-pro",
        )
        p.generate("Say hi")

    def test_system_prompt_included(self):
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            messages = body["messages"]
            assert messages[0] == {"role": "system", "content": "You are helpful."}
            assert messages[1] == {"role": "user", "content": "Hi"}
            return httpx.Response(200, json=_ok_response())

        p = _provider(_make_transport(handler))
        p.generate("Hi", system="You are helpful.")

    def test_no_system_prompt_when_empty(self):
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            messages = body["messages"]
            assert len(messages) == 1
            assert messages[0]["role"] == "user"
            return httpx.Response(200, json=_ok_response())

        p = _provider(_make_transport(handler))
        p.generate("Hi")

    def test_options_merged_into_payload(self):
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["temperature"] == 0.5
            assert body["max_tokens"] == 1024
            assert body["seed"] == 99
            return httpx.Response(200, json=_ok_response())

        p = _provider(
            _make_transport(handler),
            options={"temperature": 0.3, "max_tokens": 1024, "seed": 99},
        )
        # Call-level options override instance options.
        p.generate("test", options={"temperature": 0.5})

    def test_deepseek_no_think_disables_thinking(self):
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["thinking"] == {"type": "disabled"}
            return httpx.Response(200, json=_ok_response())

        p = _provider(
            _make_transport(handler),
            base_url="https://api.deepseek.com",
            model="deepseek-v4-pro",
        )
        p.generate("return json", options={"no_think": True})

    def test_no_think_not_sent_to_generic_openai_endpoint(self):
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert "thinking" not in body
            assert "no_think" not in body
            return httpx.Response(200, json=_ok_response())

        p = _provider(_make_transport(handler), base_url="https://api.example.com")
        p.generate("return json", options={"no_think": True})

    def test_auth_header_present(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.headers["authorization"] == "Bearer sk-my-key"
            return httpx.Response(200, json=_ok_response())

        p = _provider(_make_transport(handler), api_key="sk-my-key")
        p.generate("test")

    def test_invalid_json_response(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"not json", headers={"content-type": "application/json"})

        p = _provider(_make_transport(handler))
        with pytest.raises(LLMResponseError, match="Invalid JSON"):
            p.generate("test")

    def test_unexpected_response_structure(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"choices": []})

        p = _provider(_make_transport(handler))
        with pytest.raises(LLMResponseError, match="Unexpected response structure"):
            p.generate("test")


# ---------------------------------------------------------------------------
# generate_stream() tests
# ---------------------------------------------------------------------------

class TestGenerateStream:
    def test_successful_stream(self):
        chunks = ["Hello", " world", "!"]

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["stream"] is True
            return httpx.Response(200, text=_sse_chunks(chunks))

        p = _provider(_make_transport(handler))
        result = list(p.generate_stream("Say hi"))
        assert result == chunks

    def test_stream_ignores_empty_and_non_data_lines(self):
        raw = "\n\n: comment\ndata: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n"

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=raw)

        p = _provider(_make_transport(handler))
        result = list(p.generate_stream("test"))
        assert result == ["ok"]


# ---------------------------------------------------------------------------
# Error mapping tests
# ---------------------------------------------------------------------------

class TestErrorMapping:
    def test_401_maps_to_connection_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"error": "unauthorized"})

        p = _provider(_make_transport(handler))
        with pytest.raises(LLMConnectionError, match="401"):
            p.generate("test")

    def test_429_maps_to_timeout_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, json={"error": "rate limited"})

        p = _provider(_make_transport(handler))
        with pytest.raises(LLMTimeoutError, match="429"):
            p.generate("test")

    def test_500_maps_to_response_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": "internal"})

        p = _provider(_make_transport(handler))
        with pytest.raises(LLMResponseError, match="500"):
            p.generate("test")


# ---------------------------------------------------------------------------
# Factory / build_llm integration tests
# ---------------------------------------------------------------------------

class TestFactory:
    def test_missing_api_key_raises_configuration_error(self):
        settings = Settings()
        settings.llm.api_key_env = ""
        with pytest.raises(ConfigurationError, match="api_key is required"):
            build_llm(settings)

    def test_api_key_env_resolves_from_environment(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-from-env")
        settings = Settings()
        settings.llm.api_key_env = "DEEPSEEK_API_KEY"
        provider = build_llm(settings)
        assert isinstance(provider, OpenAICompatibleProvider)
        assert provider.api_key == "sk-from-env"

    def test_missing_api_key_env_error_shows_env_name(self):
        settings = Settings()
        settings.llm.api_key_env = "MY_CUSTOM_KEY"
        with pytest.raises(ConfigurationError, match="MY_CUSTOM_KEY"):
            build_llm(settings)

    def test_build_llm_uses_settings_provider(self, monkeypatch):
        """Verify build_llm reads settings.llm.provider (default is now openai)."""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        settings = Settings()
        settings.llm.base_url = "https://api.example.com"
        provider = build_llm(settings)
        assert isinstance(provider, OpenAICompatibleProvider)

    def test_build_llm_explicit_provider_overrides_settings(self):
        settings = Settings()
        # Explicit provider="ollama" should override the default openai provider.
        from story_for_you.llm.ollama import OllamaProvider
        provider = build_llm(settings, provider="ollama")
        assert isinstance(provider, OllamaProvider)
        assert provider.options["num_ctx"] == settings.llm.context_window
        assert provider.options["num_predict"] == settings.llm.max_tokens

    def test_unknown_provider_raises_configuration_error(self):
        settings = Settings()
        with pytest.raises(ConfigurationError, match="Unknown LLM provider"):
            build_llm(settings, provider="nonexistent")


# ---------------------------------------------------------------------------
# Context manager / close
# ---------------------------------------------------------------------------

class TestContextManager:
    def test_context_manager_closes_provider(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_ok_response())

        p = _provider(_make_transport(handler))
        with p:
            resp = p.generate("test")
            assert resp.content == "Hello!"
        # After exiting, client should be closed.
        assert p._client.is_closed
