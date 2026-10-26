import json

import httpx
import pytest

from story_for_you.exceptions import LLMResponseError
from story_for_you.llm.budget import BudgetedLLMProvider
from story_for_you.llm.ollama import OllamaProvider
from story_for_you.utils.prompting import cache_prompt
from tests.test_openai_compat import _provider
from tests.test_review_fixes import _ResponseLLM


def test_openai_truncated_body_is_never_accepted_as_a_finished_story():
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "写到半句"}, "finish_reason": "length"}
                ],
            },
        )
    )
    with _provider(transport) as provider:
        with pytest.raises(LLMResponseError, match="truncated"):
            provider.generate(cache_prompt("正文"))


def test_ollama_truncation_is_detected():
    provider = OllamaProvider("test")
    provider._client.close()
    provider._client = httpx.Client(
        base_url=provider.base_url,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, json={"response": "半句", "done_reason": "length"}
            ),
        ),
    )
    with provider, pytest.raises(LLMResponseError, match="truncated"):
        provider.generate(cache_prompt("正文"))


def test_prompt_budget_counts_chinese_bytes_and_reserves_output():
    provider = BudgetedLLMProvider(
        _ResponseLLM("unused"), context_window=1000, max_output_tokens=300
    )
    assert provider.generate(cache_prompt("字" * 100)).content == "unused"
    with pytest.raises(LLMResponseError, match="evidence was not truncated"):
        provider.generate(cache_prompt("字" * 200))


def test_budgeted_output_reserve_is_enforced_for_both_provider_protocols():
    provider = OllamaProvider("test")
    try:
        payload = provider._build_payload(
            cache_prompt("正文"), "", False, {"max_tokens": 200}
        )
        assert payload["options"]["num_predict"] == 200
        assert "max_tokens" not in payload["options"]
    finally:
        provider.close()
    seen = []
    transport = httpx.MockTransport(
        lambda request: (
            seen.append(json.loads(request.content))
            or httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": "完整"}, "finish_reason": "stop"}
                    ],
                },
            )
        )
    )
    with _provider(transport) as delegate:
        BudgetedLLMProvider(delegate, 1000, 200).generate(cache_prompt("正文"))
    assert seen[0]["max_tokens"] == 200


def test_streamed_truncation_is_reported():
    data = (
        'data: {"choices":[{"delta":{"content":"半句"},"finish_reason":"length"}]}\n\n'
    )
    with _provider(
        httpx.MockTransport(lambda request: httpx.Response(200, text=data))
    ) as provider:
        with pytest.raises(LLMResponseError, match="truncated"):
            list(provider.generate_stream(cache_prompt("正文")))
