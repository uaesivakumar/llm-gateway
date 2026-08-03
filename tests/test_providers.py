"""Request shaping, response parsing, error classification and retries."""

from __future__ import annotations

import httpx
import pytest
from conftest import NO_WAIT, json_responder, mock_async_client, mock_client

from llm_gateway.errors import (
    AuthenticationError,
    InvalidRequest,
    ProviderTimeout,
    ProviderUnavailable,
    RateLimited,
)
from llm_gateway.policy import RetryPolicy
from llm_gateway.providers import AnthropicProvider, GoogleProvider, OpenAIProvider
from llm_gateway.types import Message

MESSAGES = (
    Message.system("You are terse."),
    Message.user("Hello"),
)

ANTHROPIC_OK = {
    "content": [{"type": "text", "text": "Hi."}],
    "usage": {"input_tokens": 12, "output_tokens": 3},
}
OPENAI_OK = {
    "choices": [{"message": {"role": "assistant", "content": "Hi."}}],
    "usage": {"prompt_tokens": 12, "completion_tokens": 3},
}
GOOGLE_OK = {
    "candidates": [{"content": {"parts": [{"text": "Hi."}]}}],
    "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 3},
}


# -- Anthropic ------------------------------------------------------------


def test_anthropic_hoists_system_and_always_sends_max_tokens():
    provider = AnthropicProvider("claude-sonnet-4-5", api_key="k")
    request = provider.prepare(MESSAGES, {})
    assert request.url.endswith("/v1/messages")
    assert request.json["system"] == "You are terse."
    assert request.json["messages"] == [{"role": "user", "content": "Hello"}]
    assert "max_tokens" in request.json, "the Messages API requires max_tokens"
    assert request.headers["x-api-key"] == "k"
    assert request.headers["anthropic-version"]


def test_anthropic_joins_multiple_system_messages():
    provider = AnthropicProvider("claude-sonnet-4-5", api_key="k")
    messages = (Message.system("A"), Message.system("B"), Message.user("hi"))
    assert provider.prepare(messages, {}).json["system"] == "A\n\nB"


def test_anthropic_parses_text_and_usage():
    provider = AnthropicProvider("claude-sonnet-4-5", api_key="k")
    parsed = provider.parse(ANTHROPIC_OK)
    assert parsed.text == "Hi."
    assert (parsed.input_tokens, parsed.output_tokens) == (12, 3)


def test_anthropic_ignores_non_text_blocks():
    provider = AnthropicProvider("claude-sonnet-4-5", api_key="k")
    payload = {
        "content": [
            {"type": "thinking", "thinking": "hmm"},
            {"type": "text", "text": "answer"},
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    assert provider.parse(payload).text == "answer"


# -- OpenAI ---------------------------------------------------------------


def test_openai_keeps_system_inline_and_maps_max_tokens():
    provider = OpenAIProvider("gpt-5.4", api_key="k")
    request = provider.prepare(MESSAGES, {"max_tokens": 64})
    assert request.url.endswith("/v1/chat/completions")
    assert request.json["messages"][0] == {"role": "system", "content": "You are terse."}
    assert request.json["max_completion_tokens"] == 64
    assert "max_tokens" not in request.json
    assert request.headers["authorization"] == "Bearer k"


def test_openai_legacy_flag_uses_original_field_name():
    provider = OpenAIProvider("local-model", api_key="k", use_legacy_max_tokens=True)
    request = provider.prepare(MESSAGES, {"max_tokens": 64})
    assert request.json["max_tokens"] == 64
    assert "max_completion_tokens" not in request.json


def test_openai_custom_base_url_for_compatible_endpoints():
    provider = OpenAIProvider("llama", api_key="k", base_url="http://localhost:8000/")
    assert provider.prepare(MESSAGES, {}).url == "http://localhost:8000/v1/chat/completions"


def test_openai_parses_text_and_usage():
    parsed = OpenAIProvider("gpt-5.4", api_key="k").parse(OPENAI_OK)
    assert parsed.text == "Hi."
    assert (parsed.input_tokens, parsed.output_tokens) == (12, 3)


def test_openai_handles_null_content():
    payload = {"choices": [{"message": {"content": None}}], "usage": {}}
    assert OpenAIProvider("gpt-5.4", api_key="k").parse(payload).text == ""


# -- Google ---------------------------------------------------------------


def test_google_maps_roles_and_system_instruction():
    provider = GoogleProvider("gemini-2.5-flash", api_key="k")
    messages = (
        Message.system("Be terse."),
        Message.user("Hi"),
        Message.assistant("Hello"),
        Message.user("More"),
    )
    request = provider.prepare(messages, {"max_tokens": 32, "temperature": 0.2})
    assert request.json["systemInstruction"]["parts"][0]["text"] == "Be terse."
    assert [c["role"] for c in request.json["contents"]] == ["user", "model", "user"]
    assert request.json["generationConfig"]["maxOutputTokens"] == 32
    assert request.json["generationConfig"]["temperature"] == 0.2


def test_google_sends_api_key_in_header_never_in_url():
    """Keys in query strings leak into proxy logs and browser history."""
    provider = GoogleProvider("gemini-2.5-flash", api_key="secret-key")
    request = provider.prepare(MESSAGES, {})
    assert "secret-key" not in request.url
    assert request.headers["x-goog-api-key"] == "secret-key"


def test_google_vertex_uses_bearer_token():
    provider = GoogleProvider(
        "gemini-2.5-flash",
        access_token="ya29.token",
        base_url="https://us-central1-aiplatform.googleapis.com",
    )
    request = provider.prepare(MESSAGES, {})
    assert request.headers["authorization"] == "Bearer ya29.token"
    assert request.url.startswith("https://us-central1-aiplatform.googleapis.com")


def test_google_parses_text_and_usage():
    parsed = GoogleProvider("gemini-2.5-flash", api_key="k").parse(GOOGLE_OK)
    assert parsed.text == "Hi."
    assert (parsed.input_tokens, parsed.output_tokens) == (12, 3)


# -- error classification -------------------------------------------------


@pytest.mark.parametrize(
    ("status", "expected", "retryable"),
    [
        (401, AuthenticationError, False),
        (403, AuthenticationError, False),
        (429, RateLimited, True),
        (400, InvalidRequest, False),
        (422, InvalidRequest, False),
        (500, ProviderUnavailable, True),
        (503, ProviderUnavailable, True),
    ],
)
def test_status_codes_map_to_the_right_error(status, expected, retryable):
    provider = AnthropicProvider("claude-sonnet-4-5", api_key="k")
    error = provider.classify(status, {"error": {"message": "nope"}}, {})
    assert isinstance(error, expected)
    assert error.retryable is retryable
    assert "nope" in str(error)


def test_auth_errors_are_fatal_for_the_provider():
    provider = OpenAIProvider("gpt-5.4", api_key="bad")
    assert provider.classify(401, {}, {}).fatal_for_provider is True


def test_rate_limit_captures_retry_after():
    provider = OpenAIProvider("gpt-5.4", api_key="k")
    error = provider.classify(429, {}, {"retry-after": "2.5"})
    assert isinstance(error, RateLimited)
    assert error.retry_after == 2.5


def test_unparseable_retry_after_is_ignored():
    provider = OpenAIProvider("gpt-5.4", api_key="k")
    error = provider.classify(429, {}, {"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"})
    assert error.retry_after is None


# -- transport behaviour --------------------------------------------------


def test_successful_call_returns_normalised_completion():
    provider = AnthropicProvider(
        "claude-sonnet-4-5", api_key="k", client=mock_client(json_responder(ANTHROPIC_OK))
    )
    result = provider.complete(MESSAGES, retry=NO_WAIT)
    assert result.text == "Hi."
    assert result.provider == "anthropic"
    assert result.usage.input_tokens == 12
    assert result.usage.cost_usd is None, "providers do not price; the gateway does"


def test_retries_transient_failures_then_succeeds():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, json={"error": {"message": "overloaded"}})
        return httpx.Response(200, json=ANTHROPIC_OK)

    provider = AnthropicProvider(
        "claude-sonnet-4-5", api_key="k", client=mock_client(handler)
    )
    assert provider.complete(MESSAGES, retry=NO_WAIT).text == "Hi."
    assert calls["n"] == 3


def test_gives_up_after_max_attempts():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, json={"error": {"message": "down"}})

    provider = AnthropicProvider(
        "claude-sonnet-4-5", api_key="k", client=mock_client(handler)
    )
    with pytest.raises(ProviderUnavailable):
        provider.complete(MESSAGES, retry=NO_WAIT)
    assert calls["n"] == NO_WAIT.max_attempts


def test_retry_is_skipped_when_backoff_would_blow_the_deadline():
    """A 1s backoff must not be taken when only 10ms of budget remains."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, json={"error": {"message": "down"}})

    provider = AnthropicProvider(
        "claude-sonnet-4-5", api_key="k", client=mock_client(handler)
    )
    slow_retry = RetryPolicy(max_attempts=5, base_delay=1.0, jitter=0.0)

    with pytest.raises(ProviderUnavailable):
        provider.complete(MESSAGES, retry=slow_retry, time_left=lambda: 0.01)

    assert calls["n"] == 1, "no retry should be attempted without time for it"


def test_retries_proceed_when_the_deadline_is_generous():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, json={"error": {"message": "down"}})

    provider = AnthropicProvider(
        "claude-sonnet-4-5", api_key="k", client=mock_client(handler)
    )
    with pytest.raises(ProviderUnavailable):
        provider.complete(MESSAGES, retry=NO_WAIT, time_left=lambda: 300.0)
    assert calls["n"] == NO_WAIT.max_attempts


def test_non_retryable_errors_fail_fast():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, json={"error": {"message": "bad request"}})

    provider = AnthropicProvider(
        "claude-sonnet-4-5", api_key="k", client=mock_client(handler)
    )
    with pytest.raises(InvalidRequest):
        provider.complete(MESSAGES, retry=NO_WAIT)
    assert calls["n"] == 1, "a malformed request must not be retried"


def test_timeouts_become_provider_timeout():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    provider = AnthropicProvider(
        "claude-sonnet-4-5", api_key="k", client=mock_client(handler)
    )
    with pytest.raises(ProviderTimeout):
        provider.complete(MESSAGES, retry=NO_WAIT)


def test_transport_errors_become_provider_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host", request=request)

    provider = OpenAIProvider("gpt-5.4", api_key="k", client=mock_client(handler))
    with pytest.raises(ProviderUnavailable):
        provider.complete(MESSAGES, retry=NO_WAIT)


def test_malformed_success_body_is_reported_not_crashed():
    provider = AnthropicProvider(
        "claude-sonnet-4-5",
        api_key="k",
        client=mock_client(lambda r: httpx.Response(200, text="<html>oops</html>")),
    )
    with pytest.raises(ProviderUnavailable):
        provider.complete(MESSAGES, retry=NO_WAIT)


async def test_async_path_mirrors_sync():
    provider = OpenAIProvider(
        "gpt-5.4", api_key="k", async_client=mock_async_client(json_responder(OPENAI_OK))
    )
    result = await provider.acomplete(MESSAGES, retry=NO_WAIT)
    assert result.text == "Hi."
    await provider.aclose()


async def test_async_retries_transient_failures():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 2:
            return httpx.Response(500, json={"error": {"message": "boom"}})
        return httpx.Response(200, json=OPENAI_OK)

    provider = OpenAIProvider(
        "gpt-5.4", api_key="k", async_client=mock_async_client(handler)
    )
    assert (await provider.acomplete(MESSAGES, retry=NO_WAIT)).text == "Hi."
    assert calls["n"] == 2
    await provider.aclose()
