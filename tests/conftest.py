"""Shared test fixtures and doubles."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import httpx
import pytest

from llm_gateway.errors import ProviderError
from llm_gateway.policy import RetryPolicy
from llm_gateway.providers.base import BaseProvider, ParsedResponse, PreparedRequest
from llm_gateway.types import Completion, Message, Usage

# A retry policy that never actually waits, for fast deterministic tests.
NO_WAIT = RetryPolicy(max_attempts=3, base_delay=0.0, jitter=0.0)


def mock_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def mock_async_client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def json_responder(
    payload: Mapping[str, Any], status_code: int = 200, headers: Mapping[str, str] | None = None
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json=dict(payload), headers=dict(headers or {}))

    return handler


class FakeClock:
    """A monotonic clock the tests advance by hand."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class ScriptedProvider(BaseProvider):
    """A provider that replays a fixed script, with no HTTP involved.

    Each script item is either an exception to raise or a ``Completion`` to
    return. The last item repeats once the script is exhausted, so a provider
    can be made permanently healthy or permanently broken with one entry.
    """

    name = "scripted"

    def __init__(
        self,
        model: str = "scripted-model",
        script: Sequence[Any] | None = None,
        *,
        name: str | None = None,
        clock: FakeClock | None = None,
        takes: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, api_key="test-key", **kwargs)
        if name is not None:
            self.name = name
        self.script = list(script or [])
        self.calls = 0
        # When given a FakeClock, each call advances it by ``takes`` seconds so
        # deadline behaviour can be tested without real waiting.
        self._clock = clock
        self._takes = takes

    def _tick(self) -> None:
        if self._clock is not None and self._takes:
            self._clock.advance(self._takes)

    def _next(self) -> Any:
        if not self.script:
            raise AssertionError("ScriptedProvider called with an empty script")
        index = min(self.calls, len(self.script) - 1)
        self.calls += 1
        return self.script[index]

    def _resolve(self, item: Any) -> Completion:
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, Completion):
            # A real provider always reports its own identity; mirror that so
            # price lookups in tests key off this provider's model.
            return Completion(
                text=item.text,
                provider=self.name,
                model=self.model,
                usage=item.usage,
                attempts=item.attempts,
                raw=item.raw,
            )
        raise AssertionError(f"unsupported script item: {item!r}")

    def complete(self, messages, **kwargs: Any) -> Completion:  # type: ignore[override]
        item = self._next()
        self._tick()
        return self._resolve(item)

    async def acomplete(self, messages, **kwargs: Any) -> Completion:  # type: ignore[override]
        item = self._next()
        self._tick()
        return self._resolve(item)

    # Abstract members are unused here but must exist.
    def prepare(
        self, messages: Sequence[Message], params: Mapping[str, Any]
    ) -> PreparedRequest:  # pragma: no cover
        return PreparedRequest(url="https://example.invalid", headers={}, json={})

    def parse(self, data: Mapping[str, Any]) -> ParsedResponse:  # pragma: no cover
        return ParsedResponse("", 0, 0)


def completion(
    text: str = "ok",
    *,
    provider: str = "scripted",
    model: str = "scripted-model",
    input_tokens: int = 1_000,
    output_tokens: int = 500,
) -> Completion:
    return Completion(
        text=text,
        provider=provider,
        model=model,
        usage=Usage(input_tokens, output_tokens, None),
    )


def provider_error(kind: type[ProviderError], message: str = "boom") -> ProviderError:
    return kind(message, provider="scripted", model="scripted-model")


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()
