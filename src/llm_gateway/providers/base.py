"""Shared provider machinery.

A concrete provider only has to describe three things:

* :meth:`BaseProvider.prepare` -- how to shape the outbound HTTP request
* :meth:`BaseProvider.parse` -- how to read text and token counts back out
* optionally :meth:`BaseProvider.classify` -- how to map status codes to errors

Everything else (transport, timeouts, retry with backoff, latency measurement,
sync/async parity) lives here once. Adding a fourth provider is ~40 lines.
"""

from __future__ import annotations

import abc
import asyncio
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import httpx

from ..errors import (
    AuthenticationError,
    InvalidRequest,
    ProviderError,
    ProviderTimeout,
    ProviderUnavailable,
    RateLimited,
)
from ..policy import RetryPolicy
from ..pricing import ModelPrice
from ..types import Completion, Message, Usage

__all__ = ["BaseProvider", "PreparedRequest", "ParsedResponse"]


@dataclass(frozen=True, slots=True)
class PreparedRequest:
    """A provider-specific HTTP request, ready to send."""

    url: str
    headers: Mapping[str, str]
    json: Mapping[str, Any]
    method: str = "POST"


@dataclass(frozen=True, slots=True)
class ParsedResponse:
    """The provider-agnostic parts of a successful response body."""

    text: str
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True, slots=True)
class _Split:
    system: str | None
    turns: tuple[Message, ...]


class BaseProvider(abc.ABC):
    """Base class for all providers."""

    name: ClassVar[str] = "base"
    default_base_url: ClassVar[str] = ""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        price: ModelPrice | None = None,
        default_params: Mapping[str, Any] | None = None,
        client: httpx.Client | None = None,
        async_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = (base_url or self.default_base_url).rstrip("/")
        self.timeout = timeout
        self.price = price
        self.default_params: dict[str, Any] = dict(default_params or {})
        self._client = client
        self._async_client = async_client
        self._owns_client = client is None
        self._owns_async_client = async_client is None

    # -- identity ---------------------------------------------------------

    @property
    def key(self) -> str:
        """Stable identifier used for circuit-breaker and health keys."""
        return f"{self.name}:{self.model}"

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} {self.key}>"

    # -- subclass contract ------------------------------------------------

    @abc.abstractmethod
    def prepare(
        self, messages: Sequence[Message], params: Mapping[str, Any]
    ) -> PreparedRequest:
        """Build the outbound request for this provider's API."""

    @abc.abstractmethod
    def parse(self, data: Mapping[str, Any]) -> ParsedResponse:
        """Extract text and token usage from a successful response body."""

    def classify(self, status_code: int, body: Any, headers: Mapping[str, str]) -> ProviderError:
        """Map an HTTP error response onto the exception hierarchy."""
        message = _extract_error_message(body) or f"HTTP {status_code}"
        common = {
            "provider": self.name,
            "model": self.model,
            "status_code": status_code,
            "body": body,
        }
        if status_code in (401, 403):
            return AuthenticationError(message, **common)
        if status_code == 429:
            return RateLimited(
                message, retry_after=_parse_retry_after(headers), **common
            )
        if status_code in (400, 404, 422):
            return InvalidRequest(message, **common)
        if status_code >= 500:
            return ProviderUnavailable(message, **common)
        return ProviderError(message, **common)

    # -- helpers for subclasses -------------------------------------------

    @staticmethod
    def _split_system(messages: Sequence[Message]) -> _Split:
        """Separate leading system prompts from the conversational turns.

        Anthropic and Gemini carry the system prompt outside the message list;
        OpenAI keeps it inline. Multiple system messages are joined with blank
        lines so no instruction is silently dropped.
        """
        systems = [m.content for m in messages if m.role == "system"]
        turns = tuple(m for m in messages if m.role != "system")
        return _Split("\n\n".join(systems) if systems else None, turns)

    def _merged_params(self, params: Mapping[str, Any]) -> dict[str, Any]:
        merged = dict(self.default_params)
        merged.update({k: v for k, v in params.items() if v is not None})
        return merged

    # -- transport --------------------------------------------------------

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    @property
    def async_client(self) -> httpx.AsyncClient:
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(timeout=self.timeout)
        return self._async_client

    def close(self) -> None:
        if self._client is not None and self._owns_client:
            self._client.close()
            self._client = None

    async def aclose(self) -> None:
        if self._async_client is not None and self._owns_async_client:
            await self._async_client.aclose()
            self._async_client = None

    def __enter__(self) -> BaseProvider:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- the call ---------------------------------------------------------

    def complete(
        self,
        messages: Sequence[Message],
        *,
        retry: RetryPolicy | None = None,
        rng: random.Random | None = None,
        **params: Any,
    ) -> Completion:
        """Call the provider, retrying retryable failures with backoff."""
        retry = retry or RetryPolicy()
        request = self.prepare(messages, self._merged_params(params))
        last: ProviderError | None = None

        for attempt in range(retry.max_attempts):
            try:
                response = self.client.request(
                    request.method,
                    request.url,
                    headers=dict(request.headers),
                    json=dict(request.json),
                    timeout=self.timeout,
                )
                return self._handle(response)
            except ProviderError as exc:
                last = exc
            except httpx.TimeoutException as exc:
                last = ProviderTimeout(
                    f"request timed out after {self.timeout}s",
                    provider=self.name,
                    model=self.model,
                )
                last.__cause__ = exc
            except httpx.RequestError as exc:
                last = ProviderUnavailable(
                    f"transport error: {exc}", provider=self.name, model=self.model
                )
                last.__cause__ = exc

            if not last.retryable or attempt == retry.max_attempts - 1:
                raise last
            time.sleep(
                retry.delay_for(
                    attempt, retry_after=getattr(last, "retry_after", None), rng=rng
                )
            )

        raise last  # pragma: no cover - unreachable

    async def acomplete(
        self,
        messages: Sequence[Message],
        *,
        retry: RetryPolicy | None = None,
        rng: random.Random | None = None,
        **params: Any,
    ) -> Completion:
        """Async twin of :meth:`complete`."""
        retry = retry or RetryPolicy()
        request = self.prepare(messages, self._merged_params(params))
        last: ProviderError | None = None

        for attempt in range(retry.max_attempts):
            try:
                response = await self.async_client.request(
                    request.method,
                    request.url,
                    headers=dict(request.headers),
                    json=dict(request.json),
                    timeout=self.timeout,
                )
                return self._handle(response)
            except ProviderError as exc:
                last = exc
            except httpx.TimeoutException as exc:
                last = ProviderTimeout(
                    f"request timed out after {self.timeout}s",
                    provider=self.name,
                    model=self.model,
                )
                last.__cause__ = exc
            except httpx.RequestError as exc:
                last = ProviderUnavailable(
                    f"transport error: {exc}", provider=self.name, model=self.model
                )
                last.__cause__ = exc

            if not last.retryable or attempt == retry.max_attempts - 1:
                raise last
            await asyncio.sleep(
                retry.delay_for(
                    attempt, retry_after=getattr(last, "retry_after", None), rng=rng
                )
            )

        raise last  # pragma: no cover - unreachable

    # -- internals --------------------------------------------------------

    def _handle(self, response: httpx.Response) -> Completion:
        if response.status_code >= 400:
            raise self.classify(
                response.status_code, _safe_json(response), response.headers
            )
        # A 2xx carrying a non-JSON body (an HTML error page from a proxy, say)
        # must fail loudly. Parsing it leniently would hand back an empty
        # completion that looks like a successful, and billable, call.
        try:
            data = response.json()
        except Exception as exc:
            raise ProviderUnavailable(
                "response body was not valid JSON",
                provider=self.name,
                model=self.model,
                status_code=response.status_code,
                body=response.text[:500],
            ) from exc
        if not isinstance(data, Mapping):
            raise ProviderUnavailable(
                "response body was not a JSON object",
                provider=self.name,
                model=self.model,
                status_code=response.status_code,
                body=data,
            )
        try:
            parsed = self.parse(data)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderUnavailable(
                f"could not parse response: {exc}",
                provider=self.name,
                model=self.model,
                status_code=response.status_code,
                body=data,
            ) from exc
        return Completion(
            text=parsed.text,
            provider=self.name,
            model=self.model,
            usage=Usage(parsed.input_tokens, parsed.output_tokens, cost_usd=None),
            raw=data,
        )


# -- module helpers -------------------------------------------------------


def _safe_json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except Exception:
        return {"_raw_text": response.text}


def _parse_retry_after(headers: Mapping[str, str]) -> float | None:
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _extract_error_message(body: Any) -> str | None:
    """Pull a human-readable message out of the many provider error shapes."""
    if isinstance(body, str):
        return body
    if not isinstance(body, Mapping):
        return None
    error = body.get("error")
    if isinstance(error, Mapping):
        message = error.get("message")
        if isinstance(message, str):
            return message
    if isinstance(error, str):
        return error
    message = body.get("message")
    if isinstance(message, str):
        return message
    raw = body.get("_raw_text")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()[:500]
    return None
