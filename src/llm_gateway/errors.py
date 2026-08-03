"""Exception hierarchy for llm-gateway.

The distinction that matters operationally is *retryable* vs *terminal*:

* Retryable errors (429, 5xx, timeouts) are worth trying again on the same
  provider after a backoff.
* Terminal errors (401, 400) will not improve with retries, so the gateway
  fails over to the next provider immediately.

``AuthenticationError`` is special-cased as ``fatal_for_provider``: a bad key
will not fix itself, so the circuit breaker opens immediately rather than
burning the failure budget one request at a time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from .types import Attempt

__all__ = [
    "GatewayError",
    "ProviderError",
    "RateLimited",
    "ProviderUnavailable",
    "ProviderTimeout",
    "AuthenticationError",
    "InvalidRequest",
    "AllProvidersFailed",
    "BudgetExceeded",
]


class GatewayError(Exception):
    """Base class for every error raised by this package."""


class ProviderError(GatewayError):
    """A single provider failed to answer.

    Attributes:
        retryable: whether retrying the same provider could plausibly succeed.
        fatal_for_provider: whether this provider should be considered down
            immediately rather than after the usual failure threshold.
    """

    retryable: bool = False
    fatal_for_provider: bool = False

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        status_code: int | None = None,
        body: Any = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.status_code = status_code
        self.body = body

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        base = super().__str__()
        where = ":".join(p for p in (self.provider, self.model) if p)
        if where and self.status_code:
            return f"[{where} HTTP {self.status_code}] {base}"
        if where:
            return f"[{where}] {base}"
        return base


class RateLimited(ProviderError):
    """HTTP 429. Retryable, and may carry a server-supplied delay."""

    retryable = True

    def __init__(self, *args: Any, retry_after: float | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.retry_after = retry_after


class ProviderUnavailable(ProviderError):
    """HTTP 5xx or a transport-level failure. Retryable."""

    retryable = True


class ProviderTimeout(ProviderUnavailable):
    """The request exceeded the configured timeout."""


class AuthenticationError(ProviderError):
    """HTTP 401/403. Not retryable, and opens the circuit immediately."""

    retryable = False
    fatal_for_provider = True


class InvalidRequest(ProviderError):
    """HTTP 400/422. The request is malformed for this provider."""

    retryable = False


class AllProvidersFailed(GatewayError):
    """Every configured provider was exhausted without a successful response."""

    def __init__(self, attempts: tuple[Attempt, ...]) -> None:
        self.attempts = attempts
        detail = "; ".join(
            f"{a.provider}:{a.model} -> {a.error_type or 'error'}: {a.error}" for a in attempts
        ) or "no providers were eligible"
        super().__init__(f"all providers failed ({len(attempts)} attempts) -- {detail}")


class BudgetExceeded(GatewayError):
    """The ledger's spend limit has been reached.

    Raised *before* a request is dispatched, so no further spend occurs.
    """

    def __init__(self, spent: float, limit: float) -> None:
        self.spent = spent
        self.limit = limit
        super().__init__(
            f"budget exhausted: spent ${spent:.6f} of ${limit:.6f} limit; refusing to dispatch"
        )
