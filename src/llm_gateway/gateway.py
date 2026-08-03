"""The gateway: ordered failover, circuit breaking, and cost accounting."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Literal

from .errors import AllProvidersFailed, ProviderError
from .ledger import Ledger
from .policy import CircuitBreaker, RetryPolicy
from .pricing import PriceBook
from .providers.base import BaseProvider
from .types import Attempt, Completion, PromptLike, Usage, normalize_messages

__all__ = ["Gateway", "Order"]

Order = Literal["declared", "cheapest"]

# Nominal call shape used to rank providers under ``order="cheapest"``.
_RANK_INPUT_TOKENS = 1_000
_RANK_OUTPUT_TOKENS = 1_000


class Gateway:
    """Routes a prompt across several providers, failing over on error.

    Example::

        gateway = Gateway([
            AnthropicProvider("claude-sonnet-4-5"),
            OpenAIProvider("gpt-5.4"),
        ], budget_usd=5.00)

        reply = gateway.complete("Explain RAG in one sentence.")
        print(reply.text, reply.provider, reply.usage.cost_usd)

    Args:
        providers: tried in order (or cheapest-first when ``order="cheapest"``).
        budget_usd: hard pre-flight spend limit; ``None`` disables budgeting.
        ledger: supply your own to share accounting across gateways.
        retry: per-provider retry policy for retryable failures.
        price_book: defaults to the bundled table; pass ``PriceBook.empty()``
            to opt out of cost estimation entirely.
        failure_threshold: consecutive failures before a provider is cut out.
        cooldown: seconds a cut-out provider stays out before one probe call.
        order: ``"declared"`` respects the list order; ``"cheapest"`` sorts by
            the price of a nominal 1k-in / 1k-out call, unpriced providers last.
        on_attempt: optional callback fired for every attempt, success or not.
            Useful for metrics; exceptions raised inside it are not caught.
    """

    def __init__(
        self,
        providers: Sequence[BaseProvider],
        *,
        budget_usd: float | None = None,
        ledger: Ledger | None = None,
        retry: RetryPolicy | None = None,
        price_book: PriceBook | None = None,
        failure_threshold: int = 3,
        cooldown: float = 30.0,
        order: Order = "declared",
        clock: Callable[[], float] = time.monotonic,
        on_attempt: Callable[[Attempt], None] | None = None,
    ) -> None:
        if not providers:
            raise ValueError("Gateway requires at least one provider")
        if order not in ("declared", "cheapest"):
            raise ValueError(f"unknown order: {order!r}")

        self.providers = tuple(providers)
        # Identity check, not truthiness: an empty Ledger has len() == 0 and is
        # therefore falsy, so `ledger or Ledger(...)` would silently discard a
        # caller-supplied ledger that had not recorded anything yet.
        self.ledger = ledger if ledger is not None else Ledger(budget_usd)
        self.retry = retry if retry is not None else RetryPolicy()
        self.price_book = price_book if price_book is not None else PriceBook.load_default()
        self.order: Order = order
        self.on_attempt = on_attempt
        self._breakers: dict[str, CircuitBreaker] = {
            p.key: CircuitBreaker(
                failure_threshold=failure_threshold, cooldown=cooldown, clock=clock
            )
            for p in self.providers
        }

    # -- introspection ----------------------------------------------------

    def health(self) -> dict[str, str]:
        """Current circuit state per provider, keyed by ``"name:model"``."""
        return {key: breaker.state for key, breaker in self._breakers.items()}

    def price_of(self, provider: BaseProvider) -> float | None:
        """Rank cost of a nominal call, or ``None`` when the model is unpriced."""
        if provider.price is not None:
            return provider.price.cost(_RANK_INPUT_TOKENS, _RANK_OUTPUT_TOKENS)
        return self.price_book.cost(
            provider.model, _RANK_INPUT_TOKENS, _RANK_OUTPUT_TOKENS
        )

    def _ordered(self) -> tuple[BaseProvider, ...]:
        if self.order == "declared":
            return self.providers
        # Unpriced providers sort last but keep their relative declared order.
        def rank(item: tuple[int, BaseProvider]) -> tuple[int, float, int]:
            index, provider = item
            cost = self.price_of(provider)
            return (1, 0.0, index) if cost is None else (0, cost, index)

        return tuple(p for _, p in sorted(enumerate(self.providers), key=rank))

    # -- the call ---------------------------------------------------------

    def complete(self, prompt: PromptLike, **params: Any) -> Completion:
        """Return the first successful completion, or raise.

        Raises:
            BudgetExceeded: the ledger limit was already reached.
            AllProvidersFailed: every eligible provider failed.
        """
        messages = normalize_messages(prompt)
        self.ledger.check_budget()
        attempts: list[Attempt] = []

        for provider in self._ordered():
            breaker = self._breakers[provider.key]
            if not breaker.allow():
                self._record_attempt(
                    attempts,
                    Attempt(
                        provider.name, provider.model, False, 0.0,
                        "circuit open", "CircuitOpen",
                    ),
                )
                continue

            started = time.perf_counter()
            try:
                completion = provider.complete(messages, retry=self.retry, **params)
            except ProviderError as exc:
                self._on_failure(attempts, provider, breaker, exc, started)
                continue

            elapsed = _elapsed_ms(started)
            breaker.record_success()
            self._record_attempt(
                attempts, Attempt(provider.name, provider.model, True, elapsed)
            )
            return self._finalize(provider, completion, attempts, elapsed)

        raise AllProvidersFailed(tuple(attempts))

    async def acomplete(self, prompt: PromptLike, **params: Any) -> Completion:
        """Async twin of :meth:`complete`."""
        messages = normalize_messages(prompt)
        self.ledger.check_budget()
        attempts: list[Attempt] = []

        for provider in self._ordered():
            breaker = self._breakers[provider.key]
            if not breaker.allow():
                self._record_attempt(
                    attempts,
                    Attempt(
                        provider.name, provider.model, False, 0.0,
                        "circuit open", "CircuitOpen",
                    ),
                )
                continue

            started = time.perf_counter()
            try:
                completion = await provider.acomplete(
                    messages, retry=self.retry, **params
                )
            except ProviderError as exc:
                self._on_failure(attempts, provider, breaker, exc, started)
                continue

            elapsed = _elapsed_ms(started)
            breaker.record_success()
            self._record_attempt(
                attempts, Attempt(provider.name, provider.model, True, elapsed)
            )
            return self._finalize(provider, completion, attempts, elapsed)

        raise AllProvidersFailed(tuple(attempts))

    # -- internals --------------------------------------------------------

    def _record_attempt(self, attempts: list[Attempt], attempt: Attempt) -> None:
        attempts.append(attempt)
        if self.on_attempt is not None:
            self.on_attempt(attempt)

    def _on_failure(
        self,
        attempts: list[Attempt],
        provider: BaseProvider,
        breaker: CircuitBreaker,
        exc: ProviderError,
        started: float,
    ) -> None:
        breaker.record_failure(fatal=exc.fatal_for_provider)
        self._record_attempt(
            attempts,
            Attempt(
                provider.name,
                provider.model,
                False,
                _elapsed_ms(started),
                str(exc),
                type(exc).__name__,
            ),
        )

    def _finalize(
        self,
        provider: BaseProvider,
        completion: Completion,
        attempts: Iterable[Attempt],
        latency_ms: float,
    ) -> Completion:
        usage = completion.usage
        if provider.price is not None:
            cost = provider.price.cost(usage.input_tokens, usage.output_tokens)
        else:
            cost = self.price_book.cost(
                completion.model, usage.input_tokens, usage.output_tokens
            )
        finalized = Completion(
            text=completion.text,
            provider=completion.provider,
            model=completion.model,
            usage=Usage(usage.input_tokens, usage.output_tokens, cost),
            attempts=tuple(attempts),
            raw=completion.raw,
        )
        self.ledger.record(finalized, latency_ms=latency_ms)
        return finalized

    # -- lifecycle --------------------------------------------------------

    def close(self) -> None:
        for provider in self.providers:
            provider.close()

    async def aclose(self) -> None:
        for provider in self.providers:
            await provider.aclose()

    def __enter__(self) -> Gateway:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<Gateway providers={[p.key for p in self.providers]} order={self.order!r}>"


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0
