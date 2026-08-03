"""Retry backoff and per-provider circuit breaking.

Both classes take an injectable clock so the behaviour can be tested without
sleeping through cooldowns.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

__all__ = ["RetryPolicy", "CircuitBreaker", "BreakerState"]

BreakerState = Literal["closed", "open", "half_open"]


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Exponential backoff with full jitter.

    Attributes:
        max_attempts: total tries per provider, including the first.
        base_delay: seconds before the second attempt.
        max_delay: ceiling for any single wait.
        jitter: 0.0 disables randomisation (handy in tests); 1.0 is full jitter.
        respect_retry_after: honour a server-supplied ``Retry-After`` when the
            provider sends one, capped by ``max_delay``.
    """

    max_attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 8.0
    jitter: float = 1.0
    respect_retry_after: bool = True

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.base_delay < 0 or self.max_delay < 0:
            raise ValueError("delays must be non-negative")
        if not 0.0 <= self.jitter <= 1.0:
            raise ValueError("jitter must be within [0.0, 1.0]")

    def delay_for(
        self,
        attempt: int,
        *,
        retry_after: float | None = None,
        rng: random.Random | None = None,
    ) -> float:
        """Seconds to wait before retry number ``attempt`` (0-indexed)."""
        if retry_after is not None and self.respect_retry_after:
            return min(float(retry_after), self.max_delay)
        raw = min(self.base_delay * (2**attempt), self.max_delay)
        if self.jitter <= 0.0:
            return raw
        rand = (rng or random).random()
        # Full jitter: uniform in [raw * (1 - jitter), raw].
        return raw * (1.0 - self.jitter) + raw * self.jitter * rand


@dataclass(slots=True)
class CircuitBreaker:
    """Stops hammering a provider that is clearly down.

    States:
        closed    -- normal operation; consecutive failures accumulate.
        open      -- all calls short-circuit until ``cooldown`` elapses.
        half_open -- exactly one probe is allowed through; success closes the
                     circuit, failure re-opens it for another cooldown.
    """

    failure_threshold: int = 3
    cooldown: float = 30.0
    clock: Callable[[], float] = time.monotonic

    _failures: int = field(default=0, init=False)
    _opened_at: float | None = field(default=None, init=False)
    _probing: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")

    @property
    def state(self) -> BreakerState:
        if self._opened_at is None:
            return "closed"
        if self.clock() - self._opened_at >= self.cooldown:
            return "half_open"
        return "open"

    def allow(self) -> bool:
        """Whether a request may be dispatched right now."""
        state = self.state
        if state == "closed":
            return True
        if state == "open":
            return False
        # half_open: admit a single probe.
        if self._probing:
            return False
        self._probing = True
        return True

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None
        self._probing = False

    def record_failure(self, *, fatal: bool = False) -> None:
        """Register a failure. ``fatal=True`` opens the circuit immediately."""
        self._probing = False
        self._failures += 1
        if fatal or self._failures >= self.failure_threshold:
            self._opened_at = self.clock()

    def reset(self) -> None:
        self.record_success()
