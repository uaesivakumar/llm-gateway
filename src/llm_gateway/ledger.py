"""Per-request cost accounting and budget enforcement."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from .errors import BudgetExceeded
from .types import Completion

__all__ = ["LedgerEntry", "Ledger"]


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    """One recorded call."""

    timestamp: float
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    latency_ms: float
    failed_over: bool = False


class Ledger:
    """Thread-safe record of spend across a gateway's lifetime.

    Budget semantics are deliberately a *pre-flight gate*, not a mid-flight
    abort: :meth:`check_budget` raises before a request is dispatched once the
    limit is reached. A single in-flight call can still overshoot the limit,
    because cancelling a request you have already paid for saves nothing. Cap
    per-call exposure with ``max_tokens`` instead.

    Calls whose model has no known price contribute ``0.0`` to the total and are
    counted in :attr:`unpriced_calls`, so an unpriced model can never silently
    look free.
    """

    def __init__(
        self,
        budget_usd: float | None = None,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if budget_usd is not None and budget_usd < 0:
            raise ValueError("budget_usd must be non-negative")
        self.budget_usd = budget_usd
        self._clock = clock
        self._entries: list[LedgerEntry] = []
        self._total = 0.0
        self._unpriced = 0
        self._lock = threading.Lock()

    # -- recording --------------------------------------------------------

    def record(self, completion: Completion, *, latency_ms: float = 0.0) -> LedgerEntry:
        """Record a successful completion and return the created entry."""
        entry = LedgerEntry(
            timestamp=self._clock(),
            provider=completion.provider,
            model=completion.model,
            input_tokens=completion.usage.input_tokens,
            output_tokens=completion.usage.output_tokens,
            cost_usd=completion.usage.cost_usd,
            latency_ms=latency_ms,
            failed_over=completion.failed_over,
        )
        with self._lock:
            self._entries.append(entry)
            if entry.cost_usd is None:
                self._unpriced += 1
            else:
                self._total += entry.cost_usd
        return entry

    # -- budget -----------------------------------------------------------

    def check_budget(self) -> None:
        """Raise :class:`BudgetExceeded` if the limit has already been reached."""
        if self.budget_usd is None:
            return
        with self._lock:
            spent = self._total
        if spent >= self.budget_usd:
            raise BudgetExceeded(spent, self.budget_usd)

    @property
    def remaining_usd(self) -> float | None:
        """Budget headroom, or ``None`` when no budget is configured."""
        if self.budget_usd is None:
            return None
        with self._lock:
            return max(0.0, self.budget_usd - self._total)

    # -- reporting --------------------------------------------------------

    @property
    def total_cost_usd(self) -> float:
        with self._lock:
            return self._total

    @property
    def unpriced_calls(self) -> int:
        """Number of recorded calls whose model had no known price."""
        with self._lock:
            return self._unpriced

    @property
    def entries(self) -> tuple[LedgerEntry, ...]:
        with self._lock:
            return tuple(self._entries)

    def summary(self) -> dict[str, dict[str, float | int]]:
        """Aggregate spend keyed by ``"provider:model"``."""
        out: dict[str, dict[str, float | int]] = {}
        for entry in self.entries:
            key = f"{entry.provider}:{entry.model}"
            bucket = out.setdefault(
                key,
                {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                    "unpriced_calls": 0,
                },
            )
            bucket["calls"] = int(bucket["calls"]) + 1
            bucket["input_tokens"] = int(bucket["input_tokens"]) + entry.input_tokens
            bucket["output_tokens"] = int(bucket["output_tokens"]) + entry.output_tokens
            if entry.cost_usd is None:
                bucket["unpriced_calls"] = int(bucket["unpriced_calls"]) + 1
            else:
                bucket["cost_usd"] = float(bucket["cost_usd"]) + entry.cost_usd
        return out

    def reset(self) -> None:
        with self._lock:
            self._entries.clear()
            self._total = 0.0
            self._unpriced = 0

    def __len__(self) -> int:
        return len(self.entries)
