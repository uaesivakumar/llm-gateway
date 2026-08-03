"""Cost accounting and budget gating."""

from __future__ import annotations

import pytest
from conftest import completion

from llm_gateway.errors import BudgetExceeded
from llm_gateway.ledger import Ledger
from llm_gateway.types import Completion, Usage


def priced(cost: float | None, *, provider: str = "anthropic", model: str = "m") -> Completion:
    return Completion(
        text="ok",
        provider=provider,
        model=model,
        usage=Usage(100, 50, cost),
    )


def test_records_accumulate():
    ledger = Ledger()
    ledger.record(priced(0.25))
    ledger.record(priced(0.75))
    assert ledger.total_cost_usd == pytest.approx(1.0)
    assert len(ledger) == 2


def test_unpriced_calls_are_counted_not_treated_as_free():
    ledger = Ledger()
    ledger.record(priced(None))
    ledger.record(priced(0.5))
    assert ledger.total_cost_usd == pytest.approx(0.5)
    assert ledger.unpriced_calls == 1, "an unpriced call must be visible, not silent"


def test_budget_gate_allows_spending_up_to_the_limit():
    ledger = Ledger(budget_usd=1.0)
    ledger.check_budget()  # nothing spent yet
    ledger.record(priced(0.99))
    ledger.check_budget()  # still under


def test_budget_gate_refuses_once_limit_is_reached():
    ledger = Ledger(budget_usd=1.0)
    ledger.record(priced(1.0))
    with pytest.raises(BudgetExceeded) as exc:
        ledger.check_budget()
    assert exc.value.spent == pytest.approx(1.0)
    assert exc.value.limit == pytest.approx(1.0)


def test_no_budget_means_no_gate():
    ledger = Ledger()
    ledger.record(priced(1_000.0))
    ledger.check_budget()
    assert ledger.remaining_usd is None


def test_remaining_never_goes_negative():
    ledger = Ledger(budget_usd=1.0)
    ledger.record(priced(2.5))
    assert ledger.remaining_usd == 0.0


def test_negative_budget_rejected():
    with pytest.raises(ValueError):
        Ledger(budget_usd=-1.0)


def test_summary_groups_by_provider_and_model():
    ledger = Ledger()
    ledger.record(priced(0.10, provider="anthropic", model="claude"))
    ledger.record(priced(0.20, provider="anthropic", model="claude"))
    ledger.record(priced(None, provider="openai", model="gpt"))

    summary = ledger.summary()
    assert summary["anthropic:claude"]["calls"] == 2
    assert summary["anthropic:claude"]["cost_usd"] == pytest.approx(0.30)
    assert summary["anthropic:claude"]["input_tokens"] == 200
    assert summary["openai:gpt"]["unpriced_calls"] == 1
    assert summary["openai:gpt"]["cost_usd"] == 0.0


def test_entry_captures_failover_flag():
    ledger = Ledger()
    base = completion()
    entry = ledger.record(base, latency_ms=12.5)
    assert entry.failed_over is False
    assert entry.latency_ms == 12.5


def test_reset_clears_everything():
    ledger = Ledger(budget_usd=1.0)
    ledger.record(priced(0.5))
    ledger.reset()
    assert ledger.total_cost_usd == 0.0
    assert len(ledger) == 0
    assert ledger.unpriced_calls == 0
