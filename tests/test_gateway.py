"""Failover, circuit breaking, ordering, budgets and cost attribution."""

from __future__ import annotations

import pytest
from conftest import (
    NO_WAIT,
    FakeClock,
    ScriptedProvider,
    completion,
    provider_error,
)

from llm_gateway.errors import (
    AllProvidersFailed,
    AuthenticationError,
    BudgetExceeded,
    InvalidRequest,
    ProviderUnavailable,
)
from llm_gateway.gateway import Gateway
from llm_gateway.ledger import Ledger
from llm_gateway.pricing import ModelPrice, PriceBook
from llm_gateway.types import Attempt


def gw(*providers, **kwargs) -> Gateway:
    kwargs.setdefault("retry", NO_WAIT)
    kwargs.setdefault("price_book", PriceBook.empty())
    return Gateway(list(providers), **kwargs)


# -- happy path -----------------------------------------------------------


def test_first_healthy_provider_answers():
    primary = ScriptedProvider("m1", [completion("from primary")], name="primary")
    secondary = ScriptedProvider("m2", [completion("from secondary")], name="secondary")

    result = gw(primary, secondary).complete("hello")

    assert result.text == "from primary"
    assert secondary.calls == 0, "a healthy primary must not touch the fallback"
    assert result.failed_over is False
    assert [a.ok for a in result.attempts] == [True]


def test_accepts_string_message_and_message_list():
    provider = ScriptedProvider("m", [completion()])
    gateway = gw(provider)
    assert gateway.complete("hi").text == "ok"
    assert gateway.complete([{"role": "user", "content": "hi"}]).text == "ok"


def test_empty_provider_list_is_rejected():
    with pytest.raises(ValueError):
        Gateway([])


# -- failover -------------------------------------------------------------


def test_fails_over_when_the_primary_is_unavailable():
    primary = ScriptedProvider(
        "m1", [provider_error(ProviderUnavailable, "503")], name="primary"
    )
    secondary = ScriptedProvider("m2", [completion("rescued")], name="secondary")

    result = gw(primary, secondary).complete("hello")

    assert result.text == "rescued"
    assert result.provider == "secondary"
    assert result.failed_over is True
    assert len(result.attempts) == 2
    assert result.attempts[0].ok is False
    assert result.attempts[0].error_type == "ProviderUnavailable"
    assert result.attempts[1].ok is True


def test_fails_over_on_non_retryable_errors_too():
    """A 400 on one provider may still be valid input for another."""
    primary = ScriptedProvider("m1", [provider_error(InvalidRequest)], name="primary")
    secondary = ScriptedProvider("m2", [completion("ok")], name="secondary")
    assert gw(primary, secondary).complete("hi").text == "ok"


def test_all_providers_failing_raises_with_the_full_trail():
    a = ScriptedProvider("m1", [provider_error(ProviderUnavailable, "a down")])
    b = ScriptedProvider("m2", [provider_error(ProviderUnavailable, "b down")])

    with pytest.raises(AllProvidersFailed) as exc:
        gw(a, b).complete("hello")

    assert len(exc.value.attempts) == 2
    assert all(not a.ok for a in exc.value.attempts)
    assert "a down" in str(exc.value)
    assert "b down" in str(exc.value)


def test_attempts_are_reported_to_the_callback():
    seen: list[Attempt] = []
    a = ScriptedProvider("m1", [provider_error(ProviderUnavailable)])
    b = ScriptedProvider("m2", [completion()])

    gw(a, b, on_attempt=seen.append).complete("hi")

    assert [x.ok for x in seen] == [False, True]


# -- circuit breaking -----------------------------------------------------


def test_circuit_opens_and_the_dead_provider_is_skipped(clock: FakeClock):
    dead = ScriptedProvider("m1", [provider_error(ProviderUnavailable)], name="dead")
    live = ScriptedProvider("m2", [completion()], name="live")
    gateway = gw(dead, live, failure_threshold=2, cooldown=30.0, clock=clock)

    gateway.complete("one")
    gateway.complete("two")
    assert dead.calls == 2
    assert gateway.health()["dead:m1"] == "open"

    result = gateway.complete("three")
    assert dead.calls == 2, "an open circuit must short-circuit before dispatch"
    assert result.attempts[0].error_type == "CircuitOpen"


def test_auth_failure_opens_the_circuit_immediately(clock: FakeClock):
    bad_key = ScriptedProvider("m1", [provider_error(AuthenticationError)], name="bad")
    live = ScriptedProvider("m2", [completion()], name="live")
    gateway = gw(bad_key, live, failure_threshold=5, cooldown=30.0, clock=clock)

    gateway.complete("one")
    assert gateway.health()["bad:m1"] == "open"


def test_circuit_recovers_after_cooldown(clock: FakeClock):
    flaky = ScriptedProvider(
        "m1",
        [provider_error(ProviderUnavailable), completion("recovered")],
        name="flaky",
    )
    live = ScriptedProvider("m2", [completion("fallback")], name="live")
    gateway = gw(flaky, live, failure_threshold=1, cooldown=30.0, clock=clock)

    assert gateway.complete("one").text == "fallback"
    assert gateway.health()["flaky:m1"] == "open"

    clock.advance(30.0)
    assert gateway.complete("two").text == "recovered"
    assert gateway.health()["flaky:m1"] == "closed"


def test_every_provider_circuit_open_raises():
    dead = ScriptedProvider("m1", [provider_error(AuthenticationError)])
    gateway = gw(dead, failure_threshold=1)

    with pytest.raises(AllProvidersFailed):
        gateway.complete("one")
    with pytest.raises(AllProvidersFailed) as exc:
        gateway.complete("two")
    assert exc.value.attempts[0].error_type == "CircuitOpen"


# -- ordering -------------------------------------------------------------


def test_cheapest_order_prefers_the_lower_priced_model():
    pricey = ScriptedProvider(
        "expensive", [completion("expensive")], name="a", price=ModelPrice(10.0, 50.0)
    )
    cheap = ScriptedProvider(
        "cheap", [completion("cheap")], name="b", price=ModelPrice(1.0, 5.0)
    )

    assert gw(pricey, cheap, order="cheapest").complete("hi").text == "cheap"
    assert gw(pricey, cheap).complete("hi").text == "expensive"


def test_unpriced_providers_sort_last_under_cheapest():
    unpriced = ScriptedProvider("u", [completion("unpriced")], name="u")
    priced = ScriptedProvider("p", [completion("priced")], name="p", price=ModelPrice(9.0, 9.0))

    assert gw(unpriced, priced, order="cheapest").complete("hi").text == "priced"


def test_unknown_order_rejected():
    with pytest.raises(ValueError):
        Gateway([ScriptedProvider("m", [completion()])], order="random")  # type: ignore[arg-type]


# -- cost and ledger ------------------------------------------------------


def test_cost_is_resolved_from_the_price_book_and_recorded():
    provider = ScriptedProvider("priced-model", [completion(input_tokens=1_000, output_tokens=500)])
    book = PriceBook({"priced-model": ModelPrice(3.0, 15.0)})
    gateway = Gateway([provider], price_book=book, retry=NO_WAIT)

    result = gateway.complete("hi")

    expected = (1_000 * 3.0 + 500 * 15.0) / 1_000_000
    assert result.usage.cost_usd == pytest.approx(expected)
    assert gateway.ledger.total_cost_usd == pytest.approx(expected)
    assert len(gateway.ledger) == 1


def test_provider_price_override_beats_the_price_book():
    provider = ScriptedProvider(
        "priced-model",
        [completion(input_tokens=1_000_000, output_tokens=0)],
        price=ModelPrice(1.0, 1.0),
    )
    book = PriceBook({"priced-model": ModelPrice(99.0, 99.0)})
    result = Gateway([provider], price_book=book, retry=NO_WAIT).complete("hi")
    assert result.usage.cost_usd == pytest.approx(1.0)


def test_unpriced_model_reports_none_rather_than_zero():
    provider = ScriptedProvider("mystery", [completion()])
    result = gw(provider).complete("hi")
    assert result.usage.cost_usd is None


def test_budget_blocks_dispatch_once_exhausted():
    provider = ScriptedProvider("m", [completion(input_tokens=1_000_000, output_tokens=0)])
    book = PriceBook({"m": ModelPrice(1.0, 1.0)})
    gateway = Gateway([provider], price_book=book, budget_usd=1.0, retry=NO_WAIT)

    gateway.complete("first")  # spends exactly $1.00
    with pytest.raises(BudgetExceeded):
        gateway.complete("second")
    assert provider.calls == 1, "no spend may occur after the budget is exhausted"


def test_supplied_ledger_is_used_even_when_empty():
    """Regression: Ledger defines __len__, so an empty one is falsy."""
    ledger = Ledger(budget_usd=10.0)
    gateway = gw(ScriptedProvider("m", [completion()]), ledger=ledger)
    assert gateway.ledger is ledger


def test_a_shared_ledger_pools_spend_across_gateways():
    book = PriceBook({"m": ModelPrice(1.0, 1.0)})
    ledger = Ledger(budget_usd=1.5)
    make = lambda: ScriptedProvider(  # noqa: E731
        "m", [completion(input_tokens=1_000_000, output_tokens=0)]
    )
    first = Gateway([make()], ledger=ledger, price_book=book, retry=NO_WAIT)
    second = Gateway([make()], ledger=ledger, price_book=book, retry=NO_WAIT)

    first.complete("a")
    second.complete("b")
    with pytest.raises(BudgetExceeded):
        first.complete("c")


def test_failed_calls_are_not_billed():
    a = ScriptedProvider("m1", [provider_error(ProviderUnavailable)])
    b = ScriptedProvider("m2", [completion()])
    gateway = gw(a, b)
    gateway.complete("hi")
    assert len(gateway.ledger) == 1, "only the successful call belongs in the ledger"


# -- deadline -------------------------------------------------------------


def test_deadline_stops_failover_before_the_next_provider(clock: FakeClock):
    """Three slow providers must not stack into an unbounded wait."""
    slow = ScriptedProvider(
        "m1", [provider_error(ProviderUnavailable)], name="slow",
        clock=clock, takes=9.0,
    )
    never = ScriptedProvider("m2", [completion("too late")], name="never")

    gateway = gw(slow, never, deadline_s=5.0, clock=clock)

    with pytest.raises(AllProvidersFailed) as exc:
        gateway.complete("hi")

    assert never.calls == 0, "the deadline must stop failover, not just warn"
    assert exc.value.attempts[-1].error_type == "DeadlineExceeded"


def test_deadline_allows_providers_that_fit(clock: FakeClock):
    quick = ScriptedProvider(
        "m1", [provider_error(ProviderUnavailable)], name="quick",
        clock=clock, takes=1.0,
    )
    backup = ScriptedProvider("m2", [completion("in time")], name="backup")

    assert gw(quick, backup, deadline_s=5.0, clock=clock).complete("hi").text == "in time"


def test_no_deadline_means_no_time_limit(clock: FakeClock):
    slow = ScriptedProvider(
        "m1", [provider_error(ProviderUnavailable)], name="slow",
        clock=clock, takes=600.0,
    )
    backup = ScriptedProvider("m2", [completion("eventually")], name="backup")

    assert gw(slow, backup, clock=clock).complete("hi").text == "eventually"


def test_deadline_is_per_call_not_per_gateway(clock: FakeClock):
    """The budget resets on each complete(), it is not a lifetime allowance."""
    provider = ScriptedProvider(
        "m", [completion()], name="p", clock=clock, takes=4.0
    )
    gateway = gw(provider, deadline_s=5.0, clock=clock)
    gateway.complete("first")
    gateway.complete("second")  # would fail if the budget were cumulative
    assert provider.calls == 2


@pytest.mark.parametrize("bad", [0, -1.0])
def test_invalid_deadline_rejected(bad):
    with pytest.raises(ValueError):
        Gateway([ScriptedProvider("m", [completion()])], deadline_s=bad)


# -- async ----------------------------------------------------------------


async def test_async_failover_matches_sync_behaviour():
    primary = ScriptedProvider("m1", [provider_error(ProviderUnavailable)], name="primary")
    secondary = ScriptedProvider("m2", [completion("rescued")], name="secondary")

    result = await gw(primary, secondary).acomplete("hello")

    assert result.text == "rescued"
    assert result.failed_over is True


async def test_async_all_failing_raises():
    a = ScriptedProvider("m1", [provider_error(ProviderUnavailable)])
    with pytest.raises(AllProvidersFailed):
        await gw(a).acomplete("hi")


async def test_async_respects_the_budget():
    provider = ScriptedProvider("m", [completion(input_tokens=1_000_000, output_tokens=0)])
    book = PriceBook({"m": ModelPrice(1.0, 1.0)})
    gateway = Gateway([provider], price_book=book, budget_usd=0.5, retry=NO_WAIT)
    await gateway.acomplete("first")
    with pytest.raises(BudgetExceeded):
        await gateway.acomplete("second")
