"""Retry backoff and circuit-breaker state machine."""

from __future__ import annotations

import random

import pytest
from conftest import FakeClock

from llm_gateway.policy import CircuitBreaker, RetryPolicy

# -- RetryPolicy ----------------------------------------------------------


def test_backoff_is_exponential_without_jitter():
    policy = RetryPolicy(base_delay=0.5, max_delay=8.0, jitter=0.0)
    assert policy.delay_for(0) == 0.5
    assert policy.delay_for(1) == 1.0
    assert policy.delay_for(2) == 2.0


def test_backoff_is_capped_at_max_delay():
    policy = RetryPolicy(base_delay=1.0, max_delay=4.0, jitter=0.0)
    assert policy.delay_for(10) == 4.0


def test_jitter_stays_within_bounds():
    policy = RetryPolicy(base_delay=1.0, max_delay=8.0, jitter=1.0)
    rng = random.Random(0)
    for attempt in range(4):
        ceiling = min(1.0 * (2**attempt), 8.0)
        for _ in range(50):
            assert 0.0 <= policy.delay_for(attempt, rng=rng) <= ceiling


def test_retry_after_overrides_backoff_but_is_capped():
    policy = RetryPolicy(base_delay=0.5, max_delay=8.0, jitter=0.0)
    assert policy.delay_for(0, retry_after=3.0) == 3.0
    assert policy.delay_for(0, retry_after=999.0) == 8.0


def test_retry_after_can_be_ignored():
    policy = RetryPolicy(base_delay=0.5, jitter=0.0, respect_retry_after=False)
    assert policy.delay_for(0, retry_after=3.0) == 0.5


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_attempts": 0},
        {"base_delay": -1.0},
        {"jitter": 1.5},
    ],
)
def test_invalid_policies_are_rejected(kwargs):
    with pytest.raises(ValueError):
        RetryPolicy(**kwargs)


# -- CircuitBreaker -------------------------------------------------------


def test_breaker_starts_closed(clock: FakeClock):
    breaker = CircuitBreaker(clock=clock)
    assert breaker.state == "closed"
    assert breaker.allow() is True


def test_breaker_opens_after_threshold(clock: FakeClock):
    breaker = CircuitBreaker(failure_threshold=3, cooldown=30.0, clock=clock)
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == "closed", "must tolerate failures below the threshold"
    breaker.record_failure()
    assert breaker.state == "open"
    assert breaker.allow() is False


def test_success_resets_the_failure_count(clock: FakeClock):
    breaker = CircuitBreaker(failure_threshold=2, clock=clock)
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    assert breaker.state == "closed"


def test_breaker_half_opens_after_cooldown_and_admits_one_probe(clock: FakeClock):
    breaker = CircuitBreaker(failure_threshold=1, cooldown=30.0, clock=clock)
    breaker.record_failure()
    assert breaker.allow() is False

    clock.advance(29.0)
    assert breaker.state == "open"

    clock.advance(1.0)
    assert breaker.state == "half_open"
    assert breaker.allow() is True, "one probe should be admitted"
    assert breaker.allow() is False, "a second concurrent probe must be refused"


def test_probe_success_closes_the_circuit(clock: FakeClock):
    breaker = CircuitBreaker(failure_threshold=1, cooldown=10.0, clock=clock)
    breaker.record_failure()
    clock.advance(10.0)
    assert breaker.allow() is True
    breaker.record_success()
    assert breaker.state == "closed"
    assert breaker.allow() is True


def test_probe_failure_reopens_for_another_cooldown(clock: FakeClock):
    breaker = CircuitBreaker(failure_threshold=1, cooldown=10.0, clock=clock)
    breaker.record_failure()
    clock.advance(10.0)
    assert breaker.allow() is True
    breaker.record_failure()
    assert breaker.state == "open"
    clock.advance(9.0)
    assert breaker.state == "open"
    clock.advance(1.0)
    assert breaker.state == "half_open"


def test_fatal_failure_opens_immediately(clock: FakeClock):
    """A bad API key should not burn the whole failure budget first."""
    breaker = CircuitBreaker(failure_threshold=5, clock=clock)
    breaker.record_failure(fatal=True)
    assert breaker.state == "open"


def test_invalid_threshold_rejected():
    with pytest.raises(ValueError):
        CircuitBreaker(failure_threshold=0)
