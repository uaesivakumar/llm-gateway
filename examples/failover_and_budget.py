"""Failover, circuit breaking and budgets -- with no API keys and no network.

Every provider here is wired to an httpx.MockTransport, so this runs anywhere
and demonstrates the behaviour you would otherwise have to wait for an outage
to see.

Run:
    python examples/failover_and_budget.py
"""

from __future__ import annotations

import httpx

from llm_gateway import (
    AnthropicProvider,
    BudgetExceeded,
    Gateway,
    ModelPrice,
    OpenAIProvider,
    RetryPolicy,
)

ANTHROPIC_BODY = {
    "content": [{"type": "text", "text": "A short answer from Anthropic."}],
    "usage": {"input_tokens": 1_000, "output_tokens": 500},
}
OPENAI_BODY = {
    "choices": [{"message": {"content": "A short answer from OpenAI."}}],
    "usage": {"prompt_tokens": 1_000, "completion_tokens": 500},
}


def always(status: int, payload: dict) -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(status, json=payload))
    )


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def main() -> None:
    # The primary is hard down; the fallback is healthy.
    primary = AnthropicProvider(
        "claude-sonnet-4-5",
        api_key="not-used",
        client=always(503, {"error": {"message": "overloaded_error"}}),
        price=ModelPrice(3.0, 15.0),
    )
    fallback = OpenAIProvider(
        "gpt-5.4",
        api_key="not-used",
        client=always(200, OPENAI_BODY),
        price=ModelPrice(1.25, 7.5),
    )

    gateway = Gateway(
        [primary, fallback],
        budget_usd=0.05,
        # Two quick attempts so the demo does not idle; production defaults are gentler.
        retry=RetryPolicy(max_attempts=2, base_delay=0.05, jitter=0.0),
        failure_threshold=2,
        cooldown=30.0,
    )

    rule("1. Failover")
    reply = gateway.complete("hello")
    print(f"answered by : {reply.provider}:{reply.model}")
    print(f"failed over : {reply.failed_over}")
    for attempt in reply.attempts:
        status = "ok" if attempt.ok else f"{attempt.error_type}: {attempt.error}"
        print(f"  {attempt.provider}:{attempt.model}  {status}")

    rule("2. Circuit breaking")
    gateway.complete("again")  # second failure trips the primary's breaker
    print(f"health: {gateway.health()}")
    reply = gateway.complete("third time")
    print(f"first attempt now: {reply.attempts[0].error_type} (no HTTP call was made)")

    rule("3. Cost tracking")
    for key, stats in gateway.ledger.summary().items():
        print(
            f"{key}: {stats['calls']} calls, "
            f"{stats['input_tokens']} in / {stats['output_tokens']} out, "
            f"${stats['cost_usd']:.6f}"
        )
    print(f"total     : ${gateway.ledger.total_cost_usd:.6f}")
    print(f"remaining : ${gateway.ledger.remaining_usd:.6f}")

    rule("4. Budget gate")
    try:
        while True:
            gateway.complete("keep going until the budget stops us")
    except BudgetExceeded as exc:
        print(f"stopped: spent ${exc.spent:.6f} against a ${exc.limit:.2f} limit")
        print("no request was dispatched for the call that would have exceeded it")

    gateway.close()


if __name__ == "__main__":
    main()
