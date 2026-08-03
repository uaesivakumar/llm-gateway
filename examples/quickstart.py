"""Minimal real-world usage.

Requires at least one API key in the environment:

    export ANTHROPIC_API_KEY=...
    export OPENAI_API_KEY=...
    export GEMINI_API_KEY=...

Run:
    python examples/quickstart.py
"""

from __future__ import annotations

import os
import sys

from llm_gateway import (
    AllProvidersFailed,
    AnthropicProvider,
    Gateway,
    GoogleProvider,
    OpenAIProvider,
)


def build_gateway() -> Gateway:
    """Include only the providers we actually have credentials for."""
    providers = []
    if os.getenv("ANTHROPIC_API_KEY"):
        providers.append(AnthropicProvider("claude-sonnet-4-5"))
    if os.getenv("OPENAI_API_KEY"):
        providers.append(OpenAIProvider("gpt-5.4"))
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        providers.append(GoogleProvider("gemini-2.5-flash"))

    if not providers:
        sys.exit("Set ANTHROPIC_API_KEY, OPENAI_API_KEY or GEMINI_API_KEY first.")

    # budget_usd is a pre-flight gate: once spend reaches it, complete() refuses
    # to dispatch rather than aborting a call already in flight.
    return Gateway(providers, budget_usd=1.00)


def main() -> None:
    gateway = build_gateway()
    print(f"providers: {[p.key for p in gateway.providers]}\n")

    try:
        reply = gateway.complete(
            "Explain retrieval-augmented generation in one sentence.",
            max_tokens=200,
        )
    except AllProvidersFailed as exc:
        print("every provider failed:")
        for attempt in exc.attempts:
            print(f"  {attempt.provider}:{attempt.model} -> {attempt.error}")
        raise SystemExit(1) from exc

    print(reply.text.strip())
    print()
    print(f"answered by : {reply.provider}:{reply.model}")
    print(f"tokens      : {reply.usage.input_tokens} in / {reply.usage.output_tokens} out")
    print(
        "cost        : "
        + (f"${reply.usage.cost_usd:.6f}" if reply.usage.cost_usd is not None else "unpriced")
    )

    if reply.failed_over:
        print("\nfailover trail:")
        for attempt in reply.attempts:
            status = "ok" if attempt.ok else f"{attempt.error_type}: {attempt.error}"
            print(f"  {attempt.provider}:{attempt.model}  {attempt.latency_ms:.0f}ms  {status}")

    gateway.close()


if __name__ == "__main__":
    main()
