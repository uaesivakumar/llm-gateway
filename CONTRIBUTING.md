# Contributing

Thanks for taking a look. This project stays small on purpose, so the bar for new
surface area is "does this belong in every user's dependency tree?"

## Setup

```bash
git clone https://github.com/uaesivakumar/llm-gateway.git
cd llm-gateway
pip install -e ".[dev]"
pytest
ruff check .
```

## Ground rules

**No live API calls in tests.** Everything runs against `httpx.MockTransport`
with injected clocks. The suite must stay deterministic and finish in under a
second — if your test needs `sleep()`, inject a clock instead (see
`tests/conftest.py::FakeClock`).

**No new runtime dependencies** without a strong argument. `httpx` is the only
one, and that is a feature.

**Prices are data, not logic.** Corrections to `src/llm_gateway/prices.json` are
very welcome — please bump `as_of` and link the provider's own pricing page in
the PR. Never add a guessed price; an absent entry correctly reports `None`.

## Adding a provider

Subclass `BaseProvider` and implement `prepare()` and `parse()` (plus
`classify()` if the API uses non-standard status codes). See
`src/llm_gateway/providers/anthropic.py` for the shortest complete example.

Please include tests covering:

- request shaping — how system prompts and `max_tokens` are mapped
- response parsing, including a missing or null content field
- error classification for 401 / 429 / 400 / 5xx
- that credentials go in headers, never in the URL

## Reporting bugs

Please include the provider and model, a minimal reproduction, and the
`Completion.attempts` chain or the `AllProvidersFailed` message if you have one.
Redact your API keys — and if one has appeared in a log or traceback, rotate it.

## Commits and PRs

Small, focused commits with a clear message. If a PR changes behaviour, the
test that would have failed before the change is the most useful part of it.
