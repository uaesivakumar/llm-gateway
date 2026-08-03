# Changelog

All notable changes to this project are documented here. This project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-08-03

First public release.

### Added

- `Gateway` with ordered or cheapest-first provider selection and automatic failover.
- Providers for Anthropic (Messages API), OpenAI (Chat Completions, plus any
  OpenAI-compatible endpoint via `base_url`) and Google (Gemini API and Vertex AI).
- `RetryPolicy` — exponential backoff with full jitter, honouring `Retry-After`.
- `CircuitBreaker` — per-provider, opens after N consecutive failures, admits a single
  probe after cooldown. Authentication failures open it immediately.
- `Ledger` — per-request cost accounting, aggregation by provider and model, and a
  pre-flight budget gate.
- `PriceBook` — pricing as data (`prices.json`, stamped `as_of`), with exact and
  longest-prefix model matching. Unknown models resolve to `None`, never a guessed `0.0`.
- `deadline_s` — an overall wall-clock budget per `complete()` call, checked before each
  provider and before each retry backoff, so failover cannot stack unboundedly.
- Full async parity: `Gateway.acomplete()` and `BaseProvider.acomplete()`.
- `Completion.attempts` — the complete provider trail on every result, so a fallback is
  never silent.
- `py.typed` marker; the package ships its own type information.

### Notes

- 115 tests, 94% coverage, no network access required to run the suite.
- One runtime dependency: `httpx`.

[0.1.0]: https://github.com/uaesivakumar/llm-gateway/releases/tag/v0.1.0
