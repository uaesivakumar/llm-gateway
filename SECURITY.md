# Security Policy

## Reporting a vulnerability

Please report security issues privately via
[GitHub Security Advisories](https://github.com/uaesivakumar/llm-gateway/security/advisories/new),
or by email to **siva@sivakumar.ai**. Please do not open a public issue for a
vulnerability.

Expect an acknowledgement within a few days. If the issue is confirmed, I will agree a
disclosure timeline with you before publishing a fix.

## How this library handles your credentials

Worth knowing, because credential handling is the most likely place for a library like
this to hurt you:

- **API keys are sent in headers, never in URLs.** Query-string credentials leak into
  proxy logs, browser history and error reporters. The Google provider uses the
  `x-goog-api-key` header rather than the `?key=` parameter that provider also accepts.
- **Keys are read from the environment or passed explicitly**, and are held only on the
  provider instance. Nothing is written to disk and nothing is cached globally.
- **Error objects carry response bodies.** `ProviderError.body` holds the provider's error
  payload so you can diagnose failures. Providers do not put credentials there, but if you
  log these objects wholesale, review what your provider returns first.
- **`Completion.raw`** holds the provider's full parsed response. Same caution applies if
  you persist it.

If a key of yours has appeared in a traceback, log or issue report, rotate it. That is
cheaper than establishing whether anyone saw it.

## Supported versions

This project is pre-1.0. Security fixes land on the latest release only.
