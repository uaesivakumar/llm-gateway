"""Normalised request and response types shared across providers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = [
    "Role",
    "Message",
    "Usage",
    "Attempt",
    "Completion",
    "normalize_messages",
    "PromptLike",
]

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class Message:
    """A single turn in a conversation."""

    role: Role
    content: str

    @staticmethod
    def system(content: str) -> Message:
        return Message("system", content)

    @staticmethod
    def user(content: str) -> Message:
        return Message("user", content)

    @staticmethod
    def assistant(content: str) -> Message:
        return Message("assistant", content)


PromptLike = str | Message | Sequence[Message | Mapping[str, str]]


def normalize_messages(prompt: PromptLike) -> tuple[Message, ...]:
    """Coerce the many convenient input shapes into a tuple of ``Message``.

    Accepts a bare string, a single ``Message``, or a sequence of ``Message``
    objects and/or ``{"role": ..., "content": ...}`` mappings.
    """
    if isinstance(prompt, str):
        return (Message("user", prompt),)
    if isinstance(prompt, Message):
        return (prompt,)
    if isinstance(prompt, (list, tuple)):
        out: list[Message] = []
        for item in prompt:
            if isinstance(item, Message):
                out.append(item)
            elif isinstance(item, Mapping):
                try:
                    role = item["role"]
                    content = item["content"]
                except KeyError as exc:  # pragma: no cover - defensive
                    raise ValueError(
                        "message mappings require 'role' and 'content' keys"
                    ) from exc
                out.append(Message(role, content))  # type: ignore[arg-type]
            else:
                raise TypeError(f"unsupported message type: {type(item)!r}")
        if not out:
            raise ValueError("prompt must contain at least one message")
        return tuple(out)
    raise TypeError(f"unsupported prompt type: {type(prompt)!r}")


@dataclass(frozen=True, slots=True)
class Usage:
    """Token counts and the resolved cost for one successful call.

    ``cost_usd`` is ``None`` when no price is known for the model. The library
    deliberately does not guess: an unpriced model reports ``None`` rather than
    a plausible-looking zero.
    """

    input_tokens: int
    output_tokens: int
    cost_usd: float | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class Attempt:
    """A record of one provider call, successful or not.

    Every ``Completion`` carries the full attempt chain, so a caller can see
    that (say) the primary provider was rate-limited before the secondary
    answered -- without turning on debug logging.
    """

    provider: str
    model: str
    ok: bool
    latency_ms: float
    error: str | None = None
    error_type: str | None = None


@dataclass(frozen=True, slots=True)
class Completion:
    """A normalised, provider-agnostic completion."""

    text: str
    provider: str
    model: str
    usage: Usage
    attempts: tuple[Attempt, ...] = ()
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @property
    def failed_over(self) -> bool:
        """True when at least one provider failed before this one succeeded."""
        return any(not a.ok for a in self.attempts)

    def with_attempts(self, attempts: Iterable[Attempt]) -> Completion:
        return Completion(
            text=self.text,
            provider=self.provider,
            model=self.model,
            usage=self.usage,
            attempts=tuple(attempts),
            raw=self.raw,
        )

    def with_usage(self, usage: Usage) -> Completion:
        return Completion(
            text=self.text,
            provider=self.provider,
            model=self.model,
            usage=usage,
            attempts=self.attempts,
            raw=self.raw,
        )
