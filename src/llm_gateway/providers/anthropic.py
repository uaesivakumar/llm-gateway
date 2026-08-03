"""Anthropic Messages API provider."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from ..types import Message
from .base import BaseProvider, ParsedResponse, PreparedRequest

__all__ = ["AnthropicProvider"]

DEFAULT_MAX_TOKENS = 1024


class AnthropicProvider(BaseProvider):
    """Talks to ``POST /v1/messages``.

    The system prompt is hoisted out of the message list into the top-level
    ``system`` field, and ``max_tokens`` is always sent because the API
    requires it.
    """

    name: ClassVar[str] = "anthropic"
    default_base_url: ClassVar[str] = "https://api.anthropic.com"
    api_version: ClassVar[str] = "2023-06-01"

    def __init__(self, model: str, *, api_key: str | None = None, **kwargs: Any) -> None:
        super().__init__(
            model, api_key=api_key or os.getenv("ANTHROPIC_API_KEY"), **kwargs
        )

    def prepare(
        self, messages: Sequence[Message], params: Mapping[str, Any]
    ) -> PreparedRequest:
        split = self._split_system(messages)
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": params.get("max_tokens", DEFAULT_MAX_TOKENS),
            "messages": [{"role": m.role, "content": m.content} for m in split.turns],
        }
        if split.system:
            body["system"] = split.system
        for key in ("temperature", "top_p", "top_k", "stop_sequences", "metadata"):
            if key in params:
                body[key] = params[key]

        headers = {
            "content-type": "application/json",
            "anthropic-version": self.api_version,
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return PreparedRequest(url=f"{self.base_url}/v1/messages", headers=headers, json=body)

    def parse(self, data: Mapping[str, Any]) -> ParsedResponse:
        blocks = data.get("content") or []
        text = "".join(
            block.get("text", "")
            for block in blocks
            if isinstance(block, Mapping) and block.get("type") == "text"
        )
        usage = data.get("usage") or {}
        return ParsedResponse(
            text=text,
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
        )
