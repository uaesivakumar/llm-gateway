"""OpenAI Chat Completions provider.

Also works against any OpenAI-compatible endpoint (Azure OpenAI, vLLM,
Together, OpenRouter, Ollama) by passing ``base_url``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from ..types import Message
from .base import BaseProvider, ParsedResponse, PreparedRequest

__all__ = ["OpenAIProvider"]


class OpenAIProvider(BaseProvider):
    """Talks to ``POST /v1/chat/completions``.

    ``max_tokens`` is translated to ``max_completion_tokens``, which current
    models require; pass ``use_legacy_max_tokens=True`` for older or
    third-party endpoints that only accept the original field name.
    """

    name: ClassVar[str] = "openai"
    default_base_url: ClassVar[str] = "https://api.openai.com"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        organization: str | None = None,
        use_legacy_max_tokens: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, api_key=api_key or os.getenv("OPENAI_API_KEY"), **kwargs)
        self.organization = organization
        self.use_legacy_max_tokens = use_legacy_max_tokens

    def prepare(
        self, messages: Sequence[Message], params: Mapping[str, Any]
    ) -> PreparedRequest:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if "max_tokens" in params:
            field = "max_tokens" if self.use_legacy_max_tokens else "max_completion_tokens"
            body[field] = params["max_tokens"]
        for key in ("temperature", "top_p", "stop", "seed", "response_format", "user"):
            if key in params:
                body[key] = params[key]

        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        if self.organization:
            headers["openai-organization"] = self.organization
        return PreparedRequest(
            url=f"{self.base_url}/v1/chat/completions", headers=headers, json=body
        )

    def parse(self, data: Mapping[str, Any]) -> ParsedResponse:
        choices = data.get("choices") or []
        text = ""
        if choices and isinstance(choices[0], Mapping):
            message = choices[0].get("message") or {}
            text = message.get("content") or ""
        usage = data.get("usage") or {}
        return ParsedResponse(
            text=text,
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
        )
