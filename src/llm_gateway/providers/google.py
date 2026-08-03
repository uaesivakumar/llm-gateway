"""Google Gemini provider (Gemini API and Vertex AI).

Two deployment shapes are supported:

* **Gemini API** -- default. Authenticates with an API key sent in the
  ``x-goog-api-key`` header (never a query string, so keys stay out of logs
  and proxy access records).
* **Vertex AI** -- pass ``base_url`` for your regional endpoint plus
  ``access_token`` for OAuth bearer auth.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from ..types import Message
from .base import BaseProvider, ParsedResponse, PreparedRequest

__all__ = ["GoogleProvider"]

_ROLE_MAP = {"user": "user", "assistant": "model"}


class GoogleProvider(BaseProvider):
    """Talks to ``POST /v1beta/models/{model}:generateContent``."""

    name: ClassVar[str] = "google"
    default_base_url: ClassVar[str] = "https://generativelanguage.googleapis.com"
    api_version: ClassVar[str] = "v1beta"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        access_token: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model,
            api_key=api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
            **kwargs,
        )
        self.access_token = access_token

    def prepare(
        self, messages: Sequence[Message], params: Mapping[str, Any]
    ) -> PreparedRequest:
        split = self._split_system(messages)
        body: dict[str, Any] = {
            "contents": [
                {
                    "role": _ROLE_MAP.get(m.role, "user"),
                    "parts": [{"text": m.content}],
                }
                for m in split.turns
            ]
        }
        if split.system:
            body["systemInstruction"] = {"parts": [{"text": split.system}]}

        generation_config: dict[str, Any] = {}
        if "max_tokens" in params:
            generation_config["maxOutputTokens"] = params["max_tokens"]
        for src, dst in (("temperature", "temperature"), ("top_p", "topP"), ("top_k", "topK")):
            if src in params:
                generation_config[dst] = params[src]
        if generation_config:
            body["generationConfig"] = generation_config

        headers = {"content-type": "application/json"}
        if self.access_token:
            headers["authorization"] = f"Bearer {self.access_token}"
        elif self.api_key:
            headers["x-goog-api-key"] = self.api_key

        url = f"{self.base_url}/{self.api_version}/models/{self.model}:generateContent"
        return PreparedRequest(url=url, headers=headers, json=body)

    def parse(self, data: Mapping[str, Any]) -> ParsedResponse:
        candidates = data.get("candidates") or []
        text = ""
        if candidates and isinstance(candidates[0], Mapping):
            content = candidates[0].get("content") or {}
            parts = content.get("parts") or []
            text = "".join(
                part.get("text", "") for part in parts if isinstance(part, Mapping)
            )
        usage = data.get("usageMetadata") or {}
        return ParsedResponse(
            text=text,
            input_tokens=int(usage.get("promptTokenCount", 0)),
            output_tokens=int(usage.get("candidatesTokenCount", 0)),
        )
