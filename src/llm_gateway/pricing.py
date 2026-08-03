"""Model pricing lookup.

Design note
-----------
Prices change, and a hardcoded table inside library logic goes stale silently
-- which is worse than having no table at all, because the numbers still *look*
authoritative. So this module keeps three properties:

1. Prices live in ``prices.json`` as data, stamped with an ``as_of`` date.
2. An unknown model resolves to ``None``, never to a guessed number. Callers
   see ``Completion.usage.cost_usd is None`` and know the cost is unmeasured.
3. Any price can be overridden per-provider at construction time, so you are
   never blocked waiting for this repo to ship a new model.

Always verify against the provider's own pricing page before you bill anyone.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources

__all__ = ["ModelPrice", "PriceBook"]


@dataclass(frozen=True, slots=True)
class ModelPrice:
    """USD per one million tokens."""

    input_per_mtok: float
    output_per_mtok: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        """Cost in USD for the given token counts."""
        return (
            input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok
        ) / 1_000_000


class PriceBook:
    """Resolves a model identifier to a :class:`ModelPrice`.

    Lookup is exact first, then longest-prefix. Providers append dated or
    regional suffixes to model ids (``claude-sonnet-4-5-20250929``,
    ``gpt-5.6-sol-2026-05-01``); prefix matching means those keep working
    without a table entry each.
    """

    def __init__(
        self,
        prices: Mapping[str, ModelPrice] | None = None,
        *,
        as_of: str | None = None,
    ) -> None:
        self._prices: dict[str, ModelPrice] = dict(prices or {})
        self.as_of = as_of

    # -- construction -----------------------------------------------------

    @classmethod
    def load_default(cls) -> PriceBook:
        """Load the bundled price table."""
        raw = json.loads(
            resources.files("llm_gateway").joinpath("prices.json").read_text("utf-8")
        )
        prices = {
            model: ModelPrice(float(v["input"]), float(v["output"]))
            for model, v in raw["models"].items()
        }
        return cls(prices, as_of=raw.get("as_of"))

    @classmethod
    def empty(cls) -> PriceBook:
        """A price book that knows nothing -- every lookup returns ``None``."""
        return cls({}, as_of=None)

    def with_overrides(self, overrides: Mapping[str, ModelPrice]) -> PriceBook:
        """Return a copy with ``overrides`` merged in (overrides win)."""
        merged = dict(self._prices)
        merged.update(overrides)
        return PriceBook(merged, as_of=self.as_of)

    # -- lookup -----------------------------------------------------------

    def lookup(self, model: str) -> ModelPrice | None:
        """Exact match, else the longest registered prefix of ``model``."""
        if model in self._prices:
            return self._prices[model]
        best_key = ""
        for key in self._prices:
            if model.startswith(key) and len(key) > len(best_key):
                best_key = key
        return self._prices[best_key] if best_key else None

    def cost(self, model: str, input_tokens: int, output_tokens: int) -> float | None:
        """Cost in USD, or ``None`` when the model is not priced."""
        price = self.lookup(model)
        if price is None:
            return None
        return price.cost(input_tokens, output_tokens)

    # -- dunder -----------------------------------------------------------

    def __contains__(self, model: str) -> bool:
        return self.lookup(model) is not None

    def __len__(self) -> int:
        return len(self._prices)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"PriceBook({len(self._prices)} models, as_of={self.as_of!r})"
