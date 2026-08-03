"""Pricing lookup and cost arithmetic."""

from __future__ import annotations

import pytest

from llm_gateway.pricing import ModelPrice, PriceBook


def test_cost_is_per_million_tokens():
    price = ModelPrice(input_per_mtok=3.0, output_per_mtok=15.0)
    # 1M input @ $3 + 1M output @ $15
    assert price.cost(1_000_000, 1_000_000) == pytest.approx(18.0)
    # 1k input + 500 output
    assert price.cost(1_000, 500) == pytest.approx(0.003 + 0.0075)


def test_zero_tokens_cost_nothing():
    assert ModelPrice(3.0, 15.0).cost(0, 0) == 0.0


def test_exact_lookup_wins():
    book = PriceBook({"gpt-5.4": ModelPrice(1.25, 7.5)})
    assert book.lookup("gpt-5.4") == ModelPrice(1.25, 7.5)


def test_longest_prefix_wins():
    book = PriceBook(
        {
            "gpt-5.4": ModelPrice(1.25, 7.5),
            "gpt-5.4-mini": ModelPrice(0.375, 2.25),
        }
    )
    # A dated suffix must resolve to the more specific base model.
    assert book.lookup("gpt-5.4-mini-2026-01-15") == ModelPrice(0.375, 2.25)
    assert book.lookup("gpt-5.4-2026-01-15") == ModelPrice(1.25, 7.5)


def test_unknown_model_returns_none_not_zero():
    """An unpriced model must be reported as unknown, never as free."""
    book = PriceBook({"gpt-5.4": ModelPrice(1.25, 7.5)})
    assert book.lookup("some-private-finetune") is None
    assert book.cost("some-private-finetune", 1_000, 1_000) is None
    assert "some-private-finetune" not in book


def test_overrides_replace_bundled_entries():
    book = PriceBook({"m": ModelPrice(1.0, 1.0)})
    patched = book.with_overrides({"m": ModelPrice(2.0, 2.0)})
    assert book.lookup("m") == ModelPrice(1.0, 1.0)  # original untouched
    assert patched.lookup("m") == ModelPrice(2.0, 2.0)


def test_empty_book_prices_nothing():
    book = PriceBook.empty()
    assert len(book) == 0
    assert book.cost("claude-sonnet-4-5", 100, 100) is None


def test_default_book_loads_and_is_stamped():
    book = PriceBook.load_default()
    assert len(book) > 0
    assert book.as_of, "bundled prices must carry an as_of date"
    # Spot-check one model from each provider family resolves.
    for model in ("claude-sonnet-4-5", "gpt-5.4", "gemini-2.5-flash"):
        assert book.lookup(model) is not None, f"{model} should be priced"


def test_default_book_output_costs_at_least_input():
    """Sanity guard against transposed columns in prices.json."""
    book = PriceBook.load_default()
    for model in ("claude-sonnet-4-5", "gpt-5.4", "gemini-2.5-flash"):
        price = book.lookup(model)
        assert price is not None
        assert price.output_per_mtok >= price.input_per_mtok
