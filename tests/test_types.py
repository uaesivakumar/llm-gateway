"""Prompt normalisation and the public result types."""

from __future__ import annotations

import pytest

from llm_gateway.types import Attempt, Completion, Message, Usage, normalize_messages

# -- normalize_messages ---------------------------------------------------


def test_a_bare_string_becomes_a_single_user_turn():
    assert normalize_messages("hello") == (Message("user", "hello"),)


def test_a_single_message_is_wrapped():
    msg = Message.system("be terse")
    assert normalize_messages(msg) == (msg,)


def test_a_sequence_of_messages_passes_through():
    turns = [Message.system("s"), Message.user("u")]
    assert normalize_messages(turns) == tuple(turns)


def test_dicts_are_accepted_for_openai_style_callers():
    assert normalize_messages([{"role": "user", "content": "hi"}]) == (
        Message("user", "hi"),
    )


def test_messages_and_dicts_can_be_mixed():
    result = normalize_messages([Message.system("s"), {"role": "user", "content": "u"}])
    assert result == (Message("system", "s"), Message("user", "u"))


def test_empty_sequence_is_rejected():
    with pytest.raises(ValueError):
        normalize_messages([])


def test_dict_missing_required_keys_is_rejected():
    with pytest.raises(ValueError):
        normalize_messages([{"role": "user"}])


@pytest.mark.parametrize("bad", [42, None, {"role": "user", "content": "hi"}])
def test_unsupported_prompt_types_are_rejected(bad):
    with pytest.raises((TypeError, ValueError)):
        normalize_messages(bad)


def test_unsupported_item_type_inside_a_sequence_is_rejected():
    with pytest.raises(TypeError):
        normalize_messages([42])


# -- result types ---------------------------------------------------------


def test_usage_totals_tokens():
    assert Usage(100, 50).total_tokens == 150


def test_usage_cost_defaults_to_unknown():
    assert Usage(1, 1).cost_usd is None


def test_failed_over_is_false_for_a_clean_first_try():
    reply = Completion(
        text="ok",
        provider="p",
        model="m",
        usage=Usage(1, 1),
        attempts=(Attempt("p", "m", True, 1.0),),
    )
    assert reply.failed_over is False


def test_failed_over_is_true_when_an_earlier_provider_failed():
    reply = Completion(
        text="ok",
        provider="b",
        model="m2",
        usage=Usage(1, 1),
        attempts=(
            Attempt("a", "m1", False, 1.0, "boom", "ProviderUnavailable"),
            Attempt("b", "m2", True, 1.0),
        ),
    )
    assert reply.failed_over is True


def test_message_helpers_set_the_right_role():
    assert Message.system("x").role == "system"
    assert Message.user("x").role == "user"
    assert Message.assistant("x").role == "assistant"
