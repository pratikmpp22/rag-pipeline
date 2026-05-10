import pytest

from src.memory import ConversationMemory


def test_add_and_get_turn():
    mem = ConversationMemory(max_turns=5)
    mem.add_turn("user", "Hello")
    mem.add_turn("assistant", "Hi there")
    history = mem.get_history()
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["content"] == "Hi there"


def test_max_turns_enforced():
    mem = ConversationMemory(max_turns=2)
    for i in range(5):
        mem.add_turn("user", f"Q{i}")
        mem.add_turn("assistant", f"A{i}")
    history = mem.get_history()
    # max_turns=2 means max 4 entries (2 pairs)
    assert len(history) <= 4


def test_clear_empties_history():
    mem = ConversationMemory(max_turns=5)
    mem.add_turn("user", "Hello")
    mem.clear()
    assert len(mem.get_history()) == 0


def test_format_empty_returns_empty_string():
    mem = ConversationMemory(max_turns=5)
    assert mem.format_for_prompt() == ""


def test_format_contains_user_and_assistant_labels():
    mem = ConversationMemory(max_turns=5)
    mem.add_turn("user", "What is RAG?")
    mem.add_turn("assistant", "Retrieval Augmented Generation")
    formatted = mem.format_for_prompt()
    assert "User:" in formatted
    assert "Assistant:" in formatted


def test_format_preserves_order():
    mem = ConversationMemory(max_turns=5)
    mem.add_turn("user", "First")
    mem.add_turn("assistant", "Second")
    mem.add_turn("user", "Third")
    formatted = mem.format_for_prompt()
    assert formatted.index("First") < formatted.index("Second") < formatted.index("Third")
