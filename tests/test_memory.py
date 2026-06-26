import pytest
from unittest.mock import MagicMock

from src.memory import HybridMemory


def test_add_and_get_turn():
    mem = HybridMemory(token_budget=500)
    mem.add_turn("user", "Hello")
    mem.add_turn("assistant", "Hi there")
    history = mem.get_history()
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["content"] == "Hi there"


def test_budget_enforced_with_summarization():
    # Provide a mock LLM that returns a fake summary
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="New summary")
    
    # 10 chars is ~2 tokens. Set budget low so it forces compression.
    mem = HybridMemory(token_budget=2, summary_llm=mock_llm)
    
    # Add multiple turns to exceed budget
    for i in range(5):
        mem.add_turn("user", f"Long question {i} that takes up tokens")
        mem.add_turn("assistant", f"Long answer {i} that takes up tokens")
        
    history = mem.get_history()
    # Given the low budget, it should compress until only 2 turns (1 pair) are left or it gets stuck at 2 turns
    assert len(history) <= 4
    
    # Should have called the LLM
    assert mock_llm.invoke.called
    assert "New summary" in mem.get_summary()


def test_clear_empties_history():
    mem = HybridMemory(token_budget=500)
    mem.add_turn("user", "Hello")
    mem.clear()
    assert len(mem.get_history()) == 0
    assert mem.get_summary() == ""


def test_format_empty_returns_empty_string():
    mem = HybridMemory(token_budget=500)
    assert mem.format_for_prompt() == ""


def test_format_contains_user_and_assistant_labels():
    mem = HybridMemory(token_budget=500)
    mem.add_turn("user", "What is RAG?")
    mem.add_turn("assistant", "Retrieval Augmented Generation")
    formatted = mem.format_for_prompt()
    assert "User:" in formatted
    assert "Assistant:" in formatted


def test_format_preserves_order():
    mem = HybridMemory(token_budget=500)
    mem.add_turn("user", "First")
    mem.add_turn("assistant", "Second")
    mem.add_turn("user", "Third")
    formatted = mem.format_for_prompt()
    assert formatted.index("First") < formatted.index("Second") < formatted.index("Third")
