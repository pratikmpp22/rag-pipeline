import pytest
from unittest.mock import patch


def test_slash_prefix_is_command():
    assert "/help".startswith("/")
    assert "/status".startswith("/")
    assert "/quit".startswith("/")


def test_plain_text_is_not_command():
    assert not "What is the refund policy?".startswith("/")
    assert not "hello".startswith("/")


def test_quit_raises_system_exit():
    from src.cli import handle_command
    from src.memory import HybridMemory

    memory = HybridMemory(token_budget=5)
    state = {
        "vectorstore": None,
        "bm25_index": None,
        "bm25_chunks": [],
        "rag_chain": None,
    }
    cfg = {"features": {}, "llm": {"model": "test"}, "memory": {"max_turns": 5}}

    with pytest.raises(SystemExit):
        handle_command("/quit", memory, None, None, [], None, None, cfg, state)
