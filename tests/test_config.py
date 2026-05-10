import pytest

from src.config import load_config, get_config


@pytest.fixture
def cfg():
    return load_config("configs/base.yaml")


def test_load_config_returns_dict(cfg):
    assert isinstance(cfg, dict)
    assert "llm" in cfg
    assert "features" in cfg


def test_chunk_size_is_integer(cfg):
    assert isinstance(cfg["chunking"]["chunk_size"], int)
    assert cfg["chunking"]["chunk_size"] > 0


def test_all_feature_flags_present(cfg):
    expected_flags = [
        "use_reranking", "use_hybrid_search", "use_multi_query",
        "use_confidence_gating", "use_self_check", "use_security",
        "use_memory", "use_query_routing", "use_streaming",
    ]
    for flag in expected_flags:
        assert flag in cfg["features"], f"Missing feature flag: {flag}"


def test_get_config_is_singleton():
    c1 = get_config()
    c2 = get_config()
    assert c1 is c2
