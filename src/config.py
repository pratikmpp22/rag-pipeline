import yaml
import copy
from pathlib import Path

_config_cache: dict[str, dict] = {}


def load_config(path: str = "configs/base.yaml") -> dict:
    """Load and validate YAML config file from disk without caching."""
    resolved = str(Path(path).resolve())
    with open(resolved, "r") as f:
        cfg = yaml.safe_load(f)
    _validate(cfg)
    return cfg


def get_config(path: str = "configs/base.yaml") -> dict:
    """Return cached config for this path. Load on first access."""
    resolved = str(Path(path).resolve())
    if resolved not in _config_cache:
        _config_cache[resolved] = load_config(path)
    return copy.deepcopy(_config_cache[resolved])


def reload_config(path: str = "configs/base.yaml") -> dict:
    """Force re-read from disk, update cache, and return the new config."""
    resolved = str(Path(path).resolve())
    _config_cache[resolved] = load_config(path)
    return copy.deepcopy(_config_cache[resolved])


def reset_config():
    """Clear all cached configs. Useful for testing."""
    _config_cache.clear()


def _validate(cfg: dict):
    """Fail fast on invalid config values."""
    if "chunking" in cfg:
        assert cfg["chunking"].get("chunk_size", 0) > 0, "chunk_size must be positive"
    if "retrieval" in cfg:
        assert cfg["retrieval"].get("top_k", 0) > 0, "top_k must be positive"
        assert cfg["retrieval"].get("top_n", 0) > 0, "top_n must be positive"
