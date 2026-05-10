import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

_config_cache = None


def load_config(path="configs/base.yaml"):
    """Load YAML config file and return dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_config(path="configs/base.yaml"):
    """Return cached config, loading once on first call."""
    global _config_cache
    if _config_cache is None:
        _config_cache = load_config(path)
    return _config_cache
