"""
AIRS configuration loader.

Reads YAML config files and provides a dict-like object for the entire system.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = _PROJECT_ROOT / "configs" / "default.yaml"


def load_config(path: Optional[str] = None) -> dict[str, Any]:
    """Load and return the configuration dictionary.

    Parameters
    ----------
    path : str, optional
        Path to a YAML config file.  Falls back to ``configs/default.yaml``.

    Returns
    -------
    dict
        Nested configuration dictionary.
    """
    config_path = Path(path) if path else _DEFAULT_CONFIG
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    return cfg


def merge_cli_overrides(cfg: dict, overrides: dict) -> dict:
    """Merge flat CLI overrides into the nested config.

    Keys use dot-notation, e.g. ``agent.algorithm=ppo``.
    """
    for key, value in overrides.items():
        parts = key.split(".")
        d = cfg
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = value
    return cfg
