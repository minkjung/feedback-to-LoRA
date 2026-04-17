"""Shared utilities for scripts: config loading, paths."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_config(path: str | Path = "configs/config.yaml") -> dict:
    config_path = ROOT / path if not Path(path).is_absolute() else Path(path)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(rel: str) -> Path:
    return ROOT / rel
