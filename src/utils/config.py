from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must contain a mapping: {path}")
    return data


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    config = load_yaml(path)
    paths_ref = config.get("paths")
    if paths_ref:
        paths_path = Path(paths_ref)
        if not paths_path.is_absolute():
            paths_path = path.parent.parent / paths_path
            if not paths_path.exists():
                paths_path = path.parent / paths_ref
        config["paths_config"] = load_yaml(paths_path)
    return config

