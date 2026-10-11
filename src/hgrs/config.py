"""Load the YAML config and resolve data paths against the repository root."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None, root: str | Path | None = None) -> dict[str, Any]:
    """Read a config file. Relative data paths are resolved against ``root`` (default: repo root)."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return resolve_paths(cfg, Path(root) if root else REPO_ROOT)


def resolve_paths(cfg: dict[str, Any], root: Path) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    data = cfg["data"]
    for key in ("interactions", "games"):
        data[key] = str(_resolve(data[key], root))
    data["real_users"] = [str(_resolve(p, root)) for p in data.get("real_users") or []]
    return cfg


def _resolve(p: str | Path, root: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else root / p
