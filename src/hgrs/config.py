"""Typed configuration, loaded from YAML.

Every section maps to a dataclass, so a typo in the YAML fails loudly at load time
instead of surfacing as a ``KeyError`` deep inside a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


@dataclass
class DataConfig:
    interactions: Path
    games: Path
    real_users: list[Path] = field(default_factory=list)
    real_user_encoding: str | None = None


@dataclass
class ContentConfig:
    tfidf_max_features: int = 5000
    tfidf_text: str = "combined"
    encoder: str = "all-MiniLM-L6-v2"
    # "cosine", or "dot" to reproduce the original notebooks (see BertRecommender)
    bert_similarity: str = "cosine"
    bert_weights: dict[str, float] = field(
        default_factory=lambda: {"content": 0.7, "genre": 0.2, "price": 0.1}
    )


@dataclass
class KNNConfig:
    k: int = 40
    sim_name: str = "msd"
    user_based: bool = True


@dataclass
class SVDConfig:
    n_factors: int = 100
    reg_all: float = 0.02
    fold_in_reg: float = 1.0


@dataclass
class PipelineConfig:
    stages: list[tuple[str, int]] = field(
        default_factory=lambda: [("tfidf", 500), ("bert", 300), ("knn", 100), ("svd", 10)]
    )


@dataclass
class WeightedConfig:
    candidate_model: str = "svd"
    n_candidates: int = 500
    weights: dict[str, float] = field(
        default_factory=lambda: {"knn": 0.5, "tfidf": 0.25, "bert": 0.25}
    )
    top_k: int = 10


@dataclass
class AdaptiveConfig:
    content_models: list[str] = field(default_factory=lambda: ["tfidf", "bert"])
    cf_model: str = "svd"
    shrinkage: float = 10.0


@dataclass
class EvaluationConfig:
    test_size: float = 0.2
    k: int = 10
    n_users: int = 200
    cold_start_users: int = 200
    cold_start_sizes: list[int] = field(default_factory=lambda: [1, 3, 5, 10, 20])


@dataclass
class Config:
    data: DataConfig
    seed: int = 42
    content: ContentConfig = field(default_factory=ContentConfig)
    knn: KNNConfig = field(default_factory=KNNConfig)
    svd: SVDConfig = field(default_factory=SVDConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    weighted: WeightedConfig = field(default_factory=WeightedConfig)
    adaptive: AdaptiveConfig = field(default_factory=AdaptiveConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any], root: Path = REPO_ROOT) -> Config:
        """Build a config from parsed YAML; relative data paths resolve against ``root``."""
        data = dict(raw["data"])
        data["interactions"] = _resolve(data["interactions"], root)
        data["games"] = _resolve(data["games"], root)
        data["real_users"] = [_resolve(p, root) for p in data.get("real_users") or []]

        pipeline = dict(raw.get("pipeline", {}))
        if "stages" in pipeline:
            pipeline["stages"] = [(str(name), int(n)) for name, n in pipeline["stages"]]

        return cls(
            data=DataConfig(**data),
            seed=raw.get("seed", 42),
            content=ContentConfig(**raw.get("content", {})),
            knn=KNNConfig(**raw.get("knn", {})),
            svd=SVDConfig(**raw.get("svd", {})),
            pipeline=PipelineConfig(**pipeline),
            weighted=WeightedConfig(**raw.get("weighted", {})),
            adaptive=AdaptiveConfig(**raw.get("adaptive", {})),
            evaluation=EvaluationConfig(**raw.get("evaluation", {})),
        )


def load_config(path: str | Path | None = None, root: str | Path | None = None) -> Config:
    """Read a YAML config (default: ``configs/default.yaml``)."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config.from_dict(raw, Path(root) if root else REPO_ROOT)


def _resolve(p: str | Path, root: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else root / p
