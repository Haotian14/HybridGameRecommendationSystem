import pytest

from hgrs.cli import build_models, fit_base_models
from hgrs.config import load_config
from hgrs.data import load_dataset
from hgrs.models import GameCatalog
from hgrs.synthetic import make_synthetic


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic")
    make_synthetic(root)
    return root


@pytest.fixture(scope="session")
def synthetic_cfg(synthetic_root):
    return load_config(synthetic_root / "config.yaml", root=synthetic_root)


@pytest.fixture(scope="session")
def dataset(synthetic_cfg):
    return load_dataset(synthetic_cfg)


@pytest.fixture(scope="session")
def fitted(synthetic_cfg, dataset):
    interactions, games = dataset
    models = build_models(synthetic_cfg, GameCatalog(games))
    fit_base_models(models, interactions)
    return models
