import pytest

from hgrs.config import load_config
from hgrs.data import load_dataset
from hgrs.factory import build_models
from hgrs.models import GameCatalog
from hgrs.synthetic import make_synthetic

FPS_USER = "11111111"
SPORTS_USER = "22222222"


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic")
    make_synthetic(root)
    return root


@pytest.fixture(scope="session")
def cfg(synthetic_root):
    return load_config(synthetic_root / "config.yaml", root=synthetic_root)


@pytest.fixture(scope="session")
def dataset(cfg):
    return load_dataset(cfg)


@pytest.fixture(scope="session")
def catalog(dataset):
    return GameCatalog(dataset[1])


@pytest.fixture(scope="session")
def suite(cfg, dataset, catalog):
    return build_models(cfg, catalog).fit(dataset[0])
