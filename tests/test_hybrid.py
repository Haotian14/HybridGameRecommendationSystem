import pandas as pd
import pytest
from conftest import FPS_USER

from hgrs.data import UserHistory
from hgrs.hybrid import AdaptiveHybrid, CascadeHybrid, WeightedHybrid, minmax


def test_cascade_keeps_stage_sizes_and_reports_every_stage(suite, cfg):
    table = suite["pipeline"].explain(FPS_USER)
    assert len(table) == cfg.pipeline.stages[-1][1]
    assert {"tfidf_score", "bert_score", "knn_score", "svd_score"} <= set(table.columns)
    assert table["score"].is_monotonic_decreasing
    pd.testing.assert_series_equal(table["score"], table["svd_score"], check_names=False)


def test_cascade_survivors_come_from_previous_stage(suite):
    cascade = CascadeHybrid([(suite["tfidf"], 30), (suite["bert"], 5)])
    top_tfidf = set(suite["tfidf"].score(FPS_USER).head(30).index)
    assert set(cascade.score(FPS_USER).index) <= top_tfidf


def test_weighted_hybrid_blends_normalised_scores(suite, cfg):
    table = suite["weighted"].explain(FPS_USER)
    pool = suite["svd"].score(FPS_USER).head(cfg.weighted.n_candidates)
    assert len(table) == cfg.weighted.top_k
    assert set(table.index) <= set(pool.index)
    assert table["score"].between(0, sum(cfg.weighted.weights.values()) + 1e-9).all()
    assert table["score"].is_monotonic_decreasing


def test_weighted_with_single_component_reproduces_its_ranking(suite):
    hybrid = WeightedHybrid(suite["svd"], 40, {"knn": (suite["knn"], 1.0)})
    pool = list(suite["svd"].score(FPS_USER).head(40).index)
    assert list(hybrid.score(FPS_USER).index) == list(suite["knn"].score(FPS_USER, pool).index)


def test_adaptive_weight_grows_with_history():
    hybrid = AdaptiveHybrid([object()], object(), shrinkage=10)
    assert hybrid.cf_weight(0) == 0.0
    assert hybrid.cf_weight(10) == pytest.approx(0.5)
    assert hybrid.cf_weight(1) < hybrid.cf_weight(5) < hybrid.cf_weight(50) < 1.0


def test_adaptive_blends_content_and_cf(suite):
    adaptive = suite["adaptive"]
    history = UserHistory.from_liked(["100002", "100010", "100018"])
    table = adaptive.explain_history(history)
    w = table.attrs["cf_weight"]
    assert w == pytest.approx(3 / 13)
    expected = (1 - w) * table["content_score"] + w * table["cf_score"]
    pd.testing.assert_series_equal(table["score"], expected, check_names=False)
    assert not table.empty and table["score"].is_monotonic_decreasing


def test_adaptive_serves_known_users(suite):
    assert len(suite["adaptive"].score(FPS_USER)) > 0


def test_minmax_handles_constant_input():
    assert minmax(pd.Series([2.0, 2.0])).tolist() == [0.0, 0.0]
    assert minmax(pd.Series([1.0, 3.0, 2.0])).tolist() == [0.0, 1.0, 0.5]


@pytest.mark.parametrize("name", ["pipeline", "weighted", "adaptive"])
def test_hybrids_return_empty_for_unknown_user(suite, name):
    assert suite[name].score("nobody").empty


def test_constructors_validate_arguments(suite):
    with pytest.raises(ValueError):
        CascadeHybrid([])
    with pytest.raises(ValueError):
        WeightedHybrid(suite["svd"], 10, {})
    with pytest.raises(ValueError):
        AdaptiveHybrid([], suite["svd"])
