import pandas as pd
import pytest

from hgrs.hybrid import CascadeHybrid, WeightedHybrid, minmax

USER = "11111111"  # synthetic FPS / action user


def test_cascade_keeps_stage_sizes_and_reports_every_stage(fitted, synthetic_cfg):
    table = fitted["pipeline"].explain(USER)
    final_n = synthetic_cfg["pipeline"]["stages"][-1][1]
    assert len(table) == final_n
    assert {"tfidf_score", "bert_score", "knn_score", "svd_score"} <= set(table.columns)
    assert table["score"].is_monotonic_decreasing
    # the final score is the last stage's score
    pd.testing.assert_series_equal(table["score"], table["svd_score"], check_names=False)


def test_cascade_survivors_come_from_previous_stage(fitted):
    tfidf, bert = fitted["tfidf"], fitted["bert"]
    cascade = CascadeHybrid([(tfidf, 30), (bert, 5)])
    top_tfidf = set(tfidf.score(USER).head(30).index)
    assert set(cascade.score(USER).index) <= top_tfidf


def test_weighted_hybrid_blends_normalised_scores(fitted, synthetic_cfg):
    table = fitted["weighted"].explain(USER)
    weights = synthetic_cfg["weighted"]["weights"]
    pool = fitted["svd"].score(USER).head(synthetic_cfg["weighted"]["candidates"][1])
    assert len(table) == synthetic_cfg["weighted"]["top_k"]
    assert set(table.index) <= set(pool.index)
    assert table["score"].between(0, sum(weights.values()) + 1e-9).all()
    assert table["score"].is_monotonic_decreasing


def test_weighted_with_single_component_reproduces_its_ranking(fitted):
    hybrid = WeightedHybrid(fitted["svd"], 40, {"knn": (fitted["knn"], 1.0)})
    expected = fitted["knn"].score(USER, list(fitted["svd"].score(USER).head(40).index))
    assert list(hybrid.score(USER).index) == list(expected.index)


def test_minmax_handles_constant_input():
    assert minmax(pd.Series([2.0, 2.0])).tolist() == [0.0, 0.0]
    assert minmax(pd.Series([1.0, 3.0, 2.0])).tolist() == [0.0, 1.0, 0.5]


def test_hybrids_return_empty_for_unknown_user(fitted):
    for name in ("pipeline", "weighted"):
        assert fitted[name].score("nobody").empty


def test_constructors_validate_arguments(fitted):
    with pytest.raises(ValueError):
        CascadeHybrid([])
    with pytest.raises(ValueError):
        WeightedHybrid(fitted["svd"], 10, {})
