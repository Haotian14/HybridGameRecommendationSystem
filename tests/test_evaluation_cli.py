import numpy as np
import pandas as pd
import pytest

from hgrs.cli import main
from hgrs.data import split_cold_start_users
from hgrs.evaluation import Evaluator, precision_recall_ndcg, to_markdown


def test_precision_recall_ndcg():
    p, r, n = precision_recall_ndcg(["a", "x", "b"], {"a", "b", "c", "d"}, k=3)
    assert p == pytest.approx(2 / 3)
    assert r == pytest.approx(2 / 4)
    assert 0 < n < 1
    assert precision_recall_ndcg(["a", "b"], {"a", "b"}, k=2)[2] == pytest.approx(1.0)


def test_evaluate_reports_accuracy_and_beyond(suite, dataset, catalog):
    interactions, _ = dataset
    test = interactions.groupby("user_id").head(3)
    table = Evaluator(interactions, catalog, k=5).evaluate(
        suite.select(["popularity", "random", "tfidf"]), test, n_users=20
    )
    assert list(table.index) == ["popularity", "random", "tfidf"]
    assert {"precision@5", "ndcg_se", "coverage", "diversity", "novelty", "popularity_pct"} <= set(
        table
    )
    assert table.loc["popularity", "popularity_pct"] > table.loc["random", "popularity_pct"]
    assert table.loc["popularity", "coverage"] < table.loc["random", "coverage"]
    assert to_markdown(table).splitlines()[0].startswith("| model | precision@5")


def test_diversity_of_identical_genres_is_zero(dataset, catalog):
    interactions, games = dataset
    evaluator = Evaluator(interactions, catalog)
    same_genre = games.loc[games["genre"] == "Sports", "app_id"].head(3).tolist()
    assert len(same_genre) == 3
    assert evaluator._diversity(same_genre) == pytest.approx(0.0, abs=1e-9)


def test_cold_start_curve(suite, dataset, catalog, cfg):
    interactions, _ = dataset
    _, histories = split_cold_start_users(interactions, n_users=10, min_interactions=15, seed=1)
    table = Evaluator(interactions, catalog).cold_start(
        suite.select(["popularity", "bert", "adaptive"]), histories, sizes=[1, 5]
    )
    assert list(table.columns) == ["n=1", "n=5"]
    assert table.notna().all().all()
    assert (table.to_numpy() >= 0).all() and (table.to_numpy() <= 1).all()


def test_markdown_renders_missing_values():
    table = pd.DataFrame({"x": [1.0, np.nan]}, index=pd.Index(["a", "b"], name="model"))
    assert to_markdown(table).splitlines()[-1] == "| b | – |"


def _run(synthetic_root, tmp_path, *args):
    root = str(synthetic_root)
    common = ["--config", f"{root}/config.yaml", "--root", root, "--cache-dir", str(tmp_path)]
    return main([args[0], *common, *args[1:]])


def test_cli_recommend_known_users(synthetic_root, tmp_path):
    rc = _run(
        synthetic_root,
        tmp_path,
        "recommend",
        "--strategy",
        "pipeline",
        "--top-k",
        "3",
        "--out",
        str(tmp_path),
    )
    assert rc == 0
    out = pd.read_csv(tmp_path / "recommendations_pipeline.csv", dtype={"user_id": str})
    assert set(out["user_id"]) == {"11111111", "22222222"}
    assert out.groupby("user_id")["rank"].max().eq(3).all()
    assert out["name"].notna().all()
    assert list(tmp_path.glob("embeddings_*.npz")), "BERT embeddings should be cached"


def test_cli_recommend_new_user(synthetic_root, tmp_path):
    rc = _run(
        synthetic_root, tmp_path, "recommend", "--likes", "100002", "100010", "--out", str(tmp_path)
    )
    assert rc == 0
    out = pd.read_csv(tmp_path / "recommendations_new_user.csv", dtype={"app_id": str})
    assert len(out) == 10 and not {"100002", "100010"} & set(out["app_id"])


def test_cli_cold_start_writes_markdown(synthetic_root, tmp_path):
    target = tmp_path / "cold.md"
    assert _run(synthetic_root, tmp_path, "cold-start", "--n-users", "5", "--out", str(target)) == 0
    assert "recall@10 (n=1)" in target.read_text(encoding="utf-8")
