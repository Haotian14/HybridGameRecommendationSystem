import pandas as pd
import pytest

from hgrs.cli import main
from hgrs.evaluation import evaluate, precision_recall_ndcg, to_markdown


def test_precision_recall_ndcg():
    p, r, n = precision_recall_ndcg(["a", "x", "b"], {"a", "b", "c", "d"}, k=3)
    assert p == pytest.approx(2 / 3)
    assert r == pytest.approx(2 / 4)
    assert 0 < n < 1
    assert precision_recall_ndcg(["a", "b"], {"a", "b"}, k=2)[2] == pytest.approx(1.0)


def test_evaluate_produces_one_row_per_model(fitted, dataset):
    interactions, _ = dataset
    test = interactions.groupby("user_id").head(3)
    table = evaluate({"svd": fitted["svd"], "tfidf": fitted["tfidf"]}, test, k=5, n_users=10)
    assert list(table.index) == ["svd", "tfidf"]
    assert table["users"].eq(10).all()
    md = to_markdown(table)
    assert md.splitlines()[0].startswith("| model | precision@5")
    assert "| 10 |" in md


def test_cli_recommend_writes_csv(synthetic_root, tmp_path):
    rc = main(
        [
            "recommend",
            "--config", str(synthetic_root / "config.yaml"),
            "--root", str(synthetic_root),
            "--cache-dir", str(tmp_path / "cache"),
            "--strategy", "pipeline",
            "--top-k", "3",
            "--out", str(tmp_path),
        ]
    )
    assert rc == 0
    out = pd.read_csv(tmp_path / "recommendations_pipeline.csv", dtype={"user_id": str})
    assert set(out["user_id"]) == {"11111111", "22222222"}
    assert out.groupby("user_id")["rank"].max().eq(3).all()
    assert out["name"].notna().all()
    # BERT embeddings are cached for the next run
    assert list((tmp_path / "cache").glob("embeddings_*.npz"))
