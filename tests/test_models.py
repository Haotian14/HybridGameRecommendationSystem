import pytest

from hgrs.synthetic import REAL_USERS

BASE = ["tfidf", "bert", "knn", "svd"]
USER = "22222222"  # synthetic "sports" user


@pytest.mark.parametrize("name", BASE)
def test_scores_are_sorted_and_exclude_played_games(fitted, dataset, name):
    interactions, _ = dataset
    played = set(interactions.loc[interactions["user_id"] == USER, "app_id"])
    scores = fitted[name].score(USER)
    assert len(scores) > 0
    assert scores.is_monotonic_decreasing
    assert not played & set(scores.index)


@pytest.mark.parametrize("name", BASE)
def test_candidates_restrict_scoring(fitted, name):
    full = fitted[name].score(USER)
    subset = list(full.index[:7])
    scores = fitted[name].score(USER, candidates=subset)
    assert set(scores.index) == set(subset)


@pytest.mark.parametrize("name", BASE)
def test_unknown_user_gets_nothing(fitted, name):
    assert fitted[name].score("no-such-user").empty


def test_content_models_find_the_users_favourite_genre(fitted, dataset):
    interactions, games = dataset
    played = set(interactions.loc[interactions["user_id"] == USER, "app_id"])
    sports = set(games.loc[games["genre"].str.contains("Sports"), "app_id"]) - played
    random_hits = 10 * len(sports) / (len(games) - len(played))  # expected hits of a random top 10
    for model in ("tfidf", "bert"):
        top = set(fitted[model].recommend(USER, k=10).index)
        hits = len(top & sports)
        assert hits >= max(4, 3 * random_hits), f"{model}: {hits}/{len(sports)} unplayed sports games in the top 10"


def test_svd_vectorised_prediction_matches_surprise(fitted):
    svd = fitted["svd"]
    scores = svd.score(USER)
    for app_id in scores.index[:20]:
        assert scores[app_id] == pytest.approx(svd.algo.predict(USER, app_id).est)


def test_real_users_are_included(fitted):
    for uid in REAL_USERS:
        assert not fitted["svd"].score(uid).empty
