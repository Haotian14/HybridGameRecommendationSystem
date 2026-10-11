import pandas as pd
import pytest
from conftest import SPORTS_USER

from hgrs.data import UserHistory
from hgrs.models import BertRecommender, GameCatalog, LsaEncoder

SINGLE_MODELS = ["popularity", "random", "tfidf", "bert", "knn", "svd"]
HISTORY_MODELS = ["popularity", "random", "tfidf", "bert", "svd"]
SPORTS_GAMES = ["100002", "100010", "100018"]  # synthetic games whose main genre is Sports


@pytest.mark.parametrize("name", SINGLE_MODELS)
def test_scores_are_sorted_and_exclude_played_games(suite, dataset, name):
    interactions, _ = dataset
    played = set(interactions.loc[interactions["user_id"] == SPORTS_USER, "app_id"])
    scores = suite[name].score(SPORTS_USER)
    assert len(scores) > 0
    assert scores.is_monotonic_decreasing
    assert not played & set(scores.index)


@pytest.mark.parametrize("name", SINGLE_MODELS)
def test_candidates_restrict_scoring(suite, name):
    subset = list(suite[name].score(SPORTS_USER).index[:7])
    assert set(suite[name].score(SPORTS_USER, candidates=subset).index) == set(subset)


@pytest.mark.parametrize("name", SINGLE_MODELS)
def test_unknown_user_gets_nothing(suite, name):
    assert suite[name].score("no-such-user").empty


def test_content_models_find_the_users_favourite_genre(suite, dataset):
    interactions, games = dataset
    played = set(interactions.loc[interactions["user_id"] == SPORTS_USER, "app_id"])
    sports = set(games.loc[games["genre"].str.contains("Sports"), "app_id"]) - played
    random_hits = 10 * len(sports) / (len(games) - len(played))
    for name in ("tfidf", "bert"):
        hits = len(set(suite[name].recommend(SPORTS_USER, k=10).index) & sports)
        assert hits >= max(4, 3 * random_hits), f"{name}: {hits}/{len(sports)} sports games"


@pytest.mark.parametrize("name", HISTORY_MODELS)
def test_new_user_can_be_served_from_a_history(suite, name):
    scores = suite[name].score_history(UserHistory.from_liked(SPORTS_GAMES))
    assert len(scores) > 0
    assert not set(SPORTS_GAMES) & set(scores.index)


def test_new_sports_fan_gets_sports_games_from_content(suite, dataset):
    _, games = dataset
    genre = games.set_index("app_id")["genre"]
    top = suite["bert"].recommend_history(UserHistory.from_liked(SPORTS_GAMES), k=5).index
    assert all("Sports" in genre[g] for g in top)


def test_svd_vectorised_scoring_matches_surprise(suite):
    svd = suite["svd"]
    scores = svd.score(SPORTS_USER)
    for app_id in scores.index[:20]:
        assert scores[app_id] == pytest.approx(svd.algo.predict(SPORTS_USER, app_id).est)


def test_svd_fold_in_approximates_trained_user(suite):
    """Folding in a known user's own history should rank games much like their trained factors."""
    svd = suite["svd"]
    trained = svd.score(SPORTS_USER)
    folded = svd.score_history(svd.histories[SPORTS_USER]).reindex(trained.index)
    assert trained.corr(folded, method="spearman") > 0.5


def test_popularity_ranks_by_number_of_recommendations(suite, dataset):
    interactions, _ = dataset
    liked = interactions[interactions["is_recommended"] >= 1]["app_id"].value_counts()
    top = suite["popularity"].score_history(UserHistory.from_liked([])).head(3)
    assert top.tolist() == liked.head(3).tolist()


def test_random_is_reproducible(suite):
    first = suite["random"].score(SPORTS_USER).index
    assert first.equals(suite["random"].score(SPORTS_USER).index)


def test_bert_cosine_does_not_reward_games_with_many_genre_tags():
    """With raw dot products a game's genre score grows with its number of tags."""
    games = pd.DataFrame(
        {
            "app_id": ["liked1", "liked2", "liked3", "pure", "many_tags"],
            "genre": [
                "Sports",
                "Sports",
                "Racing,Sports",
                "Sports",
                "Sports,Racing,Action,RPG,Indie",
            ],
            "About the game": [
                "football league",
                "basketball team",
                "car racing",
                "tennis",
                "shooter",
            ],
        }
    )
    catalog = GameCatalog(games)
    history = UserHistory.from_liked(["liked1", "liked2", "liked3"])
    genre_only = {"content": 0.0, "genre": 1.0, "price": 0.0}

    def top_pick(similarity):
        model = BertRecommender(catalog, LsaEncoder(), genre_only, similarity=similarity)
        return model.fit(history.to_frame().assign(user_id="u")).recommend_history(history, 1)

    assert top_pick("dot").index[0] == "many_tags"
    assert top_pick("cosine").index[0] == "pure"


def test_bert_rejects_unknown_similarity(catalog):
    with pytest.raises(ValueError):
        BertRecommender(catalog, LsaEncoder(), similarity="euclid")
