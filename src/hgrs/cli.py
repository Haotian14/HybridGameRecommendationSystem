"""Command-line interface: ``hgrs recommend``, ``hgrs evaluate`` and ``hgrs demo``."""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

from hgrs.config import REPO_ROOT, load_config
from hgrs.data import load_dataset, split_per_user
from hgrs.evaluation import evaluate, to_markdown
from hgrs.hybrid import CascadeHybrid, WeightedHybrid
from hgrs.models import (
    BertRecommender,
    GameCatalog,
    KNNRecommender,
    Recommender,
    SVDRecommender,
    TfidfRecommender,
    make_encoder,
)


def build_models(cfg: dict, catalog: GameCatalog, cache_dir: Path | None = None) -> dict[str, Recommender]:
    """Create the four base models and the two hybrids that share them (nothing is fitted yet)."""
    content, cf, seed = cfg["content"], cfg["cf"], cfg.get("seed", 42)
    base: dict[str, Recommender] = {
        "tfidf": TfidfRecommender(catalog, content["tfidf_max_features"], content["tfidf_text"]),
        "bert": BertRecommender(catalog, make_encoder(content["encoder"], seed), content["bert_weights"], cache_dir),
        "knn": KNNRecommender(**cf["knn"]),
        "svd": SVDRecommender(**cf["svd"], seed=seed),
    }
    stages = [(base[name], int(n)) for name, n in cfg["pipeline"]["stages"]]
    w = cfg["weighted"]
    cand_name, n_cand = w["candidates"]
    weighted = WeightedHybrid(
        candidate_model=base[cand_name],
        n_candidates=int(n_cand),
        components={name: (base[name], float(weight)) for name, weight in w["weights"].items()},
        top_k=w.get("top_k"),
    )
    return {**base, "pipeline": CascadeHybrid(stages), "weighted": weighted}


def fit_base_models(models: dict[str, Recommender], interactions: pd.DataFrame) -> None:
    for name in ("tfidf", "bert", "knn", "svd"):
        t = time.perf_counter()
        models[name].fit(interactions)
        print(f"  fitted {name:<6} in {time.perf_counter() - t:5.1f}s", file=sys.stderr)


def _default_users(cfg: dict, interactions: pd.DataFrame) -> list[str]:
    real = []
    for p in cfg["data"].get("real_users", []):
        if Path(p).exists():
            ids = pd.read_csv(p, usecols=["user_id"], encoding=cfg["data"].get("real_user_encoding"))["user_id"]
            real += ids.astype(str).unique().tolist()
    return real or [interactions["user_id"].iloc[0]]


def cmd_recommend(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, args.root)
    print("Loading data ...", file=sys.stderr)
    interactions, games = load_dataset(cfg)
    catalog = GameCatalog(games)
    print(
        f"  {len(interactions):,} interactions, {interactions['user_id'].nunique():,} users, {len(catalog):,} games",
        file=sys.stderr,
    )

    models = build_models(cfg, catalog, cache_dir=Path(args.cache_dir) if args.cache_dir else None)
    fit_base_models(models, interactions)

    users = args.users or _default_users(cfg, interactions)
    strategies = ["pipeline", "weighted"] if args.strategy == "both" else [args.strategy]
    names = catalog.games.set_index("app_id")["Name"] if "Name" in catalog.games else None

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for strategy in strategies:
        frames = []
        for user in users:
            table = models[strategy].explain(str(user)).head(args.top_k)
            if table.empty:
                print(f"[{strategy}] user {user}: no history, skipped", file=sys.stderr)
                continue
            table = table.reset_index()
            table.insert(0, "rank", range(1, len(table) + 1))
            table.insert(0, "user_id", str(user))
            if names is not None:
                table.insert(3, "name", table["app_id"].map(names))
            frames.append(table)
        if not frames:
            continue
        result = pd.concat(frames, ignore_index=True)
        path = out_dir / f"recommendations_{strategy}.csv"
        result.to_csv(path, index=False)
        print(f"\n=== {strategy} ===")
        print(result.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print(f"saved -> {path}", file=sys.stderr)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, args.root)
    ev, seed = cfg["evaluation"], cfg.get("seed", 42)
    interactions, games = load_dataset(cfg)
    train, test = split_per_user(interactions, ev["test_size"], seed)
    print(f"Train {len(train):,} / test {len(test):,} interactions", file=sys.stderr)

    models = build_models(cfg, GameCatalog(games), cache_dir=Path(args.cache_dir) if args.cache_dir else None)
    fit_base_models(models, train)
    selected = {m: models[m] for m in args.models}
    metrics = evaluate(selected, test, k=ev["k"], n_users=args.n_users or ev["n_users"], seed=seed)

    md = to_markdown(metrics)
    print(md)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md + "\n", encoding="utf-8")
        print(f"saved -> {args.out}", file=sys.stderr)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from hgrs.synthetic import make_synthetic

    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = make_synthetic(tmp, seed=args.seed)
        print(f"Generated a synthetic dataset in {tmp}\n", file=sys.stderr)
        common = dict(config=str(cfg_path), root=tmp, cache_dir=None)
        cmd_recommend(argparse.Namespace(**common, strategy="both", users=None, top_k=5, out=args.out))
        print()
        all_models = ["tfidf", "bert", "knn", "svd", "pipeline", "weighted"]
        return cmd_evaluate(argparse.Namespace(**common, models=all_models, n_users=None, out=None))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hgrs", description="Hybrid game recommendation system")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
        p.add_argument("--root", default=None, help="directory relative data paths resolve against (default: repo root)")
        p.add_argument("--cache-dir", default=str(REPO_ROOT / "data" / "processed"), help="where BERT embeddings are cached")

    rec = sub.add_parser("recommend", help="top-k games for one or more users")
    add_common(rec)
    rec.add_argument("--strategy", choices=["pipeline", "weighted", "both"], default="both")
    rec.add_argument("--users", nargs="+", help="user ids (default: the real users from the config)")
    rec.add_argument("--top-k", type=int, default=10)
    rec.add_argument("--out", default=str(REPO_ROOT / "results"))
    rec.set_defaults(func=cmd_recommend)

    ev = sub.add_parser("evaluate", help="offline ranking metrics on a per-user hold-out split")
    add_common(ev)
    ev.add_argument("--models", nargs="+", default=["tfidf", "bert", "knn", "svd", "pipeline", "weighted"])
    ev.add_argument("--n-users", type=int, default=None)
    ev.add_argument("--out", default=None, help="write the metrics table as markdown")
    ev.set_defaults(func=cmd_evaluate)

    demo = sub.add_parser("demo", help="run everything end-to-end on a synthetic dataset (no downloads)")
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--out", default=str(Path(tempfile.gettempdir()) / "hgrs-demo"))
    demo.set_defaults(func=cmd_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
