"""Command-line interface.

hgrs prepare      rebuild the data files from the public source datasets
hgrs recommend    top-k games for known users, or for a new user via --likes
hgrs evaluate     accuracy and beyond-accuracy metrics for every model
hgrs cold-start   how each model copes when a user has revealed only n games
hgrs demo         all of the above on a synthetic dataset, no downloads needed
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import pandas as pd

from hgrs.config import REPO_ROOT, Config, load_config
from hgrs.data import UserHistory, load_dataset, split_cold_start_users, split_per_user
from hgrs.evaluation import Evaluator, to_markdown
from hgrs.factory import ALL_MODELS, COLD_START_MODELS, ModelSuite, build_models
from hgrs.models import GameCatalog

STRATEGIES = ("pipeline", "weighted", "adaptive")


def log(message: str) -> None:
    print(message, file=sys.stderr)


def _load(args: argparse.Namespace) -> tuple[Config, pd.DataFrame, GameCatalog, ModelSuite]:
    cfg = load_config(args.config, args.root)
    log("Loading data ...")
    interactions, games = load_dataset(cfg)
    catalog = GameCatalog(games)
    log(
        f"  {len(interactions):,} interactions · {interactions['user_id'].nunique():,} users"
        f" · {len(catalog):,} games"
    )
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    return cfg, interactions, catalog, build_models(cfg, catalog, cache_dir)


def _present(table: pd.DataFrame, catalog: GameCatalog, top_k: int) -> pd.DataFrame:
    """Turn an ``explain`` table into a numbered list with game names."""
    table = table.head(top_k).reset_index()
    table.insert(0, "rank", range(1, len(table) + 1))
    names = catalog.names()
    if names is not None:
        table.insert(2, "name", table["app_id"].map(names))
    return table


def _print_table(title: str, table: pd.DataFrame) -> None:
    print(f"\n=== {title} ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


# --------------------------------------------------------------------------- commands


def cmd_recommend(args: argparse.Namespace) -> int:
    cfg, interactions, catalog, suite = _load(args)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.likes:
        return _recommend_for_new_user(args, cfg, interactions, catalog, suite)

    suite.fit(interactions)
    users = args.users or _profile_user_ids(cfg) or [interactions["user_id"].iloc[0]]
    strategies = STRATEGIES if args.strategy == "all" else (args.strategy,)
    for strategy in strategies:
        frames = []
        for user in users:
            table = suite[strategy].explain(str(user))
            if table.empty:
                log(f"[{strategy}] user {user}: no history, skipped")
                continue
            frames.append(_present(table, catalog, args.top_k).assign(user_id=str(user)))
        if frames:
            result = pd.concat(frames, ignore_index=True)
            result = result[["user_id", *result.columns.drop("user_id")]]
            _print_table(strategy, result)
            result.to_csv(out_dir / f"recommendations_{strategy}.csv", index=False)
    return 0


def _recommend_for_new_user(args, cfg, interactions, catalog, suite) -> int:
    """A visitor who is not in the data at all, described only by games they like."""
    suite.fit(interactions, only=[*cfg.adaptive.content_models, cfg.adaptive.cf_model])
    history = UserHistory.from_liked(args.likes)
    adaptive = suite["adaptive"]
    table = adaptive.explain_history(history)
    if table.empty:
        log("None of the --likes games are in the catalogue.")
        return 1
    weight = adaptive.cf_weight(len(history))
    title = f"new user who likes {len(history)} games (collaborative weight {weight:.2f})"
    result = _present(table, catalog, args.top_k)
    _print_table(title, result)
    result.to_csv(Path(args.out) / "recommendations_new_user.csv", index=False)
    return 0


def _profile_user_ids(cfg: Config) -> list[str]:
    ids: list[str] = []
    for path in cfg.data.real_users:
        if Path(path).exists():
            column = pd.read_csv(path, usecols=["user_id"], encoding=cfg.data.real_user_encoding)
            ids += column["user_id"].astype(str).unique().tolist()
    return ids


def cmd_evaluate(args: argparse.Namespace) -> int:
    cfg, interactions, catalog, suite = _load(args)
    ev = cfg.evaluation
    train, test = split_per_user(interactions, ev.test_size, cfg.seed)
    log(f"Warm-user evaluation: {len(train):,} train / {len(test):,} test interactions")
    suite.fit(train)

    evaluator = Evaluator(train, catalog, k=ev.k)
    metrics = evaluator.evaluate(
        suite.select(args.models), test, n_users=args.n_users or ev.n_users, seed=cfg.seed
    )
    _emit(metrics, args.out)
    return 0


def cmd_cold_start(args: argparse.Namespace) -> int:
    cfg, interactions, catalog, suite = _load(args)
    ev = cfg.evaluation
    train, histories = split_cold_start_users(
        interactions,
        n_users=args.n_users or ev.cold_start_users,
        min_interactions=max(ev.cold_start_sizes) + 5,
        seed=cfg.seed,
    )
    log(f"Cold-start evaluation: {len(histories)} users removed from training")
    # KNN is skipped: it can only score users it was trained on.
    needed = {*COLD_START_MODELS, *cfg.adaptive.content_models, cfg.adaptive.cf_model}
    suite.fit(train, only=needed)

    evaluator = Evaluator(train, catalog, k=ev.k)
    recall = evaluator.cold_start(suite.select(COLD_START_MODELS), histories, ev.cold_start_sizes)
    recall.columns = [f"recall@{ev.k} ({c})" for c in recall.columns]
    _emit(recall, args.out)
    return 0


def _emit(table: pd.DataFrame, out: str | None) -> None:
    markdown = to_markdown(table)
    print(markdown)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(markdown + "\n", encoding="utf-8")
        log(f"saved -> {out}")


def cmd_prepare(args: argparse.Namespace) -> int:
    from hgrs.prepare import EXPECTED, prepare

    counts = prepare(Path(args.sources), Path(args.out))
    if counts != EXPECTED:
        log("The output differs from the original project's data (see the counts above).")
        return 1
    log(f"Reproduced the original project's data in {args.out}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from hgrs.synthetic import make_synthetic

    with tempfile.TemporaryDirectory() as tmp:
        config = make_synthetic(tmp, seed=args.seed)
        log(f"Generated a synthetic dataset in {tmp}")
        common = {"config": str(config), "root": tmp, "cache_dir": None}

        recommend = {**common, "users": None, "likes": None, "top_k": 5, "out": args.out}
        cmd_recommend(argparse.Namespace(**recommend, strategy="all"))
        # three sports games, as a brand-new visitor would pick them
        sports_games = ["100002", "100010", "100018"]
        cmd_recommend(
            argparse.Namespace(**{**recommend, "likes": sports_games}, strategy="adaptive")
        )

        print("\n=== warm users: accuracy and beyond ===")
        cmd_evaluate(argparse.Namespace(**common, models=list(ALL_MODELS), n_users=None, out=None))
        print("\n=== cold start: recall@10 by number of known games ===")
        return cmd_cold_start(argparse.Namespace(**common, n_users=None, out=None))


# --------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hgrs", description="Hybrid game recommendation system")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", help="YAML config (default: configs/default.yaml)")
        p.add_argument("--root", help="directory relative data paths resolve against")
        p.add_argument(
            "--cache-dir",
            default=str(REPO_ROOT / "data" / "processed"),
            help="where BERT embeddings are cached",
        )

    prep = sub.add_parser("prepare", help="rebuild the data files from the public sources")
    prep.add_argument(
        "--sources",
        default=str(REPO_ROOT / "data" / "source"),
        help="directory holding (or receiving) the downloaded source files",
    )
    prep.add_argument("--out", default=str(REPO_ROOT / "data" / "raw"))
    prep.set_defaults(func=cmd_prepare)

    rec = sub.add_parser("recommend", help="top-k games for known users or a new user")
    common(rec)
    rec.add_argument("--strategy", choices=[*STRATEGIES, "all"], default="all")
    who = rec.add_mutually_exclusive_group()
    who.add_argument("--users", nargs="+", help="user ids (default: the test profiles)")
    who.add_argument("--likes", nargs="+", metavar="APP_ID", help="games a new user likes")
    rec.add_argument("--top-k", type=int, default=10)
    rec.add_argument("--out", default=str(REPO_ROOT / "results"))
    rec.set_defaults(func=cmd_recommend)

    ev = sub.add_parser("evaluate", help="accuracy and beyond-accuracy metrics")
    common(ev)
    ev.add_argument("--models", nargs="+", choices=ALL_MODELS, default=list(ALL_MODELS))
    ev.add_argument("--n-users", type=int)
    ev.add_argument("--out", help="also write the table as markdown")
    ev.set_defaults(func=cmd_evaluate)

    cold = sub.add_parser("cold-start", help="recall as a function of known games")
    common(cold)
    cold.add_argument("--n-users", type=int)
    cold.add_argument("--out", help="also write the table as markdown")
    cold.set_defaults(func=cmd_cold_start)

    demo = sub.add_parser("demo", help="everything end-to-end on synthetic data")
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--out", default=str(Path(tempfile.gettempdir()) / "hgrs-demo"))
    demo.set_defaults(func=cmd_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
