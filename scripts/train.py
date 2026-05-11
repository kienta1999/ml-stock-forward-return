#!/usr/bin/env python3
"""Tune + train an XGBoost regressor on the panel.

- Training loss:   reg:squarederror (RMSE-equivalent — what XGBoost minimizes).
- Tuning method:   optuna TPE, default 50 trials, maximize **mean realised
                   return** of the equal-weighted top-``TOP_N`` (40)
                   portfolio on val. Objective history:
                     • pre-May 2026: val decile spread — anti-correlated
                       with top-40 backtest CAGR.
                     • May 2026 (briefly): val top-N Sharpe — degenerate
                       (best_iteration=0, val IC ≈ 0).
                     • current: val top-N mean return — non-gameable
                       (zero alpha → zero objective).
- Early stopping:  100 rounds on val top-N mean return.
- Outputs:
    models/xgb_v1.json
    reports/feature_importance.csv
    reports/optuna_trials.csv
    reports/train_metrics.json

CLI:
    uv run python scripts/train.py                # full pipeline (~10-15 min)
    uv run python scripts/train.py --trials 20    # faster tune
    uv run python scripts/train.py --quick        # skip tuning, use sane defaults
"""

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dataset import FEATURE_COLS, TARGET_COL, load_panel, split  # noqa: E402
from strategy import TOP_N  # noqa: E402  — used as the optuna scoring window

_ROOT = os.path.dirname(_HERE)
MODEL_PATH = os.path.join(_ROOT, "models", "xgb_v1.json")
REPORTS_DIR = os.path.join(_ROOT, "reports")

DEFAULT_TRIALS = 50
EARLY_STOPPING_ROUNDS = 100

# Defaults for --quick: best params from the May 2026 50-trial sweep under
# the top-N mean-return objective (trial #47, val top-40 mean return +0.0142,
# best_iteration 19, val IC +0.0350). Reproducible without tuning.
DEFAULT_PARAMS: dict = {
    "max_depth": 5,
    "learning_rate": 0.0019884662813417562,
    "n_estimators": 274,
    "min_child_weight": 1,
    "subsample": 0.6748841148505009,
    "colsample_bytree": 0.6236790005526994,
    "reg_lambda": 0.3047565711822111,
}


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────


def rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(((y_true.to_numpy() - y_pred) ** 2).mean()))


def daily_ic(dates: pd.Series, y_true: pd.Series, y_pred: np.ndarray) -> float:
    """Mean daily Spearman rank correlation between predictions and actuals.

    This is the metric the *ranker* actually cares about: on each day we sort
    stocks by predicted return and compare against the sort by realised return.
    Range −1..+1; >0.03 is considered tradable.
    """
    df = pd.DataFrame({
        "date": dates.to_numpy(),
        "y_true": y_true.to_numpy(),
        "y_pred": y_pred,
    })
    ics = df.groupby("date").apply(
        lambda g: g["y_pred"].corr(g["y_true"], method="spearman") if len(g) > 1 else np.nan
    )
    return float(ics.mean())


def decile_spread(
    dates: pd.Series, y_true: pd.Series, y_pred: np.ndarray, n_quantiles: int = 10
) -> float:
    """Mean over days of (top-decile actual return) − (bottom-decile actual return).

    Kept as a diagnostic alongside ``top_n_sharpe`` (the new optuna objective).
    Decile spread averages ranking quality across deciles 1 & 10 (~50 names
    each from a ~500-stock universe); the strategy holds only the top
    ``TOP_N`` (40), so the two metrics can disagree — a model can have higher
    decile spread but worse top-40 returns. See README §"BLOCKER" callout
    under Next steps for the empirical anti-correlation observed during the
    8-K work (May 2026).
    """
    df = pd.DataFrame({
        "date": dates.to_numpy(),
        "y_true": y_true.to_numpy(),
        "y_pred": y_pred,
    })

    def per_date(g: pd.DataFrame) -> float:
        if len(g) < n_quantiles:
            return np.nan
        g_sorted = g.sort_values("y_pred")
        bucket_size = len(g_sorted) // n_quantiles
        bot = g_sorted.iloc[:bucket_size]["y_true"].mean()
        top = g_sorted.iloc[-bucket_size:]["y_true"].mean()
        return top - bot

    return float(df.groupby("date").apply(per_date).mean())


def _top_n_daily_returns(
    dates: pd.Series, y_true: pd.Series, y_pred: np.ndarray, top_n: int = TOP_N
) -> pd.Series:
    """Per-date series of "mean realised return of the top-``top_n`` predicted names."""
    df = pd.DataFrame({
        "date": dates.to_numpy(),
        "y_true": y_true.to_numpy(),
        "y_pred": y_pred,
    })

    def per_date(g: pd.DataFrame) -> float:
        if len(g) < top_n:
            return np.nan
        return g.nlargest(top_n, "y_pred")["y_true"].mean()

    return df.groupby("date").apply(per_date).dropna()


def top_n_mean_return(
    dates: pd.Series, y_true: pd.Series, y_pred: np.ndarray, top_n: int = TOP_N
) -> float:
    """Mean realised return of an equal-weighted top-``TOP_N`` portfolio over val.

    Active optuna objective (May 2026 → present). Replaced ``top_n_sharpe``
    after a 50-trial sweep with the Sharpe objective produced
    ``best_iteration=0`` / ``val_IC=+0.0069`` — optuna gamed the Sharpe ratio
    by pushing predictions to a degenerate near-constant basin (collapsed
    std → high mean/std even with zero learned signal). Mean return cannot
    be gamed that way: zero alpha → zero objective.
    """
    daily = _top_n_daily_returns(dates, y_true, y_pred, top_n)
    if len(daily) == 0:
        return float("nan")
    return float(daily.mean())


def top_n_sharpe(
    dates: pd.Series, y_true: pd.Series, y_pred: np.ndarray, top_n: int = TOP_N
) -> float:
    """Diagnostic — Sharpe of equal-weighted top-``TOP_N`` portfolio over val.

    Was the optuna objective briefly in May 2026; demoted to diagnostic
    after the degeneracy described in :func:`top_n_mean_return`. Reported
    alongside ``val_top_n_mean_return`` to track how Sharpe moves as the
    optimiser pushes mean return up.
    """
    daily = _top_n_daily_returns(dates, y_true, y_pred, top_n)
    if len(daily) < 2:
        return float("nan")
    sd = float(daily.std(ddof=1))
    if sd == 0.0:
        return 0.0
    return float(daily.mean() / sd)


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────


def _make_top_n_mean_return_eval_metric(dates_val: np.ndarray, top_n: int = TOP_N):
    """Closure that XGBoost calls as its eval metric on each boosting round.

    Computes the mean realised return of an equal-weighted top-``top_n``
    portfolio on the val set so early stopping picks the tree count that
    maximises the metric the strategy actually trades on. The inner
    function is named ``top_n_mean_return`` because XGBoost's sklearn
    wrapper uses ``func.__name__`` as the metric label — this is what
    ``EarlyStopping(metric_name="top_n_mean_return")`` matches.

    Hot path: pre-computes per-date row indices once at closure creation
    and uses ``np.argsort(kind='stable')`` per round instead of
    pandas ``groupby().nlargest()``. ~3-4× faster than pandas while
    matching pandas tie-breaking (stable sort breaks ties by original
    index, same as ``nlargest``), so ``best_iteration`` is bit-exact.
    """
    # Pre-compute per-date row indices once; reused every boosting round.
    _, inverse = np.unique(dates_val, return_inverse=True)
    n_dates = int(inverse.max()) + 1 if len(inverse) else 0
    date_groups: list[np.ndarray] = [
        np.where(inverse == i)[0] for i in range(n_dates)
    ]
    # Drop dates with fewer than top_n names so the sort is always valid.
    date_groups = [g for g in date_groups if len(g) >= top_n]

    def top_n_mean_return(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if not date_groups:
            return float("nan")
        daily_means = np.empty(len(date_groups), dtype=np.float64)
        for i, idx in enumerate(date_groups):
            preds = y_pred[idx]
            top_idx = np.argsort(-preds, kind="stable")[:top_n]
            daily_means[i] = y_true[idx][top_idx].mean()
        return float(daily_means.mean())
    return top_n_mean_return


def _make_model(
    params: dict, dates_val: pd.Series, seed: int = 42
) -> xgb.XGBRegressor:
    """XGBoost regressor with native categorical (gics_sector) support.

    Early stopping watches val top-``TOP_N`` mean realised return
    (maximize, save_best) — the raw return the strategy delivers, which
    cannot be gamed by predicting a near-constant. Decile spread was the
    objective until May 2026 (anti-correlated with top-40 CAGR); a brief
    Sharpe-objective experiment hit a degenerate basin (best_iteration=0,
    val IC ≈ 0). Mean return is the simplest non-gameable choice.
    """
    es = xgb.callback.EarlyStopping(
        rounds=EARLY_STOPPING_ROUNDS,
        metric_name="top_n_mean_return",
        maximize=True,
        save_best=True,
    )
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        enable_categorical=True,
        random_state=seed,
        n_jobs=-1,
        eval_metric=_make_top_n_mean_return_eval_metric(dates_val.to_numpy()),
        callbacks=[es],
        **params,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Optuna tuning
# ─────────────────────────────────────────────────────────────────────────────


def _objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    dates_val: pd.Series,
    seed: int = 42,
) -> float:
    """Train one model with the trial's hyperparameters; return val top-N mean return.

    XGBoost's internal loss is RMSE (smooth gradient), but we score the
    model by the mean realised return of an equal-weighted top-``TOP_N``
    portfolio — the raw P&L the strategy delivers. Decile spread (the
    prior objective) was anti-correlated with top-40 backtest CAGR in the
    May 2026 8-K work; a brief Sharpe-objective experiment hit a
    degenerate basin (best_iteration=0). See README §"BLOCKER" callout
    for details. ``max_depth`` is fixed at 3: depth-8 produces clumpy
    predictions that score well on IC but collapse top-N selection.
    """
    params = {
        # max_depth=6 has been provably dominated across 4 sweeps (always in
        # the worst-trial bucket). Narrowed to [3, 5] to focus search.
        # Rank-normalized fundamentals introduce sector × value interactions
        # that may justify depth 4-5; depth-3 still wins so far.
        "max_depth": trial.suggest_int("max_depth", 3, 5),
        # learning_rate: tightened from [0.005, 0.3] to [0.001, 0.02] (log).
        # The slow-build basin (lr~0.005, best_iteration in the 13-43 range)
        # has consistently outperformed the aggressive-shallow basin
        # (lr~0.03+, best_iteration ~5) — TPE was burning trials on the latter
        # at 50-trial budgets, posting val spread ~0.0175 vs the ~0.0235
        # achievable in the slow basin. Capping at 0.02 forces optuna there.
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.02, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        # colsample_bytree: tightened from [0.6, 1.0] to [0.55, 0.75].
        # Project memory: 0.629 was identified as the actual lever that
        # unlocked cross-sectional signal; values >0.8 give every tree
        # almost-all features and collapse decile spread. Window stays
        # asymmetric around the known-good point so optuna can still wander.
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 0.75),
        # reg_lambda: floor lowered from 0.01 to 0.001 (log). The saved
        # known-good params hit 0.01003 — bumping the prior wall — so the
        # true optimum likely lives below 0.01. Ceiling tightened from 10.0
        # to 1.0 after a 50-trial run on the 47-feature panel landed at
        # 5.86 (paired with high lr + best_iter=34 = val-overfit trap that
        # cost ~3.9 CAGR pts on test). Keeping the asymmetric-around-
        # known-good shape used for colsample_bytree.
        "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 1.0, log=True),
    }
    model = _make_model(params, dates_val, seed=seed)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    y_pred = model.predict(X_val)
    return top_n_mean_return(dates_val, y_val, y_pred)


def tune(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    dates_val: pd.Series,
    n_trials: int,
    seed: int = 42,
) -> tuple[dict, optuna.study.Study]:
    """Run optuna search; return (best_params, study)."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(
        lambda t: _objective(t, X_train, y_train, X_val, y_val, dates_val, seed=seed),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    return study.best_params, study


# ─────────────────────────────────────────────────────────────────────────────
# Final fit + metrics
# ─────────────────────────────────────────────────────────────────────────────


def fit_and_evaluate(
    params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    dates_train: pd.Series,
    dates_val: pd.Series,
    seed: int = 42,
) -> tuple[xgb.XGBRegressor, dict]:
    model = _make_model(params, dates_val, seed=seed)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    metrics: dict = {
        "params": params,
        "best_iteration": int(model.best_iteration),
        "train_rmse": rmse(y_train, y_train_pred),
        "val_rmse": rmse(y_val, y_val_pred),
        "train_ic": daily_ic(dates_train, y_train, y_train_pred),
        "val_ic": daily_ic(dates_val, y_val, y_val_pred),
        # Active optuna objective: mean realised return of top-N portfolio over val.
        "val_top_n_mean_return": top_n_mean_return(dates_val, y_val, y_val_pred),
        # Diagnostics — track how the optimiser trades risk vs raw return.
        "val_top_n_sharpe": top_n_sharpe(dates_val, y_val, y_val_pred),
        "val_decile_spread": decile_spread(dates_val, y_val, y_val_pred),
    }
    return model, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────


def save_model(model: xgb.XGBRegressor, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save_model(path)


def save_reports(
    model: xgb.XGBRegressor,
    study: optuna.study.Study | None,
    metrics: dict,
    feature_cols: list[str],
    reports_dir: str,
    seed: int = 42,
) -> None:
    os.makedirs(reports_dir, exist_ok=True)

    fi_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    fi_df.to_csv(os.path.join(reports_dir, "feature_importance.csv"), index=False)
    # When sweeping seeds, also save a per-seed copy so the stability-selection
    # post-processing can pool importances across runs.
    if seed != 42:
        fi_df.to_csv(
            os.path.join(reports_dir, f"feature_importance_seed{seed}.csv"),
            index=False,
        )

    if study is not None:
        study.trials_dataframe().to_csv(
            os.path.join(reports_dir, "optuna_trials.csv"), index=False
        )

    with open(os.path.join(reports_dir, "train_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _print_metrics(metrics: dict) -> None:
    print("\nFinal metrics:")
    print(
        f"  train RMSE: {metrics['train_rmse']:.4f}   "
        f"train IC: {metrics['train_ic']:+.4f}"
    )
    print(
        f"  val   RMSE: {metrics['val_rmse']:.4f}   "
        f"val   IC: {metrics['val_ic']:+.4f}"
    )
    print(
        f"  val top-{TOP_N} mean return: {metrics['val_top_n_mean_return']:+.4f}  "
        f"(Sharpe diag: {metrics['val_top_n_sharpe']:+.4f}  "
        f"decile spread diag: {metrics['val_decile_spread']:+.4f})"
    )
    print(f"  best iteration: {metrics['best_iteration']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_TRIALS,
        help=f"Optuna trial count (default {DEFAULT_TRIALS}).",
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Skip tuning; use sane default hyperparameters.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (XGBoost random_state + optuna TPE seed). "
             "Use different values for stability-selection sweeps.",
    )
    ap.add_argument("--out", default=MODEL_PATH)
    args = ap.parse_args()

    panel = load_panel()
    train_df, val_df, _ = split(panel)

    X_train, y_train, dates_train = (
        train_df[FEATURE_COLS], train_df[TARGET_COL], train_df["date"]
    )
    X_val, y_val, dates_val = (
        val_df[FEATURE_COLS], val_df[TARGET_COL], val_df["date"]
    )

    print(f"\nTrain: {len(X_train):,} rows  |  Val: {len(X_val):,} rows")

    print(f"Seed: {args.seed}")

    if args.quick:
        print("\n--quick: skipping optuna, using default hyperparameters.")
        best_params = dict(DEFAULT_PARAMS)
        study = None
    else:
        print(
            f"\nRunning optuna ({args.trials} trials, maximize val top-{TOP_N} mean return)..."
        )
        best_params, study = tune(
            X_train, y_train, X_val, y_val, dates_val, args.trials, seed=args.seed
        )
        print(
            f"\nBest val top-{TOP_N} mean return during tuning: {study.best_value:+.4f}"
        )
        print("Best params:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")

    print("\nFitting final model with best params...")
    model, metrics = fit_and_evaluate(
        best_params,
        X_train, y_train, X_val, y_val,
        dates_train, dates_val,
        seed=args.seed,
    )

    _print_metrics(metrics)

    save_model(model, args.out)
    save_reports(model, study, metrics, FEATURE_COLS, REPORTS_DIR, seed=args.seed)
    print(f"\n  -> model saved to {args.out}")
    print(f"  -> reports saved to {REPORTS_DIR}/")


if __name__ == "__main__":
    main()
