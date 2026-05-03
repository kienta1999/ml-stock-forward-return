#!/usr/bin/env python3
"""Tune + train an XGBoost regressor on the panel.

- Training loss:   reg:squarederror (RMSE-equivalent — what XGBoost minimizes).
- Tuning method:   optuna TPE, ~50 trials, maximize mean daily IC on val.
- Early stopping:  50 rounds on val RMSE.
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

_ROOT = os.path.dirname(_HERE)
MODEL_PATH = os.path.join(_ROOT, "models", "xgb_v1.json")
REPORTS_DIR = os.path.join(_ROOT, "reports")

DEFAULT_TRIALS = 50
EARLY_STOPPING_ROUNDS = 100

# Defaults for --quick: best params from the 200-trial sweep
# (val decile spread +0.0297, val IC +0.0554). Reproducible without tuning.
DEFAULT_PARAMS: dict = {
    "max_depth": 3,
    "learning_rate": 0.0822,
    "n_estimators": 860,
    "min_child_weight": 11,
    "subsample": 0.9487,
    "colsample_bytree": 0.6285,
    "reg_lambda": 0.4457,
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

    Direct measure of how much money the strategy would make from the model's
    ranking, before any backtest realism (costs, holding period, sleeves).
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


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────


def _make_decile_spread_eval_metric(dates_val: np.ndarray, n_quantiles: int = 10):
    """Closure that XGBoost calls as its eval metric on each boosting round.

    Computes mean daily decile spread on the val set so early stopping picks
    the tree count that maximises the metric we actually trade on. The inner
    function is named ``decile_spread`` because XGBoost's sklearn wrapper
    uses ``func.__name__`` as the metric label — this is what
    ``EarlyStopping(metric_name="decile_spread")`` matches.
    """
    def decile_spread(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        df = pd.DataFrame({"date": dates_val, "y_true": y_true, "y_pred": y_pred})

        def per_date(g: pd.DataFrame) -> float:
            if len(g) < n_quantiles:
                return np.nan
            g_sorted = g.sort_values("y_pred")
            bucket_size = len(g_sorted) // n_quantiles
            bot = g_sorted.iloc[:bucket_size]["y_true"].mean()
            top = g_sorted.iloc[-bucket_size:]["y_true"].mean()
            return top - bot

        return float(df.groupby("date").apply(per_date).mean())
    return decile_spread


def _make_model(params: dict, dates_val: pd.Series) -> xgb.XGBRegressor:
    """XGBoost regressor with native categorical (gics_sector) support.

    Early stopping watches val decile spread (maximize, save_best) — the
    metric the strategy actually cares about. IC and decile spread can
    disagree (e.g. depth-8 trees produce clumpy predictions with high IC
    but poor decile separation), so we score on the one tied to P&L.
    """
    es = xgb.callback.EarlyStopping(
        rounds=EARLY_STOPPING_ROUNDS,
        metric_name="decile_spread",
        maximize=True,
        save_best=True,
    )
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        enable_categorical=True,
        random_state=42,
        n_jobs=-1,
        eval_metric=_make_decile_spread_eval_metric(dates_val.to_numpy()),
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
) -> float:
    """Train one model with the trial's hyperparameters; return val decile spread.

    XGBoost's internal loss is RMSE (smooth gradient), but we score the model
    by decile spread — the strategy actually trades the top/bottom buckets,
    not the rank-correlation order. ``max_depth`` is fixed at 3: depth-8
    produces clumpy predictions that score well on IC but collapse decile
    separation, and depth-4/5 give equivalent val spread to depth-3 (we
    explored 3-5 in optuna and the basin is flat at the top, so we pick the
    shallowest config — most trees, smoothest predictions, lowest single-tree
    risk). See README v1 results.
    """
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
    }
    model = _make_model(params, dates_val)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    y_pred = model.predict(X_val)
    return decile_spread(dates_val, y_val, y_pred)


def tune(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    dates_val: pd.Series,
    n_trials: int,
) -> tuple[dict, optuna.study.Study]:
    """Run optuna search; return (best_params, study)."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(
        lambda t: _objective(t, X_train, y_train, X_val, y_val, dates_val),
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
) -> tuple[xgb.XGBRegressor, dict]:
    model = _make_model(params, dates_val)
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
) -> None:
    os.makedirs(reports_dir, exist_ok=True)

    fi_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    fi_df.to_csv(os.path.join(reports_dir, "feature_importance.csv"), index=False)

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
        f"val   IC: {metrics['val_ic']:+.4f}   "
        f"decile spread: {metrics['val_decile_spread']:+.4f}"
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

    if args.quick:
        print("\n--quick: skipping optuna, using default hyperparameters.")
        best_params = dict(DEFAULT_PARAMS)
        study = None
    else:
        print(f"\nRunning optuna ({args.trials} trials, maximize val decile spread)...")
        best_params, study = tune(
            X_train, y_train, X_val, y_val, dates_val, args.trials
        )
        print(f"\nBest val decile spread during tuning: {study.best_value:+.4f}")
        print("Best params:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")

    print("\nFitting final model with best params...")
    model, metrics = fit_and_evaluate(
        best_params,
        X_train, y_train, X_val, y_val,
        dates_train, dates_val,
    )

    _print_metrics(metrics)

    save_model(model, args.out)
    save_reports(model, study, metrics, FEATURE_COLS, REPORTS_DIR)
    print(f"\n  -> model saved to {args.out}")
    print(f"  -> reports saved to {REPORTS_DIR}/")


if __name__ == "__main__":
    main()
