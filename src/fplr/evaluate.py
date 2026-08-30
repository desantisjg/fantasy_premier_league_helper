"""Walk-forward evaluation.

Every number here comes from expanding-window, gameweek-by-gameweek backtesting:
fit on everything before gameweek *n*, predict gameweek *n*, move on. A random
train/test split would let the model see the rest of the season and would flatter it
badly, because player form is strongly autocorrelated.

**Ranking is reported alongside error**, and matters more. No decision in FPL needs
a player's exact score; every decision needs to know who is worth more than whom.
A model can win on MAE by shrinking everything toward the mean and still be useless
for picking a captain.

**Two populations are scored separately.** Across all players, rank correlation is
dominated by the easy question of who will play at all -- half the league is
benched or injured in any given week. The `starters` view restricts to players with
real recent minutes, which is the population actual transfer decisions are drawn
from, and is the harder and more relevant test.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .features import TARGET, walk_forward_splits

#: A player is treated as a live option if they averaged at least this many minutes
#: over their last five fixtures.
STARTER_MINUTES = 45.0


def _safe_corr(method, predicted: np.ndarray, actual: np.ndarray) -> float:
    """Rank correlation, or NaN where it is undefined.

    A constant predictor (the predict-the-mean baseline) has no rank ordering at
    all, so the correlation does not exist. Tested with a tolerance rather than an
    exact zero, since a "constant" array reconstructed through float arithmetic is
    rarely bit-identical.
    """
    if len(predicted) < 3:
        return float("nan")
    if np.ptp(predicted) < 1e-12 or np.ptp(actual) < 1e-12:
        return float("nan")
    return float(method(predicted, actual).statistic)


def precision_at_k(predicted: np.ndarray, actual: np.ndarray, k: int) -> float:
    """Share of the top-k predicted players who were genuinely top-k scorers."""
    if len(predicted) < k:
        return float("nan")
    top_predicted = set(np.argsort(-predicted)[:k])
    top_actual = set(np.argsort(-actual)[:k])
    return len(top_predicted & top_actual) / k


def score_predictions(predicted: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    """Error and rank metrics for one set of predictions."""
    error = predicted - actual
    variance = np.var(actual)
    return {
        "n": float(len(actual)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
        "r2": float(1 - np.mean(error**2) / variance) if variance > 0 else float("nan"),
        "spearman": _safe_corr(stats.spearmanr, predicted, actual),
        "kendall": _safe_corr(stats.kendalltau, predicted, actual),
    }


# --- Baselines --------------------------------------------------------------
# A model is only as good as what it improves on. These are the cheap answers a
# manager could reach for without any model at all.


def baseline_predictions(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, np.ndarray]:
    """Naive predictors evaluated on the same fold as the model."""
    return {
        # What you get for doing nothing at all.
        "predict_mean": np.full(len(test), train[TARGET].astype(float).mean()),
        # What a manager does by eye: recent points, over three horizons. `form_l5`
        # is the closest analogue to FPL's own on-site `form` figure.
        "form_l3": test["total_points_mean_l3"].fillna(0).to_numpy(),
        "form_l5": test["total_points_mean_l5"].fillna(0).to_numpy(),
        "form_l10": test["total_points_mean_l10"].fillna(0).to_numpy(),
    }


@dataclass
class FoldResult:
    season: str
    gameweek: int
    predictions: pd.DataFrame  # one row per player, all models plus the truth


def walk_forward(
    frame: pd.DataFrame,
    models: dict[str, object],
    *,
    min_train_gameweeks: int = 8,
    include_baselines: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """Backtest every model gameweek by gameweek.

    Returns one row per player-fixture per fold, holding each model's prediction
    beside the realised points, so any metric can be recomputed afterwards without
    refitting.
    """
    rows = []
    splits = walk_forward_splits(frame, min_train_gameweeks=min_train_gameweeks)

    for train_index, test_index in splits:
        train = frame.loc[train_index]
        test = frame.loc[test_index]
        season = test["season"].iloc[0]
        gameweek = int(test["gameweek"].iloc[0])

        block = test[
            ["season", "gameweek", "player_code", "web_name", "position", "minutes", TARGET]
        ].copy()
        block["is_starter"] = (
            test["minutes_mean_l5"].fillna(0) >= STARTER_MINUTES
        ).to_numpy()

        for name, prototype in models.items():
            # Refit a fresh copy each fold; deep-copying the prototype preserves its
            # hyperparameters without carrying over the previous fold's fit.
            model = copy.deepcopy(prototype)
            model.fit(train)
            block[f"pred__{name}"] = model.predict(test)

        if include_baselines:
            for name, values in baseline_predictions(train, test).items():
                block[f"pred__{name}"] = values

        rows.append(block)
        if verbose:
            print(f"  {season} GW{gameweek}: train={len(train):,} test={len(test):,}", flush=True)

    if not rows:
        raise RuntimeError("no walk-forward folds were produced")
    return pd.concat(rows, ignore_index=True)


def summarise(
    predictions: pd.DataFrame, *, population: str = "all"
) -> pd.DataFrame:
    """Aggregate fold predictions into one metric row per model.

    Error metrics (MAE, RMSE, R^2) are pooled across every prediction. Rank metrics
    are computed *within each gameweek and then averaged*, because that is the shape
    of the real decision: you choose among the players available this week, never
    against players from three months ago. Pooling ranks across the whole backtest
    also silently rewards a model for knowing that April is not August.
    """
    frame = predictions
    if population == "starters":
        frame = frame[frame["is_starter"]]

    actual = frame[TARGET].astype(float).to_numpy()
    model_columns = [c for c in frame.columns if c.startswith("pred__")]

    results = {}
    for column in model_columns:
        name = column.removeprefix("pred__")
        metrics = score_predictions(frame[column].astype(float).to_numpy(), actual)

        per_week = []
        for _, block in frame.groupby(["season", "gameweek"]):
            truth = block[TARGET].astype(float).to_numpy()
            guess = block[column].astype(float).to_numpy()
            per_week.append(
                {
                    "spearman": _safe_corr(stats.spearmanr, guess, truth),
                    "precision_at_10": precision_at_k(guess, truth, 10),
                    "precision_at_20": precision_at_k(guess, truth, 20),
                }
            )
        weekly = pd.DataFrame(per_week).mean(numeric_only=True)
        metrics["spearman_weekly"] = float(weekly["spearman"])
        metrics["precision_at_10"] = float(weekly["precision_at_10"])
        metrics["precision_at_20"] = float(weekly["precision_at_20"])
        results[name] = metrics

    return pd.DataFrame(results).T.sort_values("spearman_weekly", ascending=False)


def summarise_by_gameweek(
    predictions: pd.DataFrame, model: str, *, population: str = "all"
) -> pd.DataFrame:
    """Per-gameweek metrics for one model, for stability plots.

    Note that at the start of a season every rolling feature is empty, so no player
    qualifies as a starter and that fold drops out of the `starters` view entirely.
    """
    frame = predictions
    if population == "starters":
        frame = frame[frame["is_starter"]]

    rows = []
    for (season, gameweek), block in frame.groupby(["season", "gameweek"]):
        metrics = score_predictions(
            block[f"pred__{model}"].astype(float).to_numpy(),
            block[TARGET].astype(float).to_numpy(),
        )
        rows.append({"season": season, "gameweek": gameweek, **metrics})
    return pd.DataFrame(rows).sort_values(["season", "gameweek"])


def summarise_by_position(predictions: pd.DataFrame, model: str) -> pd.DataFrame:
    """Per-position metrics for one model."""
    from .scoring import Position

    rows = []
    for position_id, block in predictions.groupby("position"):
        metrics = score_predictions(
            block[f"pred__{model}"].astype(float).to_numpy(),
            block[TARGET].astype(float).to_numpy(),
        )
        rows.append({"position": Position(int(position_id)).name, **metrics})
    return pd.DataFrame(rows)
