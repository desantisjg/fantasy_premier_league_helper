"""Model arithmetic, component consistency, and evaluation-harness behaviour."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fplr.evaluate import _safe_corr, precision_at_k, score_predictions, summarise
from fplr.features import TARGET, build_features
from fplr.model import (
    DecomposedModel,
    PooledLinearModel,
    attacking_points,
    defcon_hit,
    expected_floor_div,
)
from fplr.scoring import ASSIST_POINTS, GOAL_POINTS, Position


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    try:
        frame = build_features()
    except FileNotFoundError:
        pytest.skip("features unavailable; run `fpl data build` first")
    return frame[frame["season"] == "2025-26"]


# --- Scoring-rule arithmetic ------------------------------------------------


def test_expected_floor_div_matches_brute_force():
    """The Poisson expectation must match a direct sum over the distribution."""
    from scipy.stats import poisson

    for rate in (0.2, 1.0, 2.5, 4.0):
        counts = np.arange(0, 60)
        brute = float((counts // 2 * poisson.pmf(counts, rate)).sum())
        assert np.isclose(expected_floor_div(np.array([rate]), 2)[0], brute, atol=1e-6)


def test_expected_floor_div_is_below_the_naive_shortcut():
    """E[floor(X/2)] < E[X]/2 -- the reason the shortcut over-penalises."""
    rates = np.array([0.5, 1.0, 2.0])
    assert np.all(expected_floor_div(rates, 2) < rates / 2)


def test_expected_floor_div_handles_zero_rate():
    assert expected_floor_div(np.array([0.0]), 3)[0] == 0.0


def test_attacking_points_uses_position_specific_goal_values():
    frame = pd.DataFrame(
        {
            "position": [int(Position.DEF), int(Position.FWD)],
            "goals_scored": [1, 1],
            "assists": [1, 1],
        }
    )
    points = attacking_points(frame)
    assert points.iloc[0] == GOAL_POINTS[Position.DEF] + ASSIST_POINTS
    assert points.iloc[1] == GOAL_POINTS[Position.FWD] + ASSIST_POINTS


def test_defcon_hit_uses_position_thresholds():
    frame = pd.DataFrame(
        {
            "position": [int(Position.DEF), int(Position.MID), int(Position.GK)],
            "defensive_contribution": [10, 10, 99],
        }
    )
    hits = defcon_hit(frame)
    assert hits.tolist() == [1, 0, 0], "DEF hits at 10, MID needs 12, GK is ineligible"


# --- Fitted models ----------------------------------------------------------


def test_pooled_model_produces_finite_predictions(features):
    train = features[features["gameweek"] <= 20]
    test = features[features["gameweek"] == 21]
    predictions = PooledLinearModel().fit(train).predict(test)
    assert len(predictions) == len(test)
    assert np.isfinite(predictions).all()


def test_decomposed_components_sum_to_the_total(features):
    """The reported breakdown must actually add up to the prediction it explains."""
    train = features[features["gameweek"] <= 20]
    test = features[features["gameweek"] == 21]
    model = DecomposedModel().fit(train)
    parts = model.predict_components(test)

    components = [c for c in parts.columns if c != "expected_points"]
    assert np.allclose(
        parts[components].sum(axis=1).to_numpy(),
        parts["expected_points"].to_numpy(),
        atol=1e-9,
    )


def test_decomposed_defcon_is_conditioned_on_playing(features):
    """DEFCON is fitted on players who appeared, so it must be scaled by P(play).

    Without that scaling the component roughly doubles, because it would assert
    that a benched player still racks up tackles.
    """
    train = features[features["gameweek"] <= 25]
    test = features[features["gameweek"] > 25]
    parts = DecomposedModel().fit(train).predict_components(test)

    actual = (2 * defcon_hit(test)).mean()
    predicted = parts["defensive_contribution"].mean()
    assert predicted == pytest.approx(actual, abs=0.05), (
        f"defensive contribution mis-calibrated: predicted {predicted:.3f} "
        f"vs actual {actual:.3f}"
    )


def test_goalkeepers_never_receive_defensive_contribution_points(features):
    train = features[features["gameweek"] <= 20]
    test = features[(features["gameweek"] == 21) & (features["position"] == int(Position.GK))]
    if test.empty:
        pytest.skip("no goalkeepers in the fold")
    parts = DecomposedModel().fit(train).predict_components(test)
    assert (parts["defensive_contribution"] == 0).all()


# --- Evaluation harness -----------------------------------------------------


def test_precision_at_k_is_exact_on_a_known_case():
    predicted = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    actual = np.array([5.0, 4.0, 0.0, 0.0, 9.0])
    # Top-2 predicted are indices 0,1; top-2 actual are 4,0 -> one shared.
    assert precision_at_k(predicted, actual, 2) == 0.5


def test_safe_corr_returns_nan_for_a_constant_predictor():
    constant = np.full(50, 1.234)
    actual = np.arange(50, dtype=float)
    assert np.isnan(_safe_corr(__import__("scipy.stats", fromlist=["spearmanr"]).spearmanr,
                               constant, actual))


def test_score_predictions_reports_zero_error_for_perfect_predictions():
    actual = np.array([0.0, 2.0, 6.0, 13.0])
    metrics = score_predictions(actual.copy(), actual)
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == pytest.approx(1.0)
    assert metrics["spearman"] == pytest.approx(1.0)


def test_summarise_ranks_models_and_keeps_both_populations():
    frame = pd.DataFrame(
        {
            "season": ["2025-26"] * 8,
            "gameweek": [1] * 4 + [2] * 4,
            TARGET: [0.0, 2.0, 6.0, 1.0, 3.0, 0.0, 8.0, 2.0],
            "is_starter": [True, True, False, False, True, True, True, False],
            "pred__good": [0.0, 2.0, 6.0, 1.0, 3.0, 0.0, 8.0, 2.0],
            "pred__bad": [6.0, 1.0, 0.0, 2.0, 0.0, 8.0, 2.0, 3.0],
        }
    )
    everything = summarise(frame, population="all")
    assert everything.index[0] == "good"
    assert everything.loc["good", "mae"] == 0.0

    starters = summarise(frame, population="starters")
    assert starters.loc["good", "n"] == 5
