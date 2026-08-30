"""Models for expected FPL points.

Two models live here, and the report is expected to show the second beating the
first -- otherwise the extra structure is not earning its keep and we should
simplify.

**Pooled baseline.** One regularised linear model over every feature and every
position. This is the literal reading of "regress FPL points on player data", and it
is kept permanently as the thing to beat.

**Decomposed model.** Points are not a single quantity: they are a sum of terms with
different generating processes, and three of those terms are not linear in anything.
Appearance points are a step function of minutes; defensive contribution pays a flat
2 at a positional threshold and nothing beyond it; goals conceded and saves both
round down. Fitting one slope through all of that wastes most of the signal. So each
component is modelled on its own scale and recombined using the scoring rules from
`fplr.scoring`:

    E[points] = appearance
              + attacking returns
              + clean-sheet value x P(clean sheet)
              + 2 x P(defensive contribution threshold)
              + E[bonus]
              + goalkeeper saves
              - expected concession and card penalties

Every component stays linear (ridge, or logistic for the probabilities), so the
whole thing remains inspectable coefficient by coefficient.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import TARGET, feature_columns
from .scoring import (
    ASSIST_POINTS,
    CLEAN_SHEET_POINTS,
    DEFENSIVE_CONTRIBUTION_POINTS,
    DEFENSIVE_CONTRIBUTION_THRESHOLD,
    GOAL_POINTS,
    GOALS_CONCEDED_PER_POINT,
    SAVES_PER_POINT,
    Position,
)

#: Ridge penalty. Rolling windows of the same statistic are heavily collinear, so
#: some shrinkage is required for the coefficients to be readable at all.
DEFAULT_ALPHA = 30.0


def build_preprocessor(columns: list[str]) -> ColumnTransformer:
    """Impute, flag, and standardise.

    Missing per-90 rates mean "no minutes in the window", which is information, not
    absence of it. Median imputation alone would quietly assert that a benched
    player performs like a median starter, so `add_indicator` keeps an explicit
    missingness flag for the model to use.
    """
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer([("numeric", numeric, columns)], remainder="drop")


@dataclass
class PooledLinearModel:
    """Ridge regression of points on every feature, all positions pooled."""

    alpha: float = DEFAULT_ALPHA
    columns: list[str] = field(default_factory=list)
    pipeline: Pipeline | None = None

    def fit(self, frame: pd.DataFrame) -> "PooledLinearModel":
        self.columns = feature_columns(frame)
        self.pipeline = Pipeline(
            [
                ("prep", build_preprocessor(self.columns)),
                ("model", Ridge(alpha=self.alpha)),
            ]
        )
        self.pipeline.fit(frame, frame[TARGET].astype(float))
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("model is not fitted")
        return self.pipeline.predict(frame)


# --- Component targets ------------------------------------------------------


def attacking_points(frame: pd.DataFrame) -> pd.Series:
    """Points from goals and assists alone, per fixture."""
    goal_value = frame["position"].map({int(p): GOAL_POINTS[p] for p in Position})
    return (
        frame["goals_scored"].fillna(0) * goal_value
        + frame["assists"].fillna(0) * ASSIST_POINTS
    ).astype(float)


def defcon_hit(frame: pd.DataFrame) -> pd.Series:
    """Whether the defensive contribution threshold was reached."""
    thresholds = frame["position"].map(
        {int(p): DEFENSIVE_CONTRIBUTION_THRESHOLD[p] or np.inf for p in Position}
    )
    return (frame["defensive_contribution"].fillna(0) >= thresholds).astype(int)


def expected_floor_div(rate: np.ndarray, divisor: int, max_terms: int = 12) -> np.ndarray:
    """E[floor(X / divisor)] for X ~ Poisson(rate).

    Goals conceded and saves both score on a rounded-down count, so the expected
    penalty is not the rounded-down expectation. Using `E[X] / divisor` instead
    would systematically over-penalise low-conceding sides: a team expected to
    concede 1.0 loses 0.5 points under that shortcut, but only about 0.26 in truth.
    """
    rate = np.clip(np.asarray(rate, dtype=float), 0.0, None)
    from scipy.stats import poisson

    total = np.zeros_like(rate)
    for step in range(1, max_terms):
        total += poisson.sf(step * divisor - 1, rate)
    return total


@dataclass
class _Component:
    """One fitted sub-model plus the rows it was trained on."""

    pipeline: Pipeline
    kind: str  # "regressor" or "classifier"

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.kind == "classifier":
            if len(self.pipeline.classes_) == 1:
                return np.full(len(frame), float(self.pipeline.classes_[0]))
            return self.pipeline.predict_proba(frame)[:, 1]
        return self.pipeline.predict(frame)


def _fit_component(
    frame: pd.DataFrame,
    columns: list[str],
    target: pd.Series,
    *,
    kind: str,
    alpha: float = DEFAULT_ALPHA,
    weights: pd.Series | None = None,
) -> _Component | None:
    """Fit one component, or return None if the data cannot support it."""
    usable = target.notna()
    if usable.sum() < 50:
        return None
    frame, target = frame[usable], target[usable]
    if weights is not None:
        weights = weights[usable]

    if kind == "classifier":
        if target.nunique() < 2:
            return None
        estimator = LogisticRegression(max_iter=2000, C=1.0)
    else:
        estimator = Ridge(alpha=alpha)

    pipeline = Pipeline(
        [("prep", build_preprocessor(columns)), ("model", estimator)]
    )
    fit_params = {}
    if weights is not None:
        fit_params["model__sample_weight"] = weights.to_numpy()
    pipeline.fit(frame, target, **fit_params)

    # `Pipeline.classes_` already delegates to the final estimator.
    return _Component(pipeline=pipeline, kind=kind)


@dataclass
class PositionModel:
    """The full component set for one position."""

    position: Position
    columns: list[str]
    components: dict[str, _Component] = field(default_factory=dict)

    def _predict(self, name: str, frame: pd.DataFrame, default: float) -> np.ndarray:
        component = self.components.get(name)
        if component is None:
            return np.full(len(frame), default, dtype=float)
        return np.asarray(component.predict(frame), dtype=float)

    def expected_points(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Predicted points, returned as its components so advice can be explained."""
        played = np.clip(self._predict("played", frame, 0.0), 0.0, 1.0)
        full = np.clip(self._predict("played_60", frame, 0.0), 0.0, 1.0)
        full = np.minimum(full, played)  # a 60-minute game is also an appearance
        minutes = np.clip(self._predict("minutes", frame, 0.0), 0.0, 90.0)

        parts = pd.DataFrame(index=frame.index)
        parts["appearance"] = (played - full) * 1.0 + full * 2.0
        parts["attack"] = (
            np.clip(self._predict("attack_p90", frame, 0.0), 0.0, None) * minutes / 90.0
        )

        clean_sheet_value = CLEAN_SHEET_POINTS[self.position]
        parts["clean_sheet"] = (
            clean_sheet_value * np.clip(self._predict("clean_sheet", frame, 0.0), 0.0, 1.0) * full
        )

        if self.position in (Position.GK, Position.DEF):
            conceded = np.clip(self._predict("conceded", frame, 0.0), 0.0, None)
            parts["concession"] = -expected_floor_div(
                conceded, GOALS_CONCEDED_PER_POINT
            ) * played
        else:
            parts["concession"] = 0.0

        if self.position is Position.GK:
            saves = np.clip(self._predict("saves", frame, 0.0), 0.0, None)
            parts["saves"] = expected_floor_div(saves, SAVES_PER_POINT) * played
        else:
            parts["saves"] = 0.0

        if DEFENSIVE_CONTRIBUTION_THRESHOLD[self.position] is None:
            parts["defensive_contribution"] = 0.0
        else:
            # The classifier is fitted on rows where the player appeared, so it
            # estimates P(threshold | played) and must be weighted by P(played) --
            # exactly as every other conditional component is.
            parts["defensive_contribution"] = (
                DEFENSIVE_CONTRIBUTION_POINTS
                * np.clip(self._predict("defcon", frame, 0.0), 0.0, 1.0)
                * played
            )

        parts["bonus"] = np.clip(self._predict("bonus", frame, 0.0), 0.0, None) * played
        parts["discipline"] = -np.clip(self._predict("discipline", frame, 0.0), 0.0, None) * played
        parts["expected_points"] = parts.sum(axis=1)
        return parts


@dataclass
class DecomposedModel:
    """Per-position component models, recombined through the scoring rules."""

    alpha: float = DEFAULT_ALPHA
    columns: list[str] = field(default_factory=list)
    positions: dict[int, PositionModel] = field(default_factory=dict)

    def fit(self, frame: pd.DataFrame) -> "DecomposedModel":
        self.columns = feature_columns(frame)
        self.positions = {}

        for position in Position:
            rows = frame[frame["position"] == int(position)]
            if len(rows) < 100:
                continue

            minutes = rows["minutes"].fillna(0).astype(float)
            appeared = minutes > 0
            played_full = minutes >= 60

            model = PositionModel(position=position, columns=self.columns)
            fit = lambda target, kind, **kw: _fit_component(  # noqa: E731
                rows, self.columns, target, kind=kind, alpha=self.alpha, **kw
            )

            model.components["played"] = fit(appeared.astype(int), "classifier")
            model.components["played_60"] = fit(played_full.astype(int), "classifier")
            model.components["minutes"] = fit(minutes, "regressor")

            # Attacking output is a rate, so it is fitted only where there were
            # minutes to earn it in, weighted so a full match counts for more than
            # a cameo.
            on_pitch = rows[appeared]
            if len(on_pitch) >= 50:
                per_90 = attacking_points(on_pitch) / minutes[appeared] * 90.0
                model.components["attack_p90"] = _fit_component(
                    on_pitch,
                    self.columns,
                    per_90,
                    kind="regressor",
                    alpha=self.alpha,
                    weights=minutes[appeared],
                )
                model.components["bonus"] = _fit_component(
                    on_pitch, self.columns, on_pitch["bonus"].fillna(0).astype(float),
                    kind="regressor", alpha=self.alpha,
                )
                discipline = (
                    on_pitch["yellow_cards"].fillna(0) * 1
                    + on_pitch["red_cards"].fillna(0) * 3
                    + on_pitch["own_goals"].fillna(0) * 2
                    + on_pitch["penalties_missed"].fillna(0) * 2
                ).astype(float)
                model.components["discipline"] = _fit_component(
                    on_pitch, self.columns, discipline, kind="regressor", alpha=self.alpha
                )
                model.components["defcon"] = _fit_component(
                    on_pitch, self.columns, defcon_hit(on_pitch), kind="classifier"
                )
                if position is Position.GK:
                    model.components["saves"] = _fit_component(
                        on_pitch, self.columns, on_pitch["saves"].fillna(0).astype(float),
                        kind="regressor", alpha=self.alpha,
                    )

            # Clean sheets and concessions are only defined for a full appearance.
            full_rows = rows[played_full]
            if len(full_rows) >= 50:
                model.components["clean_sheet"] = _fit_component(
                    full_rows, self.columns,
                    full_rows["clean_sheets"].fillna(0).astype(int),
                    kind="classifier",
                )
                model.components["conceded"] = _fit_component(
                    full_rows, self.columns,
                    full_rows["goals_conceded"].fillna(0).astype(float),
                    kind="regressor", alpha=self.alpha,
                )

            self.positions[int(position)] = model

        return self

    def predict_components(self, frame: pd.DataFrame) -> pd.DataFrame:
        parts = pd.DataFrame(index=frame.index, dtype=float)
        for position_id, model in self.positions.items():
            rows = frame[frame["position"] == position_id]
            if rows.empty:
                continue
            block = model.expected_points(rows)
            for column in block.columns:
                if column not in parts.columns:
                    parts[column] = np.nan
                parts.loc[rows.index, column] = block[column].to_numpy()
        return parts

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        parts = self.predict_components(frame)
        if "expected_points" not in parts:
            return np.zeros(len(frame))
        return parts["expected_points"].fillna(0.0).to_numpy()


#: A "haul" is the return that actually wins a gameweek.
HAUL_THRESHOLD = 10


@dataclass
class HaulModel:
    """P(points >= threshold): the captaincy question, which is about the tail.

    Expected points answers "who scores most on average". Captaincy doubles one
    player's return, so it asks something different: who is most likely to have a
    big week. Those are not the same player. The residual diagnostics make the case
    quantitatively -- skew 2.8, kurtosis 16 -- a conditional mean cannot represent a
    distribution that heavy on the right, so the tail gets its own model.
    """

    threshold: int = HAUL_THRESHOLD
    columns: list[str] = field(default_factory=list)
    pipeline: Pipeline | None = None
    fallback: float = 0.0

    def fit(self, frame: pd.DataFrame) -> "HaulModel":
        self.columns = feature_columns(frame)
        target = (frame[TARGET].astype(float) >= self.threshold).astype(int)
        self.fallback = float(target.mean())
        if target.nunique() < 2:
            self.pipeline = None
            return self
        self.pipeline = Pipeline(
            [
                ("prep", build_preprocessor(self.columns)),
                ("model", LogisticRegression(max_iter=2000, C=1.0)),
            ]
        )
        self.pipeline.fit(frame, target)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            return np.full(len(frame), self.fallback)
        return self.pipeline.predict_proba(frame)[:, 1]
