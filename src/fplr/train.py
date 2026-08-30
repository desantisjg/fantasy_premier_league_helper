"""Fitting, evaluating and versioning a model.

Each run writes a self-contained, dated directory under `models/`: the fitted
estimator, the metrics as JSON, and the human-readable report. Nothing overwrites
anything, so a bad month is always recoverable and two months can be compared.

**Promotion is gated on measured improvement.** A newly trained model becomes
`models/current` only if it ranks starters at least as well as the incumbent.
Retraining monthly on a season that is still developing is exactly the situation
where a refit can quietly get worse -- an unlucky run of fixtures, a rule change
part-way through -- and silently degrading the advice is the most likely way this
project fails without anyone noticing.
"""

from __future__ import annotations

import json
import pickle
from datetime import date
from pathlib import Path

import pandas as pd

from .config import MODELS_DIR
from .evaluate import summarise, walk_forward
from .features import build_features
from .model import DecomposedModel, HaulModel, PooledLinearModel
from .report import build_report

#: Metric the promotion gate is decided on: how well the model ranks real options.
PROMOTION_METRIC = "spearman_weekly"

#: How much worse a retrain may be and still be promoted. Small negative drift is
#: noise; a real regression should block.
PROMOTION_TOLERANCE = 0.01

CURRENT_LINK = "current"


def default_tag(today: date | None = None) -> str:
    """Monthly cadence, so a month's retrain has one obvious name."""
    return (today or date.today()).strftime("%Y-%m")


def load_current_metrics(models_dir: Path = MODELS_DIR) -> dict | None:
    path = models_dir / CURRENT_LINK / "metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _promotion_decision(new: dict, incumbent: dict | None) -> tuple[bool, str]:
    if incumbent is None:
        return True, "no incumbent model; promoting by default"

    model = new["primary_model"]
    new_score = new["performance_starters"][model][PROMOTION_METRIC]
    old_model = incumbent.get("primary_model", model)
    old_score = incumbent["performance_starters"][old_model][PROMOTION_METRIC]
    delta = new_score - old_score

    if delta >= -PROMOTION_TOLERANCE:
        return True, (
            f"promoted: {PROMOTION_METRIC} {new_score:.4f} vs incumbent "
            f"{old_score:.4f} ({delta:+.4f})"
        )
    return False, (
        f"NOT promoted: {PROMOTION_METRIC} {new_score:.4f} vs incumbent "
        f"{old_score:.4f} ({delta:+.4f}), beyond the {PROMOTION_TOLERANCE} tolerance"
    )


def train(
    *,
    tag: str | None = None,
    models_dir: Path = MODELS_DIR,
    min_train_gameweeks: int = 8,
    promote: bool = True,
    verbose: bool = True,
) -> dict:
    """Backtest, fit on everything, write a versioned report, and decide promotion."""
    tag = tag or default_tag()
    out_dir = models_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    features = build_features(finalised_only=True)
    if verbose:
        print(f"features: {len(features):,} rows")

    candidates = {"pooled": PooledLinearModel(), "decomposed": DecomposedModel()}
    if verbose:
        print("walk-forward backtest...")
    predictions = walk_forward(
        features, candidates, min_train_gameweeks=min_train_gameweeks
    )
    predictions.to_parquet(out_dir / "walkforward.parquet", index=False)

    # The model that actually ranks starters best is the one that gets shipped,
    # whatever the intended design was.
    starters = summarise(predictions, population="starters")
    ranked = [n for n in starters.index if n in candidates]
    primary = max(ranked, key=lambda n: starters.loc[n, PROMOTION_METRIC])
    if verbose:
        print(f"primary model: {primary}")

    fitted = {name: model.fit(features) for name, model in candidates.items()}

    # Captaincy is a tail decision, not a mean one, so it gets its own model.
    haul = HaulModel().fit(features)

    with (out_dir / "models.pkl").open("wb") as handle:
        pickle.dump(
            {"primary": primary, "models": fitted, "haul": haul}, handle
        )

    metrics = build_report(
        features, predictions, primary_model=primary, tag=tag, out_dir=out_dir
    )

    incumbent = load_current_metrics(models_dir)
    promoted, reason = _promotion_decision(metrics, incumbent)
    metrics["promoted"] = promoted
    metrics["promotion_reason"] = reason
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )

    if promote and promoted:
        link = models_dir / CURRENT_LINK
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(out_dir.name, target_is_directory=True)

    if verbose:
        print(reason)
        print(f"report: {out_dir / 'report.html'}")
    return metrics


def load_current_model(models_dir: Path = MODELS_DIR) -> dict:
    """The promoted model bundle, ready to score with.

    Returns the ranking model, the haul model, and the decomposed model kept for
    explanation -- the last of these does not rank as well, but its component
    breakdown is what turns a number into a reason.
    """
    path = models_dir / CURRENT_LINK / "models.pkl"
    if not path.exists():
        raise FileNotFoundError("no promoted model; run `fpl train` first")
    with path.open("rb") as handle:
        bundle = pickle.load(handle)
    return {
        "name": bundle["primary"],
        "model": bundle["models"][bundle["primary"]],
        "haul": bundle.get("haul"),
        "components": bundle["models"].get("decomposed"),
    }
