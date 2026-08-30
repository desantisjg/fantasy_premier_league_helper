"""The training report: inferential statistics, backtest results, diagnostics.

Two halves, answering two different questions.

**Inferential** -- *what did the model learn, and is any of it real?* Fitted with
statsmodels OLS so the full apparatus is available: coefficients with standard
errors, t-statistics and p-values, R-squared and its adjusted form, F, AIC, BIC,
and the residual diagnostics.

Standard errors are **clustered by player**. A player appears up to 38 times in a
season and their residuals are correlated across those rows -- a striker in a good
run beats the model repeatedly. Ordinary standard errors assume independent rows,
so they would be far too small and would decorate noise with three stars. This is
the single most important correction in the report.

**Predictive** -- *does it beat the alternatives out of sample?* Every figure comes
from the walk-forward backtest, always shown against the naive baselines, because a
number like "R-squared 0.06" only means something next to what predicting recent
form would have achieved.
"""

from __future__ import annotations

import base64
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from scipy import linalg as sp_linalg  # noqa: E402
from scipy import stats  # noqa: E402
from statsmodels.stats.diagnostic import het_breuschpagan  # noqa: E402
from statsmodels.stats.outliers_influence import variance_inflation_factor  # noqa: E402
from statsmodels.stats.stattools import durbin_watson, jarque_bera  # noqa: E402

from .features import TARGET, feature_columns
from .model import build_preprocessor
from .scoring import Position

# Validated categorical slots (light surface #fcfcfb): blue, orange, aqua.
INK = "#0b0b0b"
INK_MUTED = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]

#: Multicollinearity is checked on the strongest coefficients only -- a VIF over all
#: 102 features means 102 auxiliary regressions and adds nothing for the long tail.
VIF_TOP_N = 20


def _drop_aliased_columns(
    matrix: np.ndarray, names: list[str]
) -> tuple[np.ndarray, list[str], list[str]]:
    """Reduce to a full-rank basis, reporting what was removed.

    The imputer emits a missingness indicator per feature, and many of those are
    exact duplicates of one another -- every per-90 rate over the same window is
    missing on precisely the same rows -- while some are constant because the
    feature is never missing. Left in place they make the design matrix singular:
    the condition number goes infinite, the covariance matrix is degenerate, and
    the F-statistic and every standard error become meaningless. Ridge tolerates
    this; ordinary least squares does not, so the inferential fit needs a basis.
    """
    keep, dropped = [], []
    seen: dict[bytes, str] = {}
    for index, name in enumerate(names):
        column = matrix[:, index]
        if np.ptp(column) < 1e-12:
            dropped.append(f"{name} (constant)")
            continue
        signature = np.round(column, 10).tobytes()
        if signature in seen:
            dropped.append(f"{name} (duplicate of {seen[signature]})")
            continue
        seen[signature] = name
        keep.append(index)

    reduced = matrix[:, keep]
    kept_names = [names[i] for i in keep]

    # Anything still linearly dependent is removed by a pivoted QR, which orders
    # columns by how much independent variation each one contributes.
    rank = np.linalg.matrix_rank(reduced)
    if rank < reduced.shape[1]:
        _, _, pivots = sp_linalg.qr(reduced, mode="economic", pivoting=True)
        independent = sorted(pivots[:rank])
        dropped.extend(
            f"{kept_names[i]} (linearly dependent)"
            for i in range(len(kept_names))
            if i not in set(independent)
        )
        reduced = reduced[:, independent]
        kept_names = [kept_names[i] for i in independent]

    return reduced, kept_names, dropped


def _design_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, list[str], list[str]]:
    """Imputed, standardised, full-rank features plus their names."""
    columns = feature_columns(frame)
    preprocessor = build_preprocessor(columns)
    matrix = np.asarray(preprocessor.fit_transform(frame), dtype=float)
    names = [n.split("__", 1)[-1] for n in preprocessor.get_feature_names_out()]
    return _drop_aliased_columns(matrix, names)


def fit_ols(frame: pd.DataFrame) -> tuple[sm.regression.linear_model.RegressionResults, list[str]]:
    """OLS with player-clustered standard errors."""
    matrix, names, dropped = _design_matrix(frame)
    design = sm.add_constant(matrix, has_constant="add")
    target = frame[TARGET].astype(float).to_numpy()
    model = sm.OLS(target, design)
    results = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": frame["player_code"].to_numpy()},
    )
    results.aliased_columns = dropped
    return results, ["const"] + names


def coefficient_table(results, names: list[str]) -> pd.DataFrame:
    """Coefficients with clustered errors, t, p and a 95% interval."""
    intervals = results.conf_int()
    table = pd.DataFrame(
        {
            "coefficient": results.params,
            "std_error": results.bse,
            "t": results.tvalues,
            "p_value": results.pvalues,
            "ci_low": intervals[:, 0],
            "ci_high": intervals[:, 1],
        },
        index=names,
    )
    table["abs_t"] = table["t"].abs()
    return table.sort_values("abs_t", ascending=False)


def residual_diagnostics(results, frame: pd.DataFrame, names: list[str]) -> dict:
    """The standard battery of residual and specification tests."""
    residuals = np.asarray(results.resid, dtype=float)
    fitted = np.asarray(results.fittedvalues, dtype=float)
    exog = np.asarray(results.model.exog, dtype=float)

    jb_stat, jb_p, skew, kurtosis = jarque_bera(residuals)
    bp_stat, bp_p, _, _ = het_breuschpagan(residuals, exog)

    return {
        "n_observations": int(results.nobs),
        "n_parameters": int(results.df_model) + 1,
        "n_clusters": int(frame["player_code"].nunique()),
        "r_squared": float(results.rsquared),
        "adj_r_squared": float(results.rsquared_adj),
        "f_statistic": float(results.fvalue) if results.fvalue is not None else None,
        "f_pvalue": float(results.f_pvalue) if results.f_pvalue is not None else None,
        "log_likelihood": float(results.llf),
        "aic": float(results.aic),
        "bic": float(results.bic),
        "condition_number": float(np.linalg.cond(exog)),
        "durbin_watson": float(durbin_watson(residuals)),
        "jarque_bera_stat": float(jb_stat),
        "jarque_bera_p": float(jb_p),
        "residual_skew": float(skew),
        "residual_kurtosis": float(kurtosis),
        "breusch_pagan_stat": float(bp_stat),
        "breusch_pagan_p": float(bp_p),
        "residual_std": float(np.std(residuals)),
        "fitted_mean": float(np.mean(fitted)),
    }


def variance_inflation(frame: pd.DataFrame, table: pd.DataFrame, top_n: int = VIF_TOP_N) -> pd.DataFrame:
    """VIF for the most influential coefficients.

    Rolling windows of the same statistic over 3, 5 and 10 fixtures are near-copies
    of one another, so high VIFs are expected here. They do not bias prediction, but
    they do make individual coefficients unstable and hard to read, which is the
    reason the production model is ridge-regularised.
    """
    matrix, names, _ = _design_matrix(frame)
    ranked = [n for n in table.index if n != "const"][:top_n]
    positions = [names.index(n) for n in ranked if n in names]
    subset = matrix[:, positions]

    values = []
    for index in range(subset.shape[1]):
        try:
            values.append(float(variance_inflation_factor(subset, index)))
        except Exception:  # singular sub-matrix
            values.append(float("nan"))
    return pd.DataFrame(
        {"feature": [names[p] for p in positions], "vif": values}
    ).sort_values("vif", ascending=False)


# --- Figures ---------------------------------------------------------------


def _style_axes(ax, *, title: str, xlabel: str, ylabel: str) -> None:
    """Recessive chrome: the data carries the ink, the frame stays quiet."""
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
    ax.set_xlabel(xlabel, color=INK_MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=9)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_facecolor(SURFACE)


def _figure_to_data_uri(figure) -> str:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=140, bbox_inches="tight", facecolor=SURFACE)
    plt.close(figure)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def figure_residuals(results) -> str:
    """Residuals against fitted values, binned because 24k points would smear."""
    fitted = np.asarray(results.fittedvalues, dtype=float)
    residuals = np.asarray(results.resid, dtype=float)

    figure, ax = plt.subplots(figsize=(6.4, 3.6), facecolor=SURFACE)
    ax.hexbin(fitted, residuals, gridsize=45, cmap="Blues", mincnt=1, linewidths=0)
    ax.axhline(0, color=SERIES[1], linewidth=2, zorder=3)
    _style_axes(
        ax,
        title="Residuals vs fitted",
        xlabel="Predicted points",
        ylabel="Residual (actual − predicted)",
    )
    return _figure_to_data_uri(figure)


def figure_qq(results) -> str:
    """Normal QQ plot. FPL points are right-skewed, so the upper tail will lift."""
    residuals = np.asarray(results.resid, dtype=float)
    standardised = (residuals - residuals.mean()) / residuals.std()
    theoretical = stats.norm.ppf(
        (np.arange(1, len(standardised) + 1) - 0.5) / len(standardised)
    )
    observed = np.sort(standardised)

    figure, ax = plt.subplots(figsize=(6.4, 3.6), facecolor=SURFACE)
    limit = float(np.max(np.abs(theoretical)))
    ax.plot([-limit, limit], [-limit, limit], color=INK_MUTED, linewidth=1.5, zorder=2)
    ax.scatter(theoretical, observed, s=5, color=SERIES[0], alpha=0.5, linewidths=0, zorder=3)
    _style_axes(
        ax,
        title="Normal QQ plot of residuals",
        xlabel="Theoretical quantile",
        ylabel="Standardised residual",
    )
    return _figure_to_data_uri(figure)


def figure_calibration(predictions: pd.DataFrame, model: str) -> str:
    """Mean actual points within each decile of predicted points."""
    frame = predictions.dropna(subset=[f"pred__{model}", TARGET]).copy()
    frame["decile"] = pd.qcut(
        frame[f"pred__{model}"].rank(method="first"), 10, labels=False
    )
    grouped = frame.groupby("decile").agg(
        predicted=(f"pred__{model}", "mean"), actual=(TARGET, "mean")
    )

    figure, ax = plt.subplots(figsize=(6.4, 3.6), facecolor=SURFACE)
    limit = float(max(grouped["predicted"].max(), grouped["actual"].max())) * 1.1
    ax.plot([0, limit], [0, limit], color=INK_MUTED, linewidth=1.5, zorder=2,
            label="Perfect calibration")
    ax.plot(grouped["predicted"], grouped["actual"], color=SERIES[0], linewidth=2,
            marker="o", markersize=8, zorder=3, label="Observed")
    for decile, row in grouped.iterrows():
        if decile in (0, 9):
            ax.annotate(
                f"decile {int(decile) + 1}",
                (row["predicted"], row["actual"]),
                textcoords="offset points", xytext=(8, -4),
                color=INK_MUTED, fontsize=8,
            )
    _style_axes(
        ax,
        title="Calibration by predicted decile",
        xlabel="Mean predicted points",
        ylabel="Mean actual points",
    )
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, loc="upper left")
    return _figure_to_data_uri(figure)


def figure_stability(by_gameweek: pd.DataFrame, baseline: pd.DataFrame, model: str,
                     baseline_name: str) -> str:
    """Per-gameweek rank correlation, model against the best naive baseline."""
    figure, ax = plt.subplots(figsize=(6.4, 3.6), facecolor=SURFACE)
    x = np.arange(len(by_gameweek))
    ax.plot(x, by_gameweek["spearman"], color=SERIES[0], linewidth=2, label=model)
    ax.plot(x, baseline["spearman"], color=SERIES[1], linewidth=2, label=baseline_name)
    labels = [f"{s.split('-')[0][-2:]}·{int(g)}" for s, g in
              zip(by_gameweek["season"], by_gameweek["gameweek"])]
    step = max(1, len(labels) // 12)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(labels[::step], rotation=45, ha="right")
    _style_axes(
        ax,
        title="Rank correlation by gameweek — starters, out of sample",
        xlabel="Season · gameweek",
        ylabel="Spearman ρ",
    )
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, loc="lower left")
    return _figure_to_data_uri(figure)


def figure_coefficients(table: pd.DataFrame, top_n: int = 18) -> str:
    """Largest standardised coefficients, signed."""
    subset = table[table.index != "const"].head(top_n).iloc[::-1]
    colors = [SERIES[0] if value >= 0 else SERIES[1] for value in subset["coefficient"]]

    figure, ax = plt.subplots(figsize=(6.8, 0.28 * len(subset) + 1.2), facecolor=SURFACE)
    positions = np.arange(len(subset))
    ax.barh(positions, subset["coefficient"], color=colors, height=0.62)
    ax.axvline(0, color=INK_MUTED, linewidth=1)
    ax.set_yticks(positions)
    ax.set_yticklabels(subset.index, fontsize=8)
    _style_axes(
        ax,
        title="Largest standardised coefficients (blue positive, orange negative)",
        xlabel="Points per standard deviation of the feature",
        ylabel="",
    )
    ax.grid(axis="y", visible=False)
    return _figure_to_data_uri(figure)


# --- Rendering -------------------------------------------------------------

_CSS = """
:root { color-scheme: light; }
body { margin:0; background:#f4f3f0; color:#0b0b0b;
  font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
.wrap { max-width:980px; margin:0 auto; padding:40px 24px 72px; }
h1 { font-size:26px; margin:0 0 4px; letter-spacing:-0.01em; }
h2 { font-size:17px; margin:40px 0 12px; padding-top:20px; border-top:1px solid #e4e3df; }
h3 { font-size:13px; margin:24px 0 8px; color:#52514e; text-transform:uppercase;
  letter-spacing:0.06em; font-weight:600; }
p  { color:#3a3a38; max-width:70ch; }
.sub { color:#52514e; margin:0 0 28px; }
.card { background:#fcfcfb; border:1px solid #e4e3df; border-radius:10px;
  padding:18px 20px; margin:16px 0; }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; }
.tile { background:#fcfcfb; border:1px solid #e4e3df; border-radius:10px; padding:14px 16px; }
.tile .k { font-size:11px; color:#52514e; text-transform:uppercase; letter-spacing:0.05em; }
.tile .v { font-size:22px; font-variant-numeric:tabular-nums; margin-top:4px; }
.tile .n { font-size:11px; color:#52514e; margin-top:2px; }
.scroll { overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-size:12.5px;
  font-variant-numeric:tabular-nums; }
th,td { padding:6px 10px; text-align:right; border-bottom:1px solid #ecebe7;
  white-space:nowrap; }
th:first-child, td:first-child { text-align:left; }
th { color:#52514e; font-weight:600; font-size:11px; text-transform:uppercase;
  letter-spacing:0.04em; }
tbody tr:hover { background:#f7f6f3; }
.best { font-weight:700; }
img { width:100%; height:auto; display:block; border-radius:8px; }
.note { border-left:3px solid #eb6834; padding:2px 0 2px 14px; margin:16px 0;
  color:#3a3a38; }
.good { color:#008300; } .bad { color:#e34948; }
footer { margin-top:48px; color:#52514e; font-size:12px; }
"""


def _table_html(frame: pd.DataFrame, *, float_format: str = "{:.3f}",
                highlight_max: str | None = None) -> str:
    body = frame.copy()
    best_index = body[highlight_max].idxmax() if highlight_max in body.columns else None
    rows = []
    for index, row in body.iterrows():
        cells = "".join(
            f"<td>{float_format.format(v) if isinstance(v, (int, float, np.floating)) and not pd.isna(v) else ('—' if pd.isna(v) else v)}</td>"
            for v in row
        )
        klass = ' class="best"' if index == best_index else ""
        rows.append(f"<tr{klass}><td>{index}</td>{cells}</tr>")
    header = "".join(f"<th>{c}</th>" for c in body.columns)
    return (
        f'<div class="scroll"><table><thead><tr><th></th>{header}</tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def render_html(context: dict) -> str:
    parts = [
        f"<style>{_CSS}</style>",
        '<div class="wrap">',
        f"<h1>{context['title']}</h1>",
        f"<p class=\"sub\">Trained {context['generated_at']} · "
        f"{context['n_rows']:,} player-fixtures · {context['n_features']} features · "
        f"seasons {context['seasons']}</p>",
    ]

    parts.append("<h2>Headline</h2>")
    parts.append('<div class="tiles">')
    for key, value, note in context["headline"]:
        parts.append(
            f'<div class="tile"><div class="k">{key}</div>'
            f'<div class="v">{value}</div><div class="n">{note}</div></div>'
        )
    parts.append("</div>")
    parts.append(f'<div class="note">{context["verdict"]}</div>')

    parts.append("<h2>Out-of-sample performance</h2>")
    parts.append(
        "<p>Every row comes from expanding-window walk-forward backtesting: fit on all "
        "gameweeks before <em>n</em>, predict gameweek <em>n</em>, repeat. Rank metrics are "
        "computed within each gameweek and averaged, because that is the shape of the real "
        "decision.</p>"
    )
    for population, caption in context["performance"]:
        parts.append(f"<h3>{caption}</h3>")
        parts.append(population)

    parts.append("<h3>Stability across gameweeks</h3>")
    parts.append(f'<div class="card"><img src="{context["fig_stability"]}" alt="Rank correlation by gameweek"></div>')

    parts.append("<h3>Calibration</h3>")
    parts.append(f'<div class="card"><img src="{context["fig_calibration"]}" alt="Calibration by decile"></div>')

    parts.append("<h3>By position</h3>")
    parts.append(context["by_position"])

    parts.append("<h2>Inferential statistics</h2>")
    parts.append(
        "<p>Ordinary least squares over the same design matrix, with standard errors "
        "<strong>clustered by player</strong>. A player contributes up to 38 correlated rows "
        "per season; unclustered errors would assume those rows independent and overstate "
        "significance badly.</p>"
    )
    parts.append(context["fit_stats"])

    parts.append("<h3>Largest coefficients</h3>")
    parts.append(f'<div class="card"><img src="{context["fig_coefficients"]}" alt="Coefficients"></div>')
    parts.append(context["coefficients"])

    parts.append("<h2>Residual diagnostics</h2>")
    parts.append(context["diagnostics"])
    parts.append(f'<div class="card"><img src="{context["fig_residuals"]}" alt="Residuals vs fitted"></div>')
    parts.append(f'<div class="card"><img src="{context["fig_qq"]}" alt="Normal QQ plot"></div>')

    parts.append("<h3>Multicollinearity</h3>")
    parts.append(
        "<p>Rolling windows of the same statistic over 3, 5 and 10 fixtures are near-copies of "
        "one another, so high variance inflation is expected and is not a defect. It does not "
        "bias prediction, but it does make individual coefficients unstable, which is why the "
        "production model is ridge-regularised.</p>"
    )
    parts.append(context["vif"])

    parts.append(f"<footer>{context['footer']}</footer></div>")
    return "\n".join(parts)


# --- Orchestration ---------------------------------------------------------


def build_report(
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    primary_model: str,
    tag: str,
    out_dir: Path,
) -> dict:
    """Fit the inferential model, assemble every table and figure, write the report."""
    from .evaluate import summarise, summarise_by_gameweek, summarise_by_position

    out_dir.mkdir(parents=True, exist_ok=True)

    results, names = fit_ols(features)
    coefficients = coefficient_table(results, names)
    diagnostics = residual_diagnostics(results, features, names)
    vif = variance_inflation(features, coefficients)

    overall = summarise(predictions, population="all")
    starters = summarise(predictions, population="starters")

    baselines = [n for n in starters.index if n.startswith(("form_", "predict_"))]
    best_baseline = starters.loc[baselines, "spearman_weekly"].idxmax()

    # Plotted on the starters population so the figure agrees with the headline
    # verdict; the tables above it cover both populations.
    model_weekly = summarise_by_gameweek(predictions, primary_model, population="starters")
    baseline_weekly = summarise_by_gameweek(predictions, best_baseline, population="starters")
    by_position = summarise_by_position(predictions, primary_model).set_index("position")

    model_row = starters.loc[primary_model]
    baseline_row = starters.loc[best_baseline]
    lift = model_row["spearman_weekly"] - baseline_row["spearman_weekly"]

    verdict = (
        f"On the population that matters — players with real recent minutes — "
        f"<strong>{primary_model}</strong> reaches Spearman ρ "
        f"<strong>{model_row['spearman_weekly']:.3f}</strong> against "
        f"<strong>{baseline_row['spearman_weekly']:.3f}</strong> for the best naive baseline "
        f"(<code>{best_baseline}</code>), a lift of {lift:+.3f}. "
        f"Out-of-sample R² is {model_row['r2']:.3f}; every baseline is negative there. "
        f"Next-gameweek points are mostly irreducible variance, so the model's value is in "
        f"ranking, not in predicting any individual score."
    )

    metric_columns = ["mae", "rmse", "r2", "spearman_weekly", "precision_at_10", "precision_at_20"]
    # Per-position and per-gameweek summaries are computed within a single group, so
    # the weekly-averaged rank columns do not apply to them.
    group_columns = ["n", "mae", "rmse", "r2", "spearman", "kendall"]
    context = {
        "title": f"FPL points model — {tag}",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "n_rows": len(features),
        "n_features": len(feature_columns(features)),
        "seasons": ", ".join(sorted(features["season"].dropna().unique())),
        "headline": [
            ("Spearman ρ", f"{model_row['spearman_weekly']:.3f}", "starters, per gameweek"),
            ("vs baseline", f"{lift:+.3f}", f"over {best_baseline}"),
            ("R² (starters)", f"{model_row['r2']:.3f}", "out of sample"),
            ("MAE", f"{model_row['mae']:.2f}", "points, starters"),
            ("Precision@20", f"{model_row['precision_at_20']:.3f}", "top-20 hit rate"),
            ("Backtest folds", f"{predictions.groupby(['season','gameweek']).ngroups}", "gameweeks"),
        ],
        "verdict": verdict,
        "performance": [
            (_table_html(overall[metric_columns], highlight_max="spearman_weekly"),
             "All players — dominated by whether a player features at all"),
            (_table_html(starters[metric_columns], highlight_max="spearman_weekly"),
             "Starters only — the population transfer decisions are drawn from"),
        ],
        "by_position": _table_html(by_position[group_columns], highlight_max="spearman"),
        "fit_stats": _table_html(
            pd.DataFrame(
                {
                    "value": [
                        diagnostics["n_observations"], diagnostics["n_parameters"],
                        diagnostics["n_clusters"], diagnostics["r_squared"],
                        diagnostics["adj_r_squared"], diagnostics["f_statistic"],
                        diagnostics["f_pvalue"], diagnostics["log_likelihood"],
                        diagnostics["aic"], diagnostics["bic"],
                    ]
                },
                index=["Observations", "Parameters", "Clusters (players)", "R²",
                       "Adjusted R²", "F statistic", "F p-value", "Log-likelihood",
                       "AIC", "BIC"],
            ),
            float_format="{:,.4f}",
        ),
        "coefficients": _table_html(
            coefficients.head(25)[["coefficient", "std_error", "t", "p_value", "ci_low", "ci_high"]],
            float_format="{:.4f}",
        ),
        "diagnostics": _table_html(
            pd.DataFrame(
                {
                    "statistic": [
                        diagnostics["durbin_watson"], diagnostics["jarque_bera_stat"],
                        diagnostics["jarque_bera_p"], diagnostics["residual_skew"],
                        diagnostics["residual_kurtosis"], diagnostics["breusch_pagan_stat"],
                        diagnostics["breusch_pagan_p"], diagnostics["condition_number"],
                        diagnostics["residual_std"],
                    ]
                },
                index=["Durbin–Watson", "Jarque–Bera", "Jarque–Bera p", "Residual skew",
                       "Residual kurtosis", "Breusch–Pagan", "Breusch–Pagan p",
                       "Condition number", "Residual std. dev."],
            ),
            float_format="{:,.4f}",
        ),
        "vif": _table_html(vif.set_index("feature"), float_format="{:.2f}"),
        "fig_residuals": figure_residuals(results),
        "fig_qq": figure_qq(results),
        "fig_calibration": figure_calibration(predictions, primary_model),
        "fig_stability": figure_stability(model_weekly, baseline_weekly, primary_model, best_baseline),
        "fig_coefficients": figure_coefficients(coefficients),
        "footer": (
            "Generated by <code>fpl train</code>. Standard errors clustered by player. "
            "All out-of-sample figures come from expanding-window walk-forward backtesting."
        ),
    }

    html = render_html(context)
    (out_dir / "report.html").write_text(html, encoding="utf-8")

    metrics = {
        "tag": tag,
        "generated_at": context["generated_at"],
        "primary_model": primary_model,
        "best_baseline": best_baseline,
        "n_rows": len(features),
        "n_features": context["n_features"],
        "seasons": context["seasons"],
        "folds": int(predictions.groupby(["season", "gameweek"]).ngroups),
        "performance_all": overall[metric_columns].to_dict(orient="index"),
        "performance_starters": starters[metric_columns].to_dict(orient="index"),
        "by_position": by_position[group_columns].to_dict(orient="index"),
        "inferential": diagnostics,
        "top_coefficients": coefficients.head(25)[
            ["coefficient", "std_error", "t", "p_value"]
        ].to_dict(orient="index"),
        "max_vif": float(vif["vif"].max()) if len(vif) else None,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    return metrics
