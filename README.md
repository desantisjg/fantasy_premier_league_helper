# FPL Analytics Assistant

An analytical Fantasy Premier League manager. A linear model is fitted to FPL points,
packaged behind command-line tools, and driven by an LLM agent that queries those
tools — including the model's own accuracy statistics — before recommending a captain,
transfers and chip timing.

The design goal was that **no recommendation is ever asserted**. Every number in the
weekly brief is computed by the regression and traceable back to it, and the agent is
told what the model's error bars are so it can say when a difference is too small to
act on.

---

## The idea

Most FPL advice is vibes. This project replaces the vibes with a fitted model, then
uses an LLM for the part a regression genuinely cannot do: reading team news, weighing
trade-offs, and explaining a decision.

Three layers, each with one job:

```
┌─────────────────────────────────────────────────────────────────────┐
│  DATA        FPL API + community archive → one per-fixture table    │
│              30,983 player-fixtures across 2025/26 and 2026/27      │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│  MODEL       99 leakage-guarded features → ridge regression         │
│              + a separate logistic model for P(10+ points)          │
│              validated by walk-forward backtest over 31 gameweeks   │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│  AGENT       Claude Opus 5 calls the model as tools, adds team      │
│              news via web search, writes the weekly brief           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## How the agent actually works

This is the part worth understanding. The agent does **not** reason about football from
memory. It has six tools, each a thin wrapper over the same functions the CLI calls, so
it physically cannot report a projection that `fpl score` would not show you.

| Tool | Returns | Why the agent needs it |
|---|---|---|
| `get_model_metrics` | Backtest results, **and the naive baseline to compare against** | So it knows how much to trust itself |
| `score_players` | Ranked projections, availability-adjusted | The core ranking |
| `captaincy_candidates` | P(10+ points) per player | Captaincy is a tail question, not a mean one |
| `explain_player` | Projection split into scoring components + recent form | To justify a pick instead of asserting it |
| `get_fixtures` | Upcoming fixtures with each club's rolling form | Fixture difficulty, derived not assumed |
| `get_my_team` | Your actual squad, bank, chips | So advice is about *your* team |
| `web_search` | Live team news *(server-side)* | The one thing the model cannot know |

### The tool that makes it honest

`get_model_metrics` is the unusual one. Before making claims, the agent asks how good
the model is, and gets back an answer that includes the naive baseline:

```json
{
  "starters": { "spearman_weekly": 0.299, "r2": 0.064, "precision_at_10": 0.120 },
  "best_naive_baseline": {
    "name": "form_l3",
    "starters": { "spearman_weekly": 0.205, "r2": -0.212 }
  },
  "interpretation": "Rank quality is what matters ... R^2 near 0.06 is normal for
   single-gameweek FPL points and does not mean the model is broken — most of the
   variance is irreducible. Treat projections as a ranking, not as forecasts."
}
```

That shapes the output. The brief opens by stating the model's accuracy, and later
rejects a transfer because *"+0.54 is inside the noise"* — a judgement it could only
make having read R² = 0.064.

### A real run

```
$ fpl brief
  → get_my_team, get_model_metrics, captaincy_candidates, score_players
  → get_fixtures, explain_player, explain_player, explain_player
  → explain_player × 6
  → explain_player × 4
  → explain_player × 3, score_players, score_players

## GW3 Brief — deadline Wed 4 Sep, 17:30 UTC

**Model health first:** weekly rank correlation 0.30, R² 0.064, precision@10 0.12.
It ranks better than form (0.20 Spearman) but does not forecast scores.

### 1. Captain: Haaland. Vice: B.Fernandes — swap your current armband
On haul probability, the metric that matters for a doubled return, Haaland is
0.421 against Fernandes's 0.317 ... his projection is almost entirely attacking
(4.69 attack, 1.11 bonus) off 0.70 xG/90 and 33 BPS/90 over five.
...
Both Hull moves lean on one fixture, so they correlate — accept that consciously.

tokens: 76,034 in / 6,662 out (161,439 cached)
```

Five turns, 23 tool calls, ~$0.63. **All 19 figures it cited reproduce exactly from
the model artifact** — verified by re-running the tools and diffing.

What is the model's, and what is the agent's, is worth separating:

- **The model's:** projections, haul probabilities, component breakdowns, rolling form.
- **The agent's:** that +0.54 is inside the noise; that two players from the same
  fixture are a correlated bet (the model scores players independently and has no
  concept of correlation); the team news; chip advice.

---

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
cp .env.example .env          # ANTHROPIC_API_KEY, FPL_ENTRY_ID

.venv/bin/fpl sync --backfill # ~12 min cold: ~700 player histories + archive
.venv/bin/fpl data build      # normalise both sources into one table
.venv/bin/fpl data features   # build the design matrix
.venv/bin/fpl train           # backtest, fit, write models/<YYYY-MM>/report.html
.venv/bin/fpl score --top 20  # rank the upcoming gameweek
.venv/bin/fpl brief           # full agent run
```

Find `FPL_ENTRY_ID` by logging in, opening the **Points** tab, and reading the number
in the URL: `fantasy.premierleague.com/entry/<ID>/event/<GW>`.

### The local chat UI

```bash
.venv/bin/fpl serve      # http://127.0.0.1:8000, opens automatically
```

A dark-mode chat app that opens with *"Hi Jordan, here is the latest gameweek
report…"* and the current brief, then lets you interrogate it — "why him?", "what
about a cheaper defender?", "how confident are you?". The assistant answers with the
same tools the brief uses, so follow-ups are grounded in the model rather than in
what it just said.

Tool activity streams live as it works (`ranking players`, `checking haul odds`,
`searching the news`), because a full agent turn can take a minute and a bare spinner
is indistinguishable from a hang.

It is a small server rather than a standalone HTML file for one reason: **the
Anthropic key stays server-side**. A page calling the API directly would have to ship
the key to the browser, where anyone can read it. Binds to localhost only.

### Commands

| Command | Does |
|---|---|
| `fpl sync` | Latest snapshot + per-fixture histories. `--backfill` adds archived seasons. |
| `fpl data build` / `features` / `info` | Normalise, build the design matrix, report disk state. |
| `fpl train` | Walk-forward backtest, refit, versioned report, promotion gate. |
| `fpl score` | Rank players. `--position`, `--max-price`, `--top`, `--output json`. |
| `fpl haul` | Rank by P(10+ points) — the captaincy question. |
| `fpl brief` | Agent run → `reports/gwNN_brief.md`. |
| `fpl serve` | Local dark-mode chat UI on `127.0.0.1:8000`. |

---

## The model

### Scoring rules as code, validated against reality

`fplr/scoring.py` encodes the 2026/27 rules — including **defensive contribution**,
introduced in 2025/26: a flat 2 points at 10 combined clearances/blocks/interceptions/
tackles for defenders, or 12 of those plus recoveries for midfielders and forwards,
capped there.

The project's gate is that this module **reproduces FPL's published points for all
29,747 player-fixtures of 2025/26, with zero mismatches** — checked per component, not
just per total, so a wrong clean-sheet rule cannot hide behind a compensating error.
Rare branches genuinely execute in that data: 11 penalty saves, 44 red cards, 9 forward
DEFCON hits.

If that test fails, nothing downstream is trustworthy.

### Two models, and an honest bake-off

The plan committed in advance to dropping extra structure if it failed to beat a simple
model. It failed, and was dropped.

- **Pooled ridge** — one regularised linear model over all features and positions.
- **Decomposed** — separate components for appearance, attack, clean sheet, DEFCON,
  bonus and saves, recombined through the scoring rules. Theoretically better motivated:
  three of those terms are not linear in anything (appearance is a step function,
  DEFCON is a threshold, concessions and saves round down).

The decomposed model lost. It is still fitted and shipped, because its component
breakdown is what lets the agent explain *why* a projection is what it is — but the
pooled ridge does the ranking.

### Results (walk-forward, 31 gameweek folds)

Restricted to **starters** — players with real recent minutes, which is the population
transfer decisions are actually drawn from:

| model | MAE | RMSE | R² | Spearman ρ | P@10 | P@20 |
|---|---|---|---|---|---|---|
| **pooled ridge** | **2.295** | **3.041** | **+0.064** | **0.299** | **0.120** | **0.190** |
| decomposed | 2.356 | 3.140 | +0.002 | 0.246 | 0.100 | 0.160 |
| form, last 3 | 2.591 | 3.460 | −0.212 | 0.205 | 0.097 | 0.137 |
| form, last 10 | 2.425 | 3.201 | −0.038 | 0.185 | 0.113 | 0.153 |
| predict the mean | 2.298 | 3.599 | −0.311 | — | 0.050 | 0.103 |

**+0.095 Spearman over the best naive baseline, and the only model with positive
out-of-sample R².**

**Choosing the population changes the answer completely.** Across *all* players, naive
form out-ranks the model (ρ 0.73 vs 0.69) — because that population is dominated by the
easy question of who plays at all. Reporting only that number would have been flattering
and useless, so the report shows both.

### Captaincy gets its own model

Residual diagnostics show **skew 2.78, kurtosis 16.2** — a very heavy right tail. A
conditional mean cannot represent that, and captaincy doubles one return, so it is a bet
on the tail rather than the average. `fpl haul` fits `P(points ≥ 10)` separately, and the
two rankings genuinely differ — a test asserts it.

### The training report

`fpl train` writes a self-contained `report.html` with the full inferential battery:

- Coefficients with **standard errors clustered by player** — a player contributes up to
  38 correlated rows per season, and unclustered errors would decorate noise with stars.
- R², adjusted R², F, AIC/BIC, log-likelihood
- Durbin–Watson **2.005** (no residual autocorrelation)
- Breusch–Pagan p ≈ 0 (strong heteroskedasticity — why robust errors are mandatory)
- Jarque–Bera p ≈ 0, skew 2.78, kurtosis 16.2
- VIF, condition number, and residual/QQ/calibration/stability plots

Plus the walk-forward results above, always beside the naive baselines.

---

## Engineering notes

The parts that were harder than expected, and what they cost.

**Leakage is guarded by experiment, not inspection.** Each row's features describe what
was knowable *before* that fixture; the target is that fixture's points. A test corrupts
one fixture's stats to 999 and asserts its own 99 features do not move. If any rolling
window forgot to shift, it fails loudly.

**Scoring must be per fixture, never per gameweek.** `goals_conceded // 2` and
`saves // 3` both round down, so conceding 1+1 across a double gameweek costs nothing in
reality but −1 on the aggregate.

**FPL's own team-strength ratings are unusable, twice over.** The live bootstrap reports
every `strength_attack_*` and `strength_defence_*` as `0`, and the archive's `teams.csv`
is an end-of-season snapshot that leaks how the season finished. Team strength is derived
instead from lagged rolling goals and xG.

**A club-attribution bug the scoring oracle could never catch.** Player club was first
taken from `players_raw.csv`, which records a player's club at *season end* — so anyone
transferred in January had that club stamped onto their August fixtures. The 29,747-row
scoring replay passed the whole time, because club does not affect points. Club now comes
from the fixture itself.

**Projections decayed across the horizon for a subtle reason.** A rolling window counts
*rows*, and an unplayed fixture is a row — so a projection four gameweeks out spent most
of its window on fixtures that had not happened. Haaland decayed 6.60 → 3.89 across
GW3–GW7, an artefact indistinguishable from a hard run of fixtures. Form is now frozen at
the first unplayed fixture. The first fix then over-corrected and froze the *opponent's*
form too, which defeats the point of a horizon.

**A rank-deficient design matrix invalidated the entire inferential report.** The first
run gave F = 0.049 (p = 0.999) *alongside* R² = 0.32, with an infinite condition number:
the imputer emits a missingness indicator per feature and many are exact duplicates.
Ridge tolerates a singular design; OLS does not. Reducing to a full-rank basis first (89
aliased columns) gives F = 111.4, p < 0.001, R² unchanged.

**The live agent run found a bug no offline test could.** `explain_player("White")`
returned *Gibbs-White* — substring matching resolved a shorter name to a longer one and
reported the wrong player's numbers under the right player's name. The most dangerous
shape of tool bug: internally consistent, no error, entirely plausible. Exact names now
win, and genuine ambiguity returns the candidate list instead of guessing.

---

## Operations

**Monthly retrain is gated on measured improvement.** A new model is written to
`models/<YYYY-MM>/` but only promoted to `models/current` if it ranks starters at least
as well as the incumbent. Silently degrading advice is the most likely way a project like
this fails without anyone noticing.

**Snapshots are immutable.** Every sync writes a dated directory of raw JSON; training
reads from a named snapshot, never the live API, so any model version can be rebuilt
byte-for-byte.

**Only finalised gameweeks are trainable.** FPL revises bonus points and defensive
contributions while Opta reviews a gameweek. The API exposes this as `data_checked`, and
provisional rows are marked rather than dropped — useless for training, but exactly what
you want as *features* when scoring the next gameweek.

```cron
0 8 * * 5  /path/to/premier_league_llm/scripts/weekly.sh   # Friday: brief
0 6 1 * *  /path/to/premier_league_llm/scripts/monthly.sh  # 1st: retrain
```

---

## Layout

```
src/fplr/
  scoring.py    2026/27 rules — the oracle the project is validated against
  ingest.py     FPL API client, dated snapshots, per-fixture history cache
  datasets.py   normalise both sources into one per-fixture table
  features.py   design matrix with leakage guards
  model.py      pooled ridge, decomposed components, haul model
  evaluate.py   walk-forward backtest and metrics
  report.py     inferential + predictive ML report
  train.py      versioning and the promotion gate
  score.py      score upcoming fixtures
  agent/        runner (shared loop), tools, weekly brief
  web/          FastAPI server + the chat UI
.claude/skills/fpl/   Claude Code skill
```

## Tests

```bash
.venv/bin/python -m pytest -q     # 74 passed
```

| File | Covers |
|---|---|
| `test_scoring.py` | Replay of real gameweeks against FPL's own attribution |
| `test_datasets.py` | Full-season replay, schema, fixture integrity |
| `test_features.py` | Leakage, horizon freezing, walk-forward ordering |
| `test_model.py` | Component arithmetic, calibration, evaluation harness |
| `test_agent.py` | Tool contracts, ambiguity handling, baseline visibility |
| `test_web.py` | UI server endpoints, brief ordering, credential failure path |

---

## Caveats

**Projections are a ranking, not a forecast.** R² ≈ 0.064 on starters. Most of the
variance in a single gameweek is irreducible, and no model will change that. The value is
in ordering players better than form does.

**A linear model predicts conditional means and will under-predict hauls.** That is why
captaincy has a separate tail model.

**The 2026/27 BPS rework makes prior-season bonus data partly stale**, so the bonus
component is the weakest early in the season.

**Only ~1.3 seasons of DEFCON-era data exists.** Defensive contribution began in 2025/26,
so earlier seasons have a different target distribution and are deliberately excluded.
This is the binding constraint on model complexity, and the main reason a regularised
linear model is the right call rather than something deeper.

Design decisions, findings and the reasoning behind them: [PLAN.md](PLAN.md).
