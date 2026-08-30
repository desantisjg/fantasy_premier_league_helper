"""Command line interface.

Every command is a thin wrapper over a plain function in the package, so the agent
tools in M6 and this CLI share one implementation rather than drifting apart.
"""

from __future__ import annotations

import json

import typer

from . import ingest
from .config import CURRENT_SEASON, PROCESSED_DIR, TRAINING_SEASONS

app = typer.Typer(add_completion=False, help="Analytical Fantasy Premier League assistant.")
data_app = typer.Typer(help="Inspect and build the local dataset.")
app.add_typer(data_app, name="data")


def _echo_json(payload: object) -> None:
    typer.echo(json.dumps(payload, indent=2, default=str))


@app.command()
def sync(
    histories: bool = typer.Option(
        True, help="Also refresh per-fixture player histories (~700 requests on a cold cache)."
    ),
    backfill: bool = typer.Option(
        False, help="Download the archived seasons used for training."
    ),
    force: bool = typer.Option(False, help="Refetch data already cached."),
) -> None:
    """Pull the latest FPL data into a dated snapshot.

    Only gameweeks FPL has marked `finished` and `data_checked` are captured as
    final -- bonus points and defensive contributions are still mutable before the
    Opta review completes.
    """
    client = ingest.FPLClient()
    manifest = ingest.sync(client=client, force=force)
    typer.echo(
        f"snapshot {manifest['snapshot']}: "
        f"latest finalised GW{manifest['latest_finalised']}, "
        f"next GW{manifest['next_gameweek']} deadline {manifest['next_deadline']}"
    )

    if histories:
        typer.echo("refreshing per-fixture histories...")
        result = ingest.sync_histories(client=client, force=force, progress=True)
        typer.echo(
            f"  {result['players_fetched']} fetched, {result['players_cached']} cached, "
            f"{len(result['failures'])} failed"
        )

    if backfill:
        from .datasets import fetch_archive_file

        for season in TRAINING_SEASONS:
            if season == CURRENT_SEASON:
                continue
            typer.echo(f"downloading {season} archive...")
            for name in ("gws/merged_gw.csv", "players_raw.csv", "teams.csv", "fixtures.csv"):
                fetch_archive_file(season, name, refresh=force)
        typer.echo("  archive cached")


@data_app.command("build")
def data_build(
    seasons: str = typer.Option(
        ",".join(TRAINING_SEASONS), help="Comma-separated seasons to include."
    ),
) -> None:
    """Normalise every source into one per-fixture table."""
    from .datasets import build_player_fixtures

    selected = tuple(s.strip() for s in seasons.split(",") if s.strip())
    frame = build_player_fixtures(selected)
    target = PROCESSED_DIR / "player_fixtures.parquet"
    typer.echo(f"wrote {len(frame):,} rows to {target}")
    for season, rows in sorted(frame.groupby("season").size().to_dict().items()):
        block = frame[frame.season == season]
        trainable = int(block["finalised"].fillna(False).sum())
        gameweeks = block["gameweek"].max()
        note = "" if trainable == rows else f" ({rows - trainable:,} provisional, excluded from training)"
        typer.echo(f"  {season}: {rows:,} rows through GW{gameweeks}, {trainable:,} trainable{note}")


@data_app.command("info")
def data_info() -> None:
    """Report what is currently on disk."""
    snapshot = ingest.latest_snapshot()
    if snapshot is None:
        typer.echo("no snapshot; run `fpl sync`")
        raise typer.Exit(1)

    manifest = snapshot.read("manifest") if snapshot.has("manifest") else {}
    history_dir = ingest.history_cache_dir(CURRENT_SEASON)
    cached = len(list(history_dir.glob("*.json.gz"))) if history_dir.exists() else 0

    parquet = PROCESSED_DIR / "player_fixtures.parquet"
    _echo_json(
        {
            "snapshot": snapshot.name,
            "latest_finalised_gameweek": manifest.get("latest_finalised"),
            "next_gameweek": manifest.get("next_gameweek"),
            "next_deadline": manifest.get("next_deadline"),
            "histories_cached": cached,
            "player_fixtures_parquet": str(parquet) if parquet.exists() else None,
        }
    )


@data_app.command("features")
def data_features(
    include_provisional: bool = typer.Option(
        False,
        help="Include gameweeks FPL is still revising. Never use this for training.",
    ),
) -> None:
    """Build the design matrix from the per-fixture table."""
    from .features import build_features, feature_columns

    frame = build_features(finalised_only=not include_provisional)
    target = PROCESSED_DIR / "features.parquet"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=False)

    columns = feature_columns(frame)
    typer.echo(f"wrote {len(frame):,} rows x {len(columns)} features to {target}")
    coverage = frame[columns].notna().mean().sort_values()
    typer.echo(f"  least-populated feature: {coverage.index[0]} ({coverage.iloc[0]:.0%} present)")
    typer.echo(f"  median feature coverage: {coverage.median():.0%}")


@app.command("train")
def train_command(
    tag: str = typer.Option(None, help="Version name for this run (default: YYYY-MM)."),
    min_train_gameweeks: int = typer.Option(8, help="Gameweeks of history before the first fold."),
    promote: bool = typer.Option(True, help="Update models/current if the model is not worse."),
) -> None:
    """Backtest, fit, and write a versioned model with a full ML report."""
    from .train import train as run_training

    metrics = run_training(tag=tag, min_train_gameweeks=min_train_gameweeks, promote=promote)
    model = metrics["primary_model"]
    starters = metrics["performance_starters"][model]
    typer.echo(
        f"\n{model}: spearman(weekly)={starters['spearman_weekly']:.4f} "
        f"R2={starters['r2']:.4f} MAE={starters['mae']:.3f} on starters"
    )
    typer.echo(metrics["promotion_reason"])


@app.command("score")
def score_command(
    horizon: int = typer.Option(5, help="How many upcoming gameweeks to project."),
    gameweek: int = typer.Option(None, help="Rank this gameweek (default: the next one)."),
    top: int = typer.Option(25, help="How many players to show."),
    position: str = typer.Option(None, help="Filter to GK, DEF, MID or FWD."),
    max_price: float = typer.Option(None, help="Only players at or below this price."),
    output: str = typer.Option("table", help="table, json, or markdown."),
) -> None:
    """Rank players for the upcoming gameweeks using the promoted model."""
    from .score import rank_for_gameweek, score_upcoming
    from .train import load_current_model

    bundle = load_current_model()
    scored = score_upcoming(
        bundle["model"],
        horizon=horizon,
        haul_model=bundle["haul"],
        components_model=bundle["components"],
    )
    ranked = rank_for_gameweek(scored, gameweek)

    if position:
        ranked = ranked[ranked["position_name"] == position.upper()]
    if max_price is not None:
        ranked = ranked[ranked["price"] <= max_price]
    ranked = ranked.head(top)

    columns = [
        "web_name", "position_name", "team_name", "price", "fixtures",
        "projected_points", "p_haul_adjusted", "chance_of_playing",
        "selected_by_percent",
    ]
    columns = [c for c in columns if c in ranked.columns]
    view = ranked[columns].round(3)

    if output == "json":
        _echo_json(view.to_dict(orient="records"))
    elif output == "markdown":
        typer.echo(view.to_markdown(index=False))
    else:
        typer.echo(view.to_string(index=False))


@app.command("haul")
def haul_command(
    top: int = typer.Option(15, help="How many players to show."),
    horizon: int = typer.Option(1, help="Upcoming gameweeks to consider."),
) -> None:
    """Rank by probability of a double-digit return — the captaincy question.

    Expected points answers who scores most on average; captaincy doubles one
    player's return, so it asks who is most likely to have a big week. The residual
    diagnostics (skew 2.8, kurtosis 16) are why these are two different questions.
    """
    from .score import rank_for_gameweek, score_upcoming
    from .train import load_current_model

    bundle = load_current_model()
    if bundle["haul"] is None:
        typer.echo("no haul model in the current bundle; run `fpl train`")
        raise typer.Exit(1)

    scored = score_upcoming(bundle["model"], horizon=horizon, haul_model=bundle["haul"])
    ranked = rank_for_gameweek(scored).sort_values("p_haul_adjusted", ascending=False)
    columns = ["web_name", "position_name", "team_name", "price",
               "p_haul_adjusted", "projected_points", "selected_by_percent"]
    typer.echo(ranked.head(top)[columns].round(3).to_string(index=False))


@app.command("brief")
def brief_command(
    question: str = typer.Option(None, help="Ask something specific instead of the standard brief."),
    write: bool = typer.Option(True, help="Save the brief to reports/."),
) -> None:
    """Run the agent and produce the pre-deadline brief.

    Needs Anthropic credentials: either ANTHROPIC_API_KEY, or a profile stored by
    `ant auth login`, which the SDK picks up with no environment variable set.
    """
    from .agent.weekly import BriefError, run_brief

    try:
        result = run_brief(question=question, write=write)
    except (BriefError, FileNotFoundError) as error:
        typer.echo(str(error))
        raise typer.Exit(1)

    typer.echo("")
    typer.echo(result.text)
    if result.path:
        typer.echo(f"\nsaved to {result.path}")
    usage = result.usage
    typer.echo(
        f"tokens: {usage['input_tokens']:,} in / {usage['output_tokens']:,} out"
        f" ({usage['cache_read_input_tokens']:,} cached)"
    )


@app.command("serve")
def serve_command(
    port: int = typer.Option(8000, help="Port to listen on."),
    host: str = typer.Option("127.0.0.1", help="Interface to bind. Localhost by default."),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open a browser window."),
) -> None:
    """Start the local chat UI.

    Binds to localhost only. The Anthropic key stays server-side and is never sent
    to the browser, which is why this is a small server rather than a bare HTML
    file — a page calling the API directly would have to ship the key to the client.
    """
    import threading
    import webbrowser

    from .web.server import serve

    url = f"http://{host}:{port}"
    typer.echo(f"FPL Assistant → {url}")
    typer.echo("  Ctrl+C to stop")
    if open_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    serve(host=host, port=port)


if __name__ == "__main__":
    app()
