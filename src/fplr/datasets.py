"""Normalising both data sources into one per-fixture table.

The live API only serves the current season, so history comes from the community
archive. Conveniently `element-summary/{id}/history` and the archive's
`merged_gw.csv` carry the same field names, so the two sources need renaming rather
than reconciling.

Two identity problems have to be solved before the seasons can be stacked:

* **Player ids are season-specific.** FPL reassigns `element` ids every season. The
  stable key is `code`, which the archive exposes in `players_raw.csv` and the live
  API in the bootstrap. Every row is keyed on `player_code`.
* **Team ids are season-specific too**, and change as clubs are promoted and
  relegated. `team_code` is stable and is carried alongside the season-local id.

The output grain is one row per player per fixture -- not per gameweek -- because
the goals-conceded and saves rules both round down, so a double gameweek scored on
aggregated stats produces the wrong answer.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

from .config import CURRENT_SEASON, PROCESSED_DIR, RAW_DIR, TRAINING_SEASONS
from .ingest import USER_AGENT, finalised_gameweeks, read_history

ARCHIVE_BASE = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
)

#: Per-fixture statistics, identical in both sources.
STAT_COLUMNS = [
    "minutes",
    "starts",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "own_goals",
    "penalties_saved",
    "penalties_missed",
    "yellow_cards",
    "red_cards",
    "saves",
    "bonus",
    "bps",
    "clearances_blocks_interceptions",
    "tackles",
    "recoveries",
    "defensive_contribution",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "value",
    "total_points",
]

IDENTITY_COLUMNS = [
    "season",
    "finalised",
    "player_code",
    "element",
    "web_name",
    "position",
    "team_code",
    "team",
    "opponent_team",
    "opponent_team_code",
    "gameweek",
    "fixture",
    "was_home",
    "kickoff_time",
]

CANONICAL_COLUMNS = IDENTITY_COLUMNS + STAT_COLUMNS

#: The archive stores position as a short string; the API uses `element_type`.
POSITION_CODES = {"GK": 1, "GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}


def _archive_cache_path(season: str, filename: str, *, root: Path = RAW_DIR) -> Path:
    return root / "archive" / season / filename


def fetch_archive_file(
    season: str, filename: str, *, root: Path = RAW_DIR, refresh: bool = False
) -> pd.DataFrame:
    """Download an archive CSV, caching it on disk.

    `filename` may include a subdirectory: the per-gameweek file lives under
    `gws/`, while the player and team metadata sit at the season root.

    Completed seasons never change, so the cache is permanent for them; the current
    season's archive files are refreshed on request.
    """
    path = _archive_cache_path(season, filename, root=root)
    if refresh or not path.exists():
        url = f"{ARCHIVE_BASE}/{season}/{filename}"
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
    raw = path.read_bytes()
    try:
        return pd.read_csv(io.BytesIO(raw))
    except UnicodeDecodeError:
        return pd.read_csv(io.BytesIO(raw), encoding="latin-1")


FIXTURE_COLUMNS = [
    "season",
    "fixture",
    "gameweek",
    "kickoff_time",
    "team_h",
    "team_a",
    "team_h_code",
    "team_a_code",
    "team_h_score",
    "team_a_score",
    "team_h_difficulty",
    "team_a_difficulty",
    "finished",
]


def load_fixtures(season: str, *, root: Path = RAW_DIR) -> pd.DataFrame:
    """The season's match list: who played whom, when, and the result.

    This is the authority for which club a player turned out for in a given match.
    Player-level snapshots such as `players_raw.csv` record only a player's *current*
    club, so using them would stamp a January transfer onto the previous August's
    fixtures.
    """
    if season == CURRENT_SEASON:
        from .ingest import latest_snapshot

        snapshot = latest_snapshot(root=root)
        if snapshot is None:
            raise FileNotFoundError("no snapshot on disk; run `fpl sync` first")
        raw = pd.DataFrame(snapshot.read("fixtures"))
        team_codes = {t["id"]: t["code"] for t in snapshot.read("bootstrap")["teams"]}
    else:
        raw = fetch_archive_file(season, "fixtures.csv", root=root)
        teams = fetch_archive_file(season, "teams.csv", root=root)
        team_codes = dict(zip(teams["id"], teams["code"]))

    frame = raw.rename(columns={"id": "fixture", "event": "gameweek"})
    frame["season"] = season
    frame["team_h_code"] = frame["team_h"].map(team_codes)
    frame["team_a_code"] = frame["team_a"].map(team_codes)
    frame["kickoff_time"] = pd.to_datetime(frame["kickoff_time"], errors="coerce", utc=True)

    for column in FIXTURE_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame = frame[FIXTURE_COLUMNS].copy()
    for column in ("fixture", "gameweek", "team_h", "team_a", "team_h_code",
                   "team_a_code", "team_h_score", "team_a_score",
                   "team_h_difficulty", "team_a_difficulty"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    return frame.sort_values(["gameweek", "fixture"]).reset_index(drop=True)


def _attach_team_from_fixture(
    frame: pd.DataFrame, season: str, *, root: Path = RAW_DIR
) -> pd.DataFrame:
    """Set each row's club from the fixture it was played in.

    A player's club is whichever side of the match they were not the opponent of:
    the home side when `was_home`, the away side otherwise. This is exact and
    survives mid-season transfers, which club-level player snapshots do not.
    """
    fixtures = load_fixtures(season, root=root)
    lookup = fixtures.set_index("fixture")[["team_h", "team_a", "team_h_code", "team_a_code"]]

    joined = frame.join(lookup, on="fixture", rsuffix="_fx")
    home = joined["was_home"].astype("boolean").fillna(False)
    frame = frame.copy()
    frame["team"] = joined["team_h"].where(home, joined["team_a"])
    frame["team_code"] = joined["team_h_code"].where(home, joined["team_a_code"])
    frame["opponent_team"] = joined["team_a"].where(home, joined["team_h"])
    frame["opponent_team_code"] = joined["team_a_code"].where(home, joined["team_h_code"])
    return frame


def _season_lookups(season: str, *, root: Path = RAW_DIR) -> tuple[pd.DataFrame, dict]:
    """Player metadata for a season, plus the season-local team id to code map."""
    players = fetch_archive_file(season, "players_raw.csv", root=root)
    teams = fetch_archive_file(season, "teams.csv", root=root)
    team_codes = dict(zip(teams["id"], teams["code"]))
    # Deliberately excludes club: `players_raw` records a player's *final* club for
    # the season, which is wrong for anyone transferred mid-season. Club is taken
    # from the fixture instead. Position is kept from here -- it is stable within a
    # season, and the full-season scoring replay would fail loudly if it were not,
    # since goal and clean-sheet values differ by position.
    meta = players[["id", "code", "element_type", "web_name"]].rename(
        columns={"id": "element", "code": "player_code", "element_type": "position"}
    )
    return meta, team_codes


def load_archive_season(season: str, *, root: Path = RAW_DIR) -> pd.DataFrame:
    """One season of per-fixture rows from the community archive."""
    gameweeks = fetch_archive_file(season, "gws/merged_gw.csv", root=root)
    meta, team_codes = _season_lookups(season, root=root)

    frame = gameweeks.merge(meta, on="element", how="left", suffixes=("_gw", ""))

    # players_raw is authoritative for position and club; merged_gw carries display
    # strings that do not join cleanly across seasons.
    missing_position = frame["position"].isna()
    if missing_position.any():
        fallback = frame.loc[missing_position, "position_gw"].map(POSITION_CODES)
        frame.loc[missing_position, "position"] = fallback

    frame["season"] = season
    frame["gameweek"] = frame["GW"]
    frame["finalised"] = True  # archived seasons are complete by definition
    frame = _attach_team_from_fixture(frame, season, root=root)
    return _finalise(frame)


def load_live_season(season: str = CURRENT_SEASON, *, root: Path = RAW_DIR) -> pd.DataFrame:
    """Per-fixture rows for the current season from the cached history files.

    Reads the snapshot's bootstrap for player and team identity so the result keys
    on the same stable codes as the archive seasons.
    """
    from .ingest import latest_snapshot

    snapshot = latest_snapshot(root=root)
    if snapshot is None:
        raise FileNotFoundError("no snapshot on disk; run `fpl sync` first")
    bootstrap = snapshot.read("bootstrap")

    team_codes = {team["id"]: team["code"] for team in bootstrap["teams"]}
    meta = {
        element["id"]: {
            "player_code": element["code"],
            "position": element["element_type"],
            "web_name": element["web_name"],
        }
        for element in bootstrap["elements"]
    }

    rows = []
    for element_id, info in meta.items():
        history = read_history(season, element_id, root=root)
        if not history:
            continue
        for row in history:
            rows.append({**row, **info, "element": element_id})

    if not rows:
        raise FileNotFoundError(
            f"no cached histories for {season}; run `fpl sync --histories` first"
        )

    frame = pd.DataFrame(rows)
    frame["season"] = season
    frame["gameweek"] = frame["round"]
    frame = _attach_team_from_fixture(frame, season, root=root)

    # `element-summary` returns rows for gameweeks that are still in progress. Their
    # bonus points and defensive contributions are provisional and will be revised,
    # so they must never reach the training target -- but they are exactly what we
    # want as *features* when scoring the upcoming gameweek. Mark rather than drop.
    settled = set(finalised_gameweeks(bootstrap))
    frame["finalised"] = frame["gameweek"].isin(settled)
    return _finalise(frame)


def _finalise(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce to the canonical schema, types and sort order."""
    for column in CANONICAL_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA

    frame = frame[CANONICAL_COLUMNS].copy()
    frame["kickoff_time"] = pd.to_datetime(frame["kickoff_time"], errors="coerce", utc=True)
    frame["was_home"] = frame["was_home"].astype("boolean")
    frame["finalised"] = frame["finalised"].astype("boolean")

    integer_stats = [
        c for c in STAT_COLUMNS if c not in {
            "expected_goals",
            "expected_assists",
            "expected_goal_involvements",
            "expected_goals_conceded",
            "influence",
            "creativity",
            "threat",
            "ict_index",
        }
    ]
    for column in integer_stats:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    for column in set(STAT_COLUMNS) - set(integer_stats):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")

    for column in ("player_code", "element", "position", "team_code", "team",
                   "opponent_team", "opponent_team_code", "gameweek", "fixture"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")

    # The archive ships a handful of exactly-duplicated rows (10 in 2025/26). They
    # are byte-identical repeats rather than genuine second fixtures, so keeping both
    # would double-count those players' minutes and points in every rolling feature.
    key = ["season", "player_code", "gameweek", "fixture"]
    frame = frame.drop_duplicates(subset=key, keep="first")

    return frame.sort_values(["season", "gameweek", "player_code"]).reset_index(drop=True)


def build_player_fixtures(
    seasons: tuple[str, ...] = TRAINING_SEASONS,
    *,
    root: Path = RAW_DIR,
    out_dir: Path = PROCESSED_DIR,
    write: bool = True,
) -> pd.DataFrame:
    """Stack every training season into one per-fixture table and persist it."""
    frames = []
    for season in seasons:
        loader = load_live_season if season == CURRENT_SEASON else load_archive_season
        try:
            frames.append(loader(season, root=root))
        except (FileNotFoundError, requests.HTTPError) as error:
            print(f"  skipping {season}: {error}")

    if not frames:
        raise RuntimeError("no seasons could be loaded")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["season", "gameweek", "player_code"]).reset_index(drop=True)

    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(out_dir / "player_fixtures.parquet", index=False)
    return combined


def load_player_fixtures(
    *,
    out_dir: Path = PROCESSED_DIR,
    finalised_only: bool = True,
) -> pd.DataFrame:
    """Read the built per-fixture table.

    Defaults to finalised rows only. Training must never see a provisional
    gameweek: bonus points and defensive contributions are still being revised, so
    those rows would inject noise straight into the target. Pass
    `finalised_only=False` when building features to score an upcoming gameweek,
    where the most recent in-progress results are legitimately useful signal.
    """
    path = out_dir / "player_fixtures.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not built; run `fpl data build` first")
    frame = pd.read_parquet(path)
    if finalised_only:
        frame = frame[frame["finalised"].fillna(False)].reset_index(drop=True)
    return frame
