"""Fetching FPL data and snapshotting it immutably.

Two ideas hold this module together:

*Snapshots are immutable.* Every sync writes a dated directory of gzipped raw JSON
under `data/raw/`. Training always reads from a named snapshot, never from the live
API, so any model version can be rebuilt exactly as it was fitted.

*Only finalised gameweeks are trainable.* FPL revises a gameweek after the final
whistle while Opta reviews the data -- bonus points and defensive contributions can
both move. The API exposes this directly as `data_checked`, which flips true once the
review completes. We train on `finished and data_checked` gameweeks only; anything
less is provisional and will silently poison the target.
"""

from __future__ import annotations

import gzip
import json
import re
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import requests

from .config import ARCHIVE_URL_TEMPLATE, CURRENT_SEASON, RAW_DIR

API_BASE = "https://fantasy.premierleague.com/api"

#: The API is unauthenticated and undocumented; throttle so we stay a good citizen.
MIN_REQUEST_INTERVAL = 1.0

USER_AGENT = "fplr/0.1 (personal FPL analytics; contact via github)"


class FPLClient:
    """Thin, rate-limited client for the public FPL endpoints."""

    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        min_interval: float = MIN_REQUEST_INTERVAL,
        timeout: float = 30.0,
    ) -> None:
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.min_interval = min_interval
        self.timeout = timeout
        self._last_request = 0.0

    def _get(self, path: str) -> Any:
        wait = self.min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        url = f"{API_BASE}/{path.lstrip('/')}"
        response = self.session.get(url, timeout=self.timeout)
        self._last_request = time.monotonic()
        response.raise_for_status()
        return response.json()

    # --- Endpoints ----------------------------------------------------------

    def bootstrap(self) -> dict:
        """Players, teams, positions and the gameweek calendar."""
        return self._get("bootstrap-static/")

    def fixtures(self) -> list[dict]:
        """All fixtures for the season, with difficulty and kickoff times."""
        return self._get("fixtures/")

    def live(self, gameweek: int) -> dict:
        """Per-player stat lines for one gameweek, with point attribution."""
        return self._get(f"event/{gameweek}/live/")

    def element_summary(self, player_id: int) -> dict:
        """One player's per-fixture history for the current season."""
        return self._get(f"element-summary/{player_id}/")

    def entry(self, entry_id: int) -> dict:
        """A manager's squad metadata: bank, value, chips, transfers."""
        return self._get(f"entry/{entry_id}/")

    def entry_picks(self, entry_id: int, gameweek: int) -> dict:
        """A manager's selected XI and bench for one gameweek."""
        return self._get(f"entry/{entry_id}/event/{gameweek}/picks/")


def finalised_gameweeks(bootstrap: dict) -> list[int]:
    """Gameweeks safe to train on, in order.

    A gameweek qualifies only once FPL has both finished it and completed the Opta
    data review (`data_checked`). Bonus points and defensive contributions are both
    still mutable before that.
    """
    return [
        event["id"]
        for event in bootstrap["events"]
        if event.get("finished") and event.get("data_checked")
    ]


def latest_finalised_gameweek(bootstrap: dict) -> int | None:
    """Highest finalised gameweek, or None if the season has not produced one."""
    gameweeks = finalised_gameweeks(bootstrap)
    return max(gameweeks) if gameweeks else None


def next_gameweek(bootstrap: dict) -> dict | None:
    """The gameweek we are about to pick a team for."""
    for event in bootstrap["events"]:
        if event.get("is_next"):
            return event
    return None


@dataclass(frozen=True)
class Snapshot:
    """An immutable dated capture of the raw API responses."""

    path: Path

    @property
    def name(self) -> str:
        return self.path.name

    def write(self, key: str, payload: Any) -> Path:
        self.path.mkdir(parents=True, exist_ok=True)
        target = self.path / f"{key}.json.gz"
        with gzip.open(target, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"))
        return target

    def read(self, key: str) -> Any:
        target = self.path / f"{key}.json.gz"
        if not target.exists():
            raise FileNotFoundError(f"{key!r} not in snapshot {self.name}")
        with gzip.open(target, "rt", encoding="utf-8") as handle:
            return json.load(handle)

    def has(self, key: str) -> bool:
        return (self.path / f"{key}.json.gz").exists()

    def keys(self) -> list[str]:
        return sorted(p.name.removesuffix(".json.gz") for p in self.path.glob("*.json.gz"))


def snapshot_for(day: date | None = None, *, root: Path = RAW_DIR) -> Snapshot:
    """The snapshot directory for a given day (today by default)."""
    return Snapshot(root / (day or date.today()).isoformat())


#: Snapshot directories are named by ISO date. The raw root also holds the archive
#: and per-fixture history caches, which must never be mistaken for a snapshot.
SNAPSHOT_DIR_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def latest_snapshot(*, root: Path = RAW_DIR) -> Snapshot | None:
    """Most recent snapshot on disk, or None if nothing has been synced."""
    candidates = sorted(
        p for p in root.glob("*") if p.is_dir() and SNAPSHOT_DIR_PATTERN.match(p.name)
    )
    return Snapshot(candidates[-1]) if candidates else None


def sync(
    *,
    client: FPLClient | None = None,
    snapshot: Snapshot | None = None,
    force: bool = False,
) -> dict:
    """Pull the current season into a dated snapshot.

    Fetches the bootstrap and fixtures every time (they change daily -- prices,
    injuries, availability), then the live stat lines for each finalised gameweek.
    Already-captured gameweeks are skipped unless `force`, since a finalised
    gameweek is by definition immutable.

    Returns a manifest describing what was written.
    """
    client = client or FPLClient()
    snapshot = snapshot or snapshot_for()

    bootstrap = client.bootstrap()
    snapshot.write("bootstrap", bootstrap)
    snapshot.write("fixtures", client.fixtures())

    finalised = finalised_gameweeks(bootstrap)
    fetched, skipped = [], []
    for gameweek in finalised:
        key = f"live_gw{gameweek:02d}"
        if snapshot.has(key) and not force:
            skipped.append(gameweek)
            continue
        snapshot.write(key, client.live(gameweek))
        fetched.append(gameweek)

    upcoming = next_gameweek(bootstrap)
    manifest = {
        "season": CURRENT_SEASON,
        "snapshot": snapshot.name,
        "finalised_gameweeks": finalised,
        "latest_finalised": max(finalised) if finalised else None,
        "next_gameweek": upcoming["id"] if upcoming else None,
        "next_deadline": upcoming["deadline_time"] if upcoming else None,
        "gameweeks_fetched": fetched,
        "gameweeks_skipped": skipped,
    }
    snapshot.write("manifest", manifest)
    return manifest


def archive_url(season: str) -> str:
    """URL of the community archive's merged gameweek file for a past season."""
    return ARCHIVE_URL_TEMPLATE.format(season=season)


# --- Per-fixture history cache ---------------------------------------------
# The live endpoint reports gameweek aggregates, but two scoring rules round down,
# so the model needs per-fixture rows. `element-summary` provides them with the same
# field names as the community archive, giving one schema across both seasons.
#
# A full pull is ~700 requests. Finalised gameweeks never change, so the cache is
# append-only: a player is refetched only when the season has progressed past the
# last round we hold for them.

HISTORY_DIR_NAME = "histories"


def history_cache_dir(season: str, *, root: Path = RAW_DIR) -> Path:
    return root / HISTORY_DIR_NAME / season


def _history_path(season: str, player_id: int, *, root: Path = RAW_DIR) -> Path:
    return history_cache_dir(season, root=root) / f"{player_id}.json.gz"


def read_history(season: str, player_id: int, *, root: Path = RAW_DIR) -> list[dict] | None:
    path = _history_path(season, player_id, root=root)
    if not path.exists():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _write_history(season: str, player_id: int, rows: list[dict], *, root: Path = RAW_DIR) -> None:
    path = _history_path(season, player_id, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(rows, handle, separators=(",", ":"))


def _is_stale(rows: list[dict] | None, latest_finalised: int) -> bool:
    """True if this player's cached history predates the latest finalised gameweek.

    A player with no rows at all is only stale while the season is young; once
    gameweeks have been played, an empty history means they genuinely have not
    featured, and refetching every such player each sync would dominate the run.
    """
    if rows is None:
        return True
    if not rows:
        return latest_finalised <= 1
    return max(row["round"] for row in rows) < latest_finalised


def sync_histories(
    *,
    client: FPLClient | None = None,
    bootstrap: dict | None = None,
    season: str = CURRENT_SEASON,
    root: Path = RAW_DIR,
    force: bool = False,
    progress: bool = False,
) -> dict:
    """Refresh the per-fixture history cache for the current season."""
    client = client or FPLClient()
    bootstrap = bootstrap or client.bootstrap()
    latest = latest_finalised_gameweek(bootstrap) or 0

    player_ids = [element["id"] for element in bootstrap["elements"]]
    fetched, cached, failed = [], [], []

    for index, player_id in enumerate(player_ids, start=1):
        rows = read_history(season, player_id, root=root)
        if not force and not _is_stale(rows, latest):
            cached.append(player_id)
            continue
        try:
            summary = client.element_summary(player_id)
        except requests.RequestException as error:  # transient; report, do not abort
            failed.append({"player": player_id, "error": str(error)})
            continue
        _write_history(season, player_id, summary["history"], root=root)
        fetched.append(player_id)
        if progress and index % 50 == 0:
            print(f"  {index}/{len(player_ids)} players", flush=True)

    return {
        "season": season,
        "latest_finalised": latest,
        "players_total": len(player_ids),
        "players_fetched": len(fetched),
        "players_cached": len(cached),
        "failures": failed,
    }
