"""Project paths and constants."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_dotenv(path: Path | None = None) -> dict[str, str]:
    """Read `.env` into the environment, without overriding what is already set.

    A real environment variable always wins over the file, which is the usual
    convention: it lets a one-off `ANTHROPIC_API_KEY=... fpl brief` override the
    stored value, and stops a stale file quietly shadowing a deliberate export.

    Deliberately hand-rolled rather than taking a dependency -- the file only ever
    holds a handful of `KEY=value` lines.
    """
    path = path or (PROJECT_ROOT / ".env")
    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key or not value:
            continue
        loaded[key] = value
        os.environ.setdefault(key, value)
    return loaded


#: Loaded on import so every entry point -- CLI, agent, tests -- sees the same config.
load_dotenv()

DATA_DIR = Path(os.environ.get("FPL_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = Path(os.environ.get("FPL_MODELS_DIR", PROJECT_ROOT / "models"))
REPORTS_DIR = Path(os.environ.get("FPL_REPORTS_DIR", PROJECT_ROOT / "reports"))

#: Season the live API is currently serving.
CURRENT_SEASON = "2026-27"

#: Seasons used for training. DEFCON scoring began in 2025/26, so earlier seasons
#: have a different target distribution and are deliberately excluded.
TRAINING_SEASONS = ("2025-26", "2026-27")

#: Historical gameweek data for completed seasons, which the live API no longer serves.
ARCHIVE_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master"
    "/data/{season}/gws/merged_gw.csv"
)

FPL_ENTRY_ID = os.environ.get("FPL_ENTRY_ID")
