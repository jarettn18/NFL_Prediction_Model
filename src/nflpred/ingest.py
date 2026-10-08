"""Download nflverse data via nflreadpy, slimmed down and cached as parquet per season."""

from __future__ import annotations

import time
from typing import Callable

import nflreadpy as nfl
import polars as pl

from nflpred.config import CACHE_DIR, CURRENT_SEASON_CACHE_HOURS, FIRST_SEASON, TEAM_ALIASES, current_season

PBP_COLUMNS = [
    "game_id", "season", "week", "season_type", "game_date", "posteam", "defteam",
    "play_type", "pass", "rush", "epa", "success", "qb_dropback",
    "passer_player_id", "rusher_player_id",
]

# First season each dataset exists in nflverse.
FIRST_AVAILABLE = {"schedules": 1999, "pbp": 1999, "injuries": 2009, "snap_counts": 2012}

TEAM_COLUMNS = {
    "schedules": ["home_team", "away_team"],
    "pbp": ["posteam", "defteam"],
    "injuries": ["team"],
    "snap_counts": ["team", "opponent"],
}


def normalize_teams(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    return df.with_columns(pl.col(c).replace(TEAM_ALIASES) for c in columns if c in df.columns)


def _is_fresh(path, season: int) -> bool:
    if not path.exists():
        return False
    if season < current_season():
        return True
    return (time.time() - path.stat().st_mtime) < CURRENT_SEASON_CACHE_HOURS * 3600


def _fetch(name: str, season: int) -> pl.DataFrame:
    loaders: dict[str, Callable[[int], pl.DataFrame]] = {
        "schedules": lambda s: nfl.load_schedules([s]),
        "pbp": lambda s: nfl.load_pbp([s]).select(PBP_COLUMNS),
        "injuries": lambda s: nfl.load_injuries([s]),
        "snap_counts": lambda s: nfl.load_snap_counts([s]),
    }
    return loaders[name](season)


def load(name: str, seasons: list[int], refresh: bool = False) -> pl.DataFrame:
    """Load one dataset for several seasons, using the parquet cache where possible."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for season in seasons:
        if season < FIRST_AVAILABLE[name]:
            continue
        path = CACHE_DIR / f"{name}_{season}.parquet"
        if refresh or not _is_fresh(path, season):
            try:
                df = _fetch(name, season)
            except Exception as exc:  # a season may not be published yet (e.g. preseason)
                if path.exists():
                    df = pl.read_parquet(path)
                else:
                    print(f"  warning: could not load {name} {season}: {exc}")
                    continue
            df.write_parquet(path)
        frames.append(pl.read_parquet(path))
    if not frames:
        return pl.DataFrame()
    df = pl.concat(frames, how="diagonal_relaxed")
    return normalize_teams(df, TEAM_COLUMNS[name])


def load_players(refresh: bool = False) -> pl.DataFrame:
    """Player id crosswalk (gsis <-> pfr). Not season-specific."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / "players.parquet"
    stale = not path.exists() or (time.time() - path.stat().st_mtime) > 7 * 24 * 3600
    if refresh or stale:
        nfl.load_players().select("gsis_id", "pfr_id", "display_name", "position").write_parquet(path)
    return pl.read_parquet(path)


def load_all(seasons: list[int] | None = None, refresh_current: bool = False) -> dict[str, pl.DataFrame]:
    """Load every dataset the feature pipeline needs."""
    seasons = seasons or list(range(FIRST_SEASON, current_season() + 1))
    cur = current_season()
    out = {}
    for name in ("schedules", "pbp", "injuries", "snap_counts"):
        past = [s for s in seasons if s != cur]
        frames = [load(name, past)] if past else []
        if cur in seasons:
            frames.append(load(name, [cur], refresh=refresh_current))
        frames = [f for f in frames if f.height]
        out[name] = pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()
    out["players"] = load_players()
    return out
