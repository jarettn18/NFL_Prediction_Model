"""Sportsbook lines from The Odds API (https://the-odds-api.com), snapshotted per week.

Every fetch is appended to data/odds/{season}_week{wk}.parquet with one row per game per book, so the
first snapshot of a game at a book is the earliest line we saw there (the "opening" line, if the agent
runs soon after lines post).
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pandas as pd
import requests

from nflpred.config import BETTING, ODDS_DIR, ODDS_TEAM_NAMES, TEAM_ALIASES

API_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
LINE_COLUMNS = ["spread_line", "spread_home_price", "spread_away_price", "total_line", "over_price", "under_price",
                "home_moneyline", "away_moneyline"]


def team_code(name: str) -> str:
    code = ODDS_TEAM_NAMES[name]
    return TEAM_ALIASES.get(code, code)


def parse_odds(events: list[dict], books: list[str] = BETTING["books"],
               fetched_at: datetime | None = None) -> pd.DataFrame:
    """One row per game per book from an Odds API /odds response. spread_line follows nflverse: + = home favored."""
    fetched_at = fetched_at or datetime.now(timezone.utc)
    rows = []
    for ev in events:
        home, away = ev["home_team"], ev["away_team"]
        for bm in ev.get("bookmakers", []):
            if bm["key"] not in books:
                continue
            row = {"event_id": ev["id"], "home_team": team_code(home), "away_team": team_code(away),
                   "commence_time": pd.Timestamp(ev["commence_time"]), "book": bm["key"],
                   "fetched_at": pd.Timestamp(fetched_at), **{c: float("nan") for c in LINE_COLUMNS}}
            for m in bm.get("markets", []):
                out = {o["name"]: o for o in m["outcomes"]}
                if m["key"] == "h2h":
                    row["home_moneyline"], row["away_moneyline"] = out[home]["price"], out[away]["price"]
                elif m["key"] == "spreads":
                    row["spread_line"] = -out[home]["point"]
                    row["spread_home_price"], row["spread_away_price"] = out[home]["price"], out[away]["price"]
                elif m["key"] == "totals":
                    row["total_line"] = out["Over"]["point"]
                    row["over_price"], row["under_price"] = out["Over"]["price"], out["Under"]["price"]
            rows.append(row)
    return pd.DataFrame(rows)


def fetch_odds(books: list[str] = BETTING["books"], api_key: str | None = None) -> tuple[pd.DataFrame, str | None]:
    """Current lines at each book for every upcoming game. Returns (lines, remaining API credits).
    Up to 10 books cost the same as one: 3 credits for the three markets."""
    api_key = api_key or os.environ.get("ODDS_API_KEY")
    if not api_key:
        raise RuntimeError("Set the ODDS_API_KEY environment variable (free key at the-odds-api.com).")
    resp = requests.get(API_URL, timeout=30, params={
        "apiKey": api_key, "regions": "us", "markets": "h2h,spreads,totals",
        "bookmakers": ",".join(books), "oddsFormat": "american"})
    resp.raise_for_status()
    return parse_odds(resp.json(), books), resp.headers.get("x-requests-remaining")


def match_games(lines: pd.DataFrame, games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach game_id/season/week from the feature table. Returns (matched, unmatched)."""
    if lines.empty:
        return lines.assign(game_id=pd.Series(dtype=str)), lines
    # Schedule kickoffs are naive Eastern time.
    et_date = lines["commence_time"].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    sched = games[["game_id", "season", "week", "home_team", "away_team", "kickoff"]].copy()
    sched["date"] = pd.to_datetime(sched["kickoff"]).dt.normalize()
    merged = lines.assign(date=et_date).merge(sched, on=["home_team", "away_team"], how="left")
    merged = merged[(merged["date_x"] - merged["date_y"]).abs() <= pd.Timedelta(days=1)]
    matched = merged.drop(columns=["date_x", "date_y", "kickoff"]).drop_duplicates(["event_id", "book"])
    unmatched = lines[~lines["event_id"].isin(matched["event_id"])]
    return matched.reset_index(drop=True), unmatched


def _path(season: int, week: int):
    return ODDS_DIR / f"{season}_week{week:02d}.parquet"


def snapshot(matched: pd.DataFrame) -> list:
    """Append matched lines to each week's snapshot file."""
    ODDS_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for (season, week), g in matched.groupby(["season", "week"]):
        path = _path(int(season), int(week))
        if path.exists():
            g = pd.concat([pd.read_parquet(path), g], ignore_index=True)
        g.to_parquet(path, index=False)
        paths.append(path)
    return paths


def load_snapshots(season: int, week: int) -> pd.DataFrame:
    path = _path(season, week)
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _pick(season: int, week: int, first: bool) -> pd.DataFrame:
    snaps = load_snapshots(season, week)
    if snaps.empty:
        return snaps
    # A market can be missing from an early snapshot; take each book's first/last snapshot that has both lines.
    snaps = snaps.dropna(subset=["spread_line", "total_line"]).sort_values("fetched_at")
    by_book = snaps.groupby(["game_id", "book"])
    return by_book.head(1) if first else by_book.tail(1)


def opening_lines(season: int, week: int) -> pd.DataFrame:
    return _pick(season, week, first=True).reset_index(drop=True)


def latest_lines(season: int, week: int) -> pd.DataFrame:
    return _pick(season, week, first=False).reset_index(drop=True)


def reference_lines(offers: pd.DataFrame, ref_book: str = BETTING["ref_book"]) -> pd.DataFrame:
    """One line per game to feed the model: `ref_book`'s, or the median across books where it has none."""
    ref = offers[offers["book"] == ref_book]
    rest = offers[~offers["game_id"].isin(ref["game_id"])]
    median = rest.groupby("game_id").agg({**{c: "median" for c in LINE_COLUMNS}, "fetched_at": "max"}).reset_index()
    return pd.concat([ref, median.assign(book="median")], ignore_index=True)
