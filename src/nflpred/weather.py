"""Game-time weather from Open-Meteo (free, no API key).

Historical games use the ERA5-based archive API; recent and upcoming games use the forecast API.
Both are requested in America/New_York time so hours line up with nflverse `gametime` (ET).
Using the same source for training and live predictions keeps the features consistent.
"""

from __future__ import annotations

import time
from datetime import date, timedelta

import pandas as pd
import requests

from nflpred.config import CACHE_DIR, STADIUMS_CSV

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY_VARS = "temperature_2m,wind_speed_10m,precipitation"
ARCHIVE_LAG_DAYS = 7  # the archive trails real time by a few days
FORECAST_HORIZON_DAYS = 15
GAME_HOURS = 4  # average conditions over kickoff and the following three hours

INDOOR_TEMP_F = 70.0
INDOOR_ROOF_TYPES = {"dome", "retractable"}


def load_stadiums() -> pd.DataFrame:
    return pd.read_csv(STADIUMS_CSV)


REQUEST_SPACING_S = 0.15  # stay under Open-Meteo's free-tier per-minute limit
_last_request = 0.0


def _request(url: str, params: dict, retries: int = 5) -> dict:
    global _last_request
    for attempt in range(retries):
        wait = REQUEST_SPACING_S - (time.time() - _last_request)
        if wait > 0:
            time.sleep(wait)
        _last_request = time.time()
        try:
            resp = requests.get(url, params=params, timeout=60)
            if resp.status_code == 429 and attempt < retries - 1:
                time.sleep(65)  # per-minute quota exhausted; wait for the window to reset
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt * 2)
    raise RuntimeError("unreachable")


def _hourly_frame(payload: dict) -> pd.DataFrame:
    h = payload["hourly"]
    return pd.DataFrame({
        "time": pd.to_datetime(h["time"]),
        "temp_f": h["temperature_2m"],
        "wind_mph": h["wind_speed_10m"],
        "precip_in": h["precipitation"],
    })


def _base_params(lat: float, lon: float) -> dict:
    return {
        "latitude": lat,
        "longitude": lon,
        "hourly": HOURLY_VARS,
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": "America/New_York",
    }


def fetch_archive(lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
    params = _base_params(lat, lon) | {"start_date": start.isoformat(), "end_date": end.isoformat()}
    return _hourly_frame(_request(ARCHIVE_URL, params))


def _date_runs(days: list[date]) -> list[tuple[date, date]]:
    """Group sorted dates into runs of consecutive days (e.g. Sat+Sun doubleheaders)."""
    runs: list[tuple[date, date]] = []
    for d in days:
        if runs and d - runs[-1][1] == timedelta(days=1):
            runs[-1] = (runs[-1][0], d)
        else:
            runs.append((d, d))
    return runs


def fetch_forecast(lat: float, lon: float) -> pd.DataFrame:
    params = _base_params(lat, lon) | {"past_days": 30, "forecast_days": 16}
    return _hourly_frame(_request(FORECAST_URL, params))


def _archive_for_stadium(stadium_id: str, lat: float, lon: float, dates: list[date]) -> pd.DataFrame:
    """Hourly archive weather covering `dates`, cached per stadium and extended when needed.

    Only game days are requested: Open-Meteo weights calls by data volume, so long ranges burn quota.
    """
    path = CACHE_DIR / f"weather_{stadium_id}.parquet"
    cached = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["time"])
    have = set(pd.to_datetime(cached["time"]).dt.date) if len(cached) else set()
    missing = sorted(d for d in dates if d not in have)
    fresh = []
    try:
        for start, end in _date_runs(missing):
            fresh.append(fetch_archive(lat, lon, start, end))
    finally:  # keep whatever was fetched even if a later request fails
        if fresh:
            cached = pd.concat([cached, *fresh], ignore_index=True).drop_duplicates("time", keep="last")
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cached.sort_values("time").to_parquet(path, index=False)
    return cached


def _window(hourly: pd.DataFrame, kickoff: pd.Timestamp) -> dict:
    kickoff = kickoff.floor("h")
    w = hourly[(hourly["time"] >= kickoff) & (hourly["time"] < kickoff + pd.Timedelta(hours=GAME_HOURS))]
    if w.empty or w["temp_f"].isna().all():
        return {"temp_f": float("nan"), "wind_mph": float("nan"), "precip_in": float("nan")}
    return {"temp_f": w["temp_f"].mean(), "wind_mph": w["wind_mph"].mean(), "precip_in": w["precip_in"].sum()}


def kickoff_et(gameday: str, gametime: str | None) -> pd.Timestamp:
    hour, minute = 13, 0
    if isinstance(gametime, str) and ":" in gametime:
        hour, minute = (int(x) for x in gametime.split(":")[:2])
    return pd.Timestamp(gameday) + pd.Timedelta(hours=hour, minutes=minute)


def is_indoor(stadium_id: str | None, schedule_roof: str | None, stadiums: pd.DataFrame) -> bool:
    """Retractable roofs count as indoor: their status is unknown before kickoff and they usually close in bad weather."""
    row = stadiums.loc[stadiums["stadium_id"] == stadium_id]
    if len(row):
        return row["roof_type"].iloc[0] in INDOOR_ROOF_TYPES
    return schedule_roof in ("dome", "closed")


def game_weather(games: pd.DataFrame, today: date | None = None, verbose: bool = True) -> pd.DataFrame:
    """Weather for each game. `games` needs game_id, stadium_id, roof, gameday, gametime.

    Returns game_id, temp_f, wind_mph, precip_in, is_indoor. Indoor games get neutral values.
    Outdoor games without data (too far in the future, or API failure) get NaN, which LightGBM handles.
    """
    today = today or date.today()
    stadiums = load_stadiums().set_index("stadium_id", drop=False)
    out = []
    games = games.copy()
    games["kickoff"] = [kickoff_et(d, t) for d, t in zip(games["gameday"], games["gametime"])]
    games["indoor"] = [is_indoor(s, r, stadiums) for s, r in zip(games["stadium_id"], games["roof"])]

    for g in games[games["indoor"]].itertuples():
        out.append({"game_id": g.game_id, "temp_f": INDOOR_TEMP_F, "wind_mph": 0.0, "precip_in": 0.0, "is_indoor": 1})

    outdoor = games[~games["indoor"]]
    for stadium_id, grp in outdoor.groupby("stadium_id"):
        if stadium_id not in stadiums.index:
            for g in grp.itertuples():
                out.append({"game_id": g.game_id, "temp_f": float("nan"), "wind_mph": float("nan"),
                            "precip_in": float("nan"), "is_indoor": 0})
            continue
        lat, lon = stadiums.loc[stadium_id, ["lat", "lon"]]
        days = grp["kickoff"].dt.date
        archive_days = sorted({d for d in days if d <= today - timedelta(days=ARCHIVE_LAG_DAYS)})
        recent = [d for d in days if d > today - timedelta(days=ARCHIVE_LAG_DAYS)]
        hourly = []
        try:
            if archive_days:
                hourly.append(_archive_for_stadium(stadium_id, lat, lon, archive_days))
            if recent and min(recent) <= today + timedelta(days=FORECAST_HORIZON_DAYS):
                hourly.append(fetch_forecast(lat, lon))
        except requests.RequestException as exc:
            if verbose:
                print(f"  warning: weather fetch failed for {stadium_id}: {exc}")
        hourly_df = pd.concat(hourly, ignore_index=True) if hourly else pd.DataFrame(columns=["time", "temp_f"])
        hourly_df["time"] = pd.to_datetime(hourly_df["time"])
        for g in grp.itertuples():
            out.append({"game_id": g.game_id, **_window(hourly_df, g.kickoff), "is_indoor": 0})

    return pd.DataFrame(out)
