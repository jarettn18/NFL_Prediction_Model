"""Game context: rest, travel, time zones, body clock, altitude, divisional / primetime / neutral site."""

from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0
ET_OFFSET = -5
HIGH_ALTITUDE_M = 1500


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def home_stadiums(games: pd.DataFrame) -> pd.DataFrame:
    """Each team's primary home stadium per season (most frequent non-neutral home venue)."""
    home = games[(games["location"] == "Home") & games["stadium_id"].notna()]
    modal = home.groupby(["season", "home_team"])["stadium_id"].agg(lambda s: s.value_counts().index[0])
    return modal.rename("home_stadium_id").reset_index().rename(columns={"home_team": "team"})


def situational_features(games: pd.DataFrame, stadiums: pd.DataFrame) -> pd.DataFrame:
    st = stadiums.set_index("stadium_id")
    hs = home_stadiums(games)
    df = games[["game_id", "season", "week", "game_type", "location", "stadium_id", "kickoff",
                "home_team", "away_team", "home_rest", "away_rest", "div_game"]].copy()

    for side in ("home", "away"):
        df = df.merge(hs.rename(columns={"team": f"{side}_team", "home_stadium_id": f"{side}_base"}),
                      on=["season", f"{side}_team"], how="left")
    venue = df["stadium_id"].where(df["stadium_id"].isin(st.index), df["home_base"])

    def lookup(ids: pd.Series, col: str) -> np.ndarray:
        return ids.map(st[col]).to_numpy(dtype=float)

    v_lat, v_lon, v_tz = lookup(venue, "lat"), lookup(venue, "lon"), lookup(venue, "tz_offset")
    kick_hour_et = df["kickoff"].dt.hour.to_numpy()
    out = pd.DataFrame({"game_id": df["game_id"]})
    for side in ("home", "away"):
        b_lat, b_lon, b_tz = (lookup(df[f"{side}_base"], c) for c in ("lat", "lon", "tz_offset"))
        out[f"{side}_travel_km"] = haversine_km(b_lat, b_lon, v_lat, v_lon)
        out[f"{side}_tz_shift"] = v_tz - b_tz
        # Kickoff time on the team's home body clock (e.g. 10:00 for a west-coast team in a 1pm ET game).
        out[f"{side}_body_clock"] = kick_hour_et + (b_tz - ET_OFFSET)

    out["home_rest"] = df["home_rest"].to_numpy(dtype=float)
    out["away_rest"] = df["away_rest"].to_numpy(dtype=float)
    out["rest_diff"] = out["home_rest"] - out["away_rest"]
    out["div_game"] = df["div_game"].fillna(0).astype(int).to_numpy()
    out["neutral"] = (df["location"] == "Neutral").astype(int).to_numpy()
    out["primetime"] = (kick_hour_et >= 19).astype(int)
    out["playoff"] = (df["game_type"] != "REG").astype(int).to_numpy()
    out["week"] = df["week"].to_numpy()
    out["high_altitude"] = (lookup(venue, "elevation_m") > HIGH_ALTITUDE_M).astype(int)
    return out
