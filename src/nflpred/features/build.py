"""Assemble one row per game (home-team perspective) with all features and targets."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import norm

from nflpred.config import MARGIN_SIGMA
from nflpred.features.injuries import injury_impact
from nflpred.features.qb import qb_game_stats, qb_ratings
from nflpred.features.situational import situational_features
from nflpred.features.team_strength import METRICS, elo_ratings, rolling_ratings, team_game_stats
from nflpred.weather import game_weather, kickoff_et, load_stadiums

MARKET_FEATURES = ["spread_line", "total_line", "mkt_home_prob"]

FOOTBALL_FEATURES = [
    # team strength
    "elo_diff", "exp_margin_epa", "exp_total_epa", "exp_home_off", "exp_away_off",
    "exp_home_pass", "exp_away_pass", "exp_home_rush", "exp_away_rush",
    "home_off_success", "away_off_success", "home_def_success", "away_def_success",
    "pts_margin", "pts_total", "pace_sum",
    # quarterbacks
    "home_qb_rating", "away_qb_rating", "qb_diff", "home_qb_exp", "away_qb_exp", "home_qb_new", "away_qb_new",
    # injuries
    "home_inj_off", "home_inj_def", "away_inj_off", "away_inj_def",
    # situation and location
    "home_rest", "away_rest", "rest_diff", "home_travel_km", "away_travel_km", "away_tz_shift",
    "home_body_clock", "away_body_clock", "div_game", "neutral", "primetime", "playoff", "week", "high_altitude",
    # weather
    "temp_f", "wind_mph", "precip_in", "is_indoor",
]

ALL_FEATURES = MARKET_FEATURES + FOOTBALL_FEATURES

ID_COLUMNS = ["game_id", "season", "week", "game_type", "gameday", "gametime", "kickoff",
              "home_team", "away_team", "home_qb_name", "away_qb_name", "stadium"]
TARGETS = ["home_score", "away_score", "result", "total", "home_win"]


def moneyline_prob(ml: pd.Series) -> pd.Series:
    ml = ml.astype(float)
    return pd.Series(np.where(ml > 0, 100 / (ml + 100), -ml / (-ml + 100)), index=ml.index)


def market_home_prob(games: pd.DataFrame) -> pd.Series:
    """Vig-free home win probability from moneylines, falling back to the spread."""
    p_home, p_away = moneyline_prob(games["home_moneyline"]), moneyline_prob(games["away_moneyline"])
    from_ml = p_home / (p_home + p_away)
    from_spread = pd.Series(norm.cdf(games["spread_line"].astype(float) / MARGIN_SIGMA), index=games.index)
    return from_ml.fillna(from_spread)


def prepare_games(schedules: pl.DataFrame) -> pd.DataFrame:
    games = schedules.to_pandas()
    games["kickoff"] = [kickoff_et(d, t) for d, t in zip(games["gameday"], games["gametime"])]
    return games.sort_values(["kickoff", "game_id"]).reset_index(drop=True)


def _side_frame(df: pd.DataFrame, games: pd.DataFrame, side: str, cols: list[str]) -> pd.DataFrame:
    """Turn a per-(game_id, team) frame into home_/away_ prefixed columns."""
    keyed = games[["game_id", f"{side}_team"]].merge(
        df, left_on=["game_id", f"{side}_team"], right_on=["game_id", "team"], how="left")
    return keyed[["game_id"] + cols].rename(columns={c: f"{side}_{c}" for c in cols})


def build_dataset(data: dict[str, pl.DataFrame], today: date | None = None, verbose: bool = True) -> pd.DataFrame:
    log = print if verbose else (lambda *a, **k: None)
    games = prepare_games(data["schedules"])
    stadiums = load_stadiums()

    log("  team ratings...")
    ratings = rolling_ratings(games, team_game_stats(data["pbp"]))
    elo = elo_ratings(games)
    log("  quarterbacks...")
    qbs = qb_ratings(games, qb_game_stats(data["pbp"]))
    log("  injuries...")
    inj = injury_impact(data["injuries"], data["snap_counts"], data["players"], games)
    log("  situational...")
    sit = situational_features(games, stadiums)
    log("  weather...")
    wx = game_weather(games[["game_id", "stadium_id", "roof", "gameday", "gametime"]], today=today, verbose=verbose)

    df = games[ID_COLUMNS + ["home_score", "away_score", "result", "total", "location",
                             "spread_line", "total_line", "home_moneyline", "away_moneyline"]].copy()
    df["mkt_home_prob"] = market_home_prob(df)
    for side in ("home", "away"):
        df = df.merge(_side_frame(ratings, games, side, METRICS), on="game_id", how="left")
        df = df.merge(_side_frame(inj, games, side, ["inj_off", "inj_def"]), on="game_id", how="left")
        df[[f"{side}_inj_off", f"{side}_inj_def"]] = df[[f"{side}_inj_off", f"{side}_inj_def"]].fillna(0.0)
    df = df.merge(elo, on="game_id", how="left").merge(qbs, on="game_id", how="left")
    df = df.merge(sit.drop(columns=["week"]), on="game_id", how="left").merge(wx, on="game_id", how="left")

    # Matchup features: def_* ratings are EPA allowed, so offense + opposing defense = expected offense.
    df["elo_diff"] = df["elo_home"] - df["elo_away"]
    df["exp_home_off"] = df["home_off_epa"] + df["away_def_epa"]
    df["exp_away_off"] = df["away_off_epa"] + df["home_def_epa"]
    df["exp_home_pass"] = df["home_off_pass_epa"] + df["away_def_pass_epa"]
    df["exp_away_pass"] = df["away_off_pass_epa"] + df["home_def_pass_epa"]
    df["exp_home_rush"] = df["home_off_rush_epa"] + df["away_def_rush_epa"]
    df["exp_away_rush"] = df["away_off_rush_epa"] + df["home_def_rush_epa"]
    df["exp_margin_epa"] = df["exp_home_off"] - df["exp_away_off"]
    df["exp_total_epa"] = df["exp_home_off"] + df["exp_away_off"]
    exp_home_pts = (df["home_points_for"] + df["away_points_against"]) / 2
    exp_away_pts = (df["away_points_for"] + df["home_points_against"]) / 2
    df["pts_margin"] = exp_home_pts - exp_away_pts
    df["pts_total"] = exp_home_pts + exp_away_pts
    df["pace_sum"] = df["home_plays"] + df["away_plays"]
    df["qb_diff"] = df["home_qb_rating"] - df["away_qb_rating"]

    df["home_win"] = np.where(df["result"].isna(), np.nan, np.where(df["result"] > 0, 1.0, np.where(df["result"] < 0, 0.0, 0.5)))
    return df
