"""Point-in-time team strength: Elo and exponentially weighted EPA / success / scoring ratings.

Every rating attached to a game is computed only from games that kicked off before it.
Unplayed (future) games receive the team's latest rating, so the same code serves training and prediction.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import polars as pl

METRICS = [
    "off_epa", "off_pass_epa", "off_rush_epa", "off_success",
    "def_epa", "def_pass_epa", "def_rush_epa", "def_success",
    "points_for", "points_against", "plays",
]


def team_game_stats(pbp: pl.DataFrame) -> pd.DataFrame:
    """Per (game_id, team) offensive and defensive efficiency from play-by-play."""
    plays = pbp.filter(((pl.col("pass") == 1) | (pl.col("rush") == 1)) & pl.col("epa").is_not_null())

    def side(team_col: str, prefix: str) -> pl.DataFrame:
        aggs = [
            pl.col("epa").mean().alias(f"{prefix}_epa"),
            pl.col("epa").filter(pl.col("pass") == 1).mean().alias(f"{prefix}_pass_epa"),
            pl.col("epa").filter(pl.col("rush") == 1).mean().alias(f"{prefix}_rush_epa"),
            pl.col("success").mean().alias(f"{prefix}_success"),
        ]
        if prefix == "off":
            aggs.append(pl.len().cast(pl.Float64).alias("plays"))
        return plays.group_by("game_id", team_col).agg(aggs).rename({team_col: "team"})

    return side("posteam", "off").join(side("defteam", "def"), on=["game_id", "team"], how="inner").to_pandas()


def team_games(games: pd.DataFrame) -> pd.DataFrame:
    """Two rows per game (one per team) in kickoff order, with points for/against."""
    base = ["game_id", "season", "kickoff"]
    home = games[base + ["home_team", "home_score", "away_score"]].rename(
        columns={"home_team": "team", "home_score": "points_for", "away_score": "points_against"})
    away = games[base + ["away_team", "away_score", "home_score"]].rename(
        columns={"away_team": "team", "away_score": "points_for", "home_score": "points_against"})
    tg = pd.concat([home, away], ignore_index=True)
    return tg.sort_values(["kickoff", "game_id", "team"]).reset_index(drop=True)


def rolling_ratings(
    games: pd.DataFrame,
    stats: pd.DataFrame,
    decay: float = 0.9,
    season_carry: float = 0.35,
    prior_games: float = 3.0,
    league_alpha: float = 0.002,
) -> pd.DataFrame:
    """Pregame EWMA rating for every metric and team-game.

    rating = (sum_w * x + prior_games * league_mean) / (sum_w + prior_games)
    Each new game decays old weights by `decay`. At a new season the accumulated weight is
    multiplied by `season_carry`, so early-season ratings lean toward the league mean.
    """
    tg = team_games(games).merge(stats, on=["game_id", "team"], how="left")
    played_mask = tg["off_epa"].notna()
    # League mean: a running average of games already played (cumulative at first, then EWMA).
    league = np.full(len(METRICS), np.nan)
    n_seen = 0

    state: dict[str, dict] = {}
    values = tg[METRICS].to_numpy(dtype=float)
    out = np.full((len(tg), len(METRICS)), np.nan)
    teams, seasons, played = tg["team"].to_numpy(), tg["season"].to_numpy(), played_mask.to_numpy()
    # Process one kickoff slot at a time: every lookup in the slot happens before any update,
    # so simultaneous games (including both sides of the same game) never see each other.
    for idx in tg.groupby("kickoff", sort=True).indices.values():
        for i in idx:
            st = state.get(teams[i])
            if st is None:
                st = state[teams[i]] = {"season": seasons[i], "sx": np.zeros(len(METRICS)), "sw": 0.0}
            if st["season"] != seasons[i]:
                st["sx"] *= season_carry
                st["sw"] *= season_carry
                st["season"] = seasons[i]
            out[i] = (st["sx"] + prior_games * league) / (st["sw"] + prior_games)
        for i in idx:
            if not played[i]:
                continue
            st, x = state[teams[i]], values[i]
            st["sx"] = decay * st["sx"] + x
            st["sw"] = decay * st["sw"] + 1.0
            n_seen += 1
            league = x.copy() if n_seen == 1 else league + max(league_alpha, 1.0 / n_seen) * (x - league)

    ratings = pd.DataFrame(out, columns=METRICS)
    ratings.insert(0, "team", tg["team"].values)
    ratings.insert(0, "game_id", tg["game_id"].values)
    return ratings


def elo_ratings(
    games: pd.DataFrame,
    k: float = 20.0,
    home_adv: float = 48.0,
    revert: float = 1 / 3,
    mean: float = 1505.0,
) -> pd.DataFrame:
    """538-style Elo with margin-of-victory multiplier and partial reversion each offseason."""
    elo: dict[str, float] = {}
    last_season: dict[str, int] = {}
    rows = []
    for g in games.sort_values(["kickoff", "game_id"]).itertuples():
        for team in (g.home_team, g.away_team):
            if team not in elo:
                elo[team], last_season[team] = mean, g.season
            elif last_season[team] != g.season:
                elo[team] = mean * revert + elo[team] * (1 - revert)
                last_season[team] = g.season
        h, a = elo[g.home_team], elo[g.away_team]
        rows.append((g.game_id, h, a))
        if pd.isna(g.result):
            continue
        hfa = 0.0 if g.location == "Neutral" else home_adv
        diff = h + hfa - a
        expected = 1.0 / (1.0 + 10 ** (-diff / 400))
        score = 1.0 if g.result > 0 else 0.0 if g.result < 0 else 0.5
        winner_diff = diff if g.result > 0 else -diff
        mult = math.log(max(abs(g.result), 1) + 1) * 2.2 / (winner_diff * 0.001 + 2.2)
        shift = k * mult * (score - expected)
        elo[g.home_team] += shift
        elo[g.away_team] -= shift
    return pd.DataFrame(rows, columns=["game_id", "elo_home", "elo_away"])
