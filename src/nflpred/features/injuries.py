"""Injury impact: players listed on the final injury report, weighted by position and recent snap share.

QBs get zero weight here because the starting-QB rating already captures a QB being out.
Players on IR never appear on weekly reports; their absence shows up gradually in the rolling team ratings.
"""

from __future__ import annotations

import pandas as pd
import polars as pl

STATUS_WEIGHTS = {"Out": 1.0, "Doubtful": 0.85, "Questionable": 0.2}

OFFENSE_WEIGHTS = {
    "T": 1.0, "OT": 1.0, "OL": 0.7, "G": 0.6, "OG": 0.6, "C": 0.6,
    "WR": 0.9, "TE": 0.6, "RB": 0.5, "HB": 0.5, "FB": 0.15, "QB": 0.0,
}
DEFENSE_WEIGHTS = {
    "DE": 0.9, "EDGE": 0.9, "OLB": 0.75, "DT": 0.7, "DL": 0.75, "NT": 0.6,
    "LB": 0.6, "ILB": 0.6, "MLB": 0.6, "CB": 0.9, "DB": 0.7, "S": 0.6, "FS": 0.6, "SS": 0.6, "SAF": 0.6,
}
DEFAULT_SNAP_SHARE = 0.35  # player with no snap history (rookies, or seasons before 2012 snap data)
SNAP_WINDOW = 8


def player_snap_shares(snaps: pl.DataFrame, players: pl.DataFrame, games: pd.DataFrame) -> pl.DataFrame:
    """Rolling mean snap share over each player's last SNAP_WINDOW games, keyed by gsis_id and kickoff."""
    xwalk = players.filter(pl.col("pfr_id").is_not_null() & pl.col("gsis_id").is_not_null()).select(
        "pfr_id", "gsis_id").unique("pfr_id")
    kick = pl.from_pandas(games[["game_id", "kickoff"]])
    return (
        snaps.join(xwalk, left_on="pfr_player_id", right_on="pfr_id", how="inner")
        .join(kick, on="game_id", how="inner")
        .with_columns(pl.max_horizontal(pl.col("offense_pct").fill_null(0), pl.col("defense_pct").fill_null(0)).alias("share"))
        .sort("kickoff")
        .with_columns(pl.col("share").rolling_mean(SNAP_WINDOW, min_samples=1).over("gsis_id").alias("snap_share"))
        .select("gsis_id", "kickoff", "snap_share")
    )


def injury_impact(injuries: pl.DataFrame, snaps: pl.DataFrame, players: pl.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Per (game_id, team): inj_off and inj_def impact scores."""
    team_kick = pd.concat([
        games[["game_id", "season", "week", "kickoff", "home_team"]].rename(columns={"home_team": "team"}),
        games[["game_id", "season", "week", "kickoff", "away_team"]].rename(columns={"away_team": "team"}),
    ])
    team_kick = pl.from_pandas(team_kick).with_columns(pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32))

    report = (
        injuries.filter(pl.col("report_status").is_in(list(STATUS_WEIGHTS)))
        .select(pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32), "team", "gsis_id", "position", "report_status")
        .unique(["season", "week", "team", "gsis_id"])
        .join(team_kick, on=["season", "week", "team"], how="inner")
        .sort("kickoff")
    )
    if snaps.height:
        shares = player_snap_shares(snaps, players, games)
        # Strictly-before join: a player's share uses only games before the one he is listed for.
        report = report.join_asof(  # both sides sorted by kickoff
           shares, on="kickoff", by="gsis_id", strategy="backward", allow_exact_matches=False)
    else:
        report = report.with_columns(pl.lit(None, dtype=pl.Float64).alias("snap_share"))

    df = report.to_pandas()
    df["snap_share"] = df["snap_share"].fillna(DEFAULT_SNAP_SHARE)
    df["status_w"] = df["report_status"].map(STATUS_WEIGHTS)
    df["off_w"] = df["position"].map(OFFENSE_WEIGHTS).fillna(0.0)
    df["def_w"] = df["position"].map(DEFENSE_WEIGHTS).fillna(0.0)
    df["inj_off"] = df["off_w"] * df["status_w"] * df["snap_share"]
    df["inj_def"] = df["def_w"] * df["status_w"] * df["snap_share"]
    return df.groupby(["game_id", "team"], as_index=False)[["inj_off", "inj_def"]].sum()
