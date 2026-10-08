"""Starting-QB quality: a shrunk, exponentially weighted EPA per dropback for the listed starter.

nflverse schedules list each game's starting QB (projected starters for upcoming games), so a
backup starting shows up directly as a lower rating. That is the single biggest injury effect in the NFL.
"""

from __future__ import annotations

import math

import pandas as pd
import polars as pl


def qb_game_stats(pbp: pl.DataFrame) -> pd.DataFrame:
    """EPA sum and dropback count per (game_id, qb_id)."""
    return (
        pbp.filter((pl.col("qb_dropback") == 1) & pl.col("epa").is_not_null())
        .with_columns(pl.coalesce("passer_player_id", "rusher_player_id").alias("qb_id"))
        .filter(pl.col("qb_id").is_not_null())
        .group_by("game_id", "qb_id")
        .agg(pl.col("epa").sum().alias("epa_sum"), pl.len().cast(pl.Float64).alias("dropbacks"))
        .to_pandas()
    )


def qb_ratings(
    games: pd.DataFrame,
    qb_stats: pd.DataFrame,
    decay: float = 0.95,
    prior_epa: float = -0.05,
    prior_dropbacks: float = 150.0,
) -> pd.DataFrame:
    """Pregame rating of each game's listed home and away starting QB.

    rating = (decayed EPA sum + prior_dropbacks * prior_epa) / (decayed dropbacks + prior_dropbacks)
    Unknown and inexperienced QBs are pulled toward a below-average prior.
    """
    kick = games.set_index("game_id")["kickoff"]
    updates = qb_stats.assign(kickoff=qb_stats["game_id"].map(kick)).dropna(subset=["kickoff"])
    updates = updates.assign(kind=1)

    lookups = []
    for side in ("home", "away"):
        lk = games[["game_id", "kickoff", f"{side}_qb_id"]].rename(columns={f"{side}_qb_id": "qb_id"})
        lookups.append(lk.assign(side=side))
    lookups = pd.concat(lookups).dropna(subset=["qb_id"]).assign(kind=0)

    # Lookups sort before updates at the same kickoff, so a game never sees its own stats.
    events = pd.concat([lookups, updates], ignore_index=True).sort_values(["kickoff", "kind"])
    state: dict[str, list[float]] = {}  # qb_id -> [decayed epa, decayed dropbacks, career dropbacks]
    out = []
    for e in events.itertuples():
        s = state.setdefault(e.qb_id, [0.0, 0.0, 0.0])
        if e.kind == 0:
            rating = (s[0] + prior_dropbacks * prior_epa) / (s[1] + prior_dropbacks)
            out.append((e.game_id, e.side, rating, math.log1p(s[2])))
        else:
            s[0] = decay * s[0] + e.epa_sum
            s[1] = decay * s[1] + e.dropbacks
            s[2] += e.dropbacks

    r = pd.DataFrame(out, columns=["game_id", "side", "qb_rating", "qb_exp"])
    wide = r.pivot_table(index="game_id", columns="side", values=["qb_rating", "qb_exp"])
    wide.columns = [f"{side}_{metric}" for metric, side in wide.columns]
    wide = wide.reset_index()

    # Flag a change of starter since the team's previous game.
    new_flags = []
    for side in ("home", "away"):
        tg = games[["game_id", "kickoff", f"{side}_team", f"{side}_qb_id"]].rename(
            columns={f"{side}_team": "team", f"{side}_qb_id": "qb_id"})
        new_flags.append(tg)
    tg = pd.concat(new_flags, keys=["home", "away"], names=["side"]).reset_index(level=0)
    tg = tg.sort_values("kickoff")
    tg["prev_qb"] = tg.groupby("team")["qb_id"].shift()
    tg["qb_new"] = ((tg["qb_id"] != tg["prev_qb"]) & tg["prev_qb"].notna()).astype(int)
    flags = tg.pivot_table(index="game_id", columns="side", values="qb_new").add_suffix("_qb_new").reset_index()
    return wide.merge(flags, on="game_id", how="outer")
