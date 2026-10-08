import numpy as np
import pandas as pd
import polars as pl
import pytest

from nflpred.features.build import market_home_prob
from nflpred.features.injuries import injury_impact
from nflpred.features.qb import qb_ratings
from nflpred.features.situational import haversine_km
from nflpred.features.team_strength import METRICS, elo_ratings, rolling_ratings
from nflpred.weather import game_weather


def make_games() -> pd.DataFrame:
    """Four weekly games: A hosts B, B hosts A, then a new season, then an unplayed game."""
    rows = [
        ("g1", 2020, 1, "2020-09-13", "A", "B", 24, 17, "qa", "qb"),
        ("g2", 2020, 2, "2020-09-20", "B", "A", 10, 31, "qb", "qa"),
        ("g3", 2021, 1, "2021-09-12", "A", "B", 20, 20, "qa2", "qb"),
        ("g4", 2021, 2, "2021-09-19", "B", "A", None, None, "qb", "qa"),
    ]
    df = pd.DataFrame(rows, columns=["game_id", "season", "week", "gameday", "home_team", "away_team",
                                     "home_score", "away_score", "home_qb_id", "away_qb_id"])
    df["kickoff"] = pd.to_datetime(df["gameday"]) + pd.Timedelta(hours=13)
    df["result"] = df["home_score"] - df["away_score"]
    df["location"] = "Home"
    return df


def make_stats(games: pd.DataFrame, bump_game: str | None = None) -> pd.DataFrame:
    rows = []
    for g in games[games["result"].notna()].itertuples():
        for team in (g.home_team, g.away_team):
            val = 0.1 if team == "A" else -0.1
            if g.game_id == bump_game:
                val += 5.0
            rows.append({"game_id": g.game_id, "team": team, **{m: val for m in METRICS if m not in ("points_for", "points_against")}})
    return pd.DataFrame(rows)


def test_rolling_ratings_ignore_current_and_future_games():
    games = make_games()
    base = rolling_ratings(games, make_stats(games))
    bumped = rolling_ratings(games, make_stats(games, bump_game="g2"))
    key = ["game_id", "team"]
    b, x = base.set_index(key), bumped.set_index(key)
    # Ratings going into g1 and g2 must not see g2's stats; g3 onward should.
    for gid in ("g1", "g2"):
        pd.testing.assert_frame_equal(b.loc[gid], x.loc[gid])
    assert not np.allclose(b.loc["g3", "off_epa"], x.loc["g3", "off_epa"])


def test_unplayed_games_get_latest_rating():
    games = make_games()
    r = rolling_ratings(games, make_stats(games)).set_index(["game_id", "team"])
    assert r.loc[("g4", "A"), "off_epa"] > r.loc[("g4", "B"), "off_epa"]


def test_season_regression_pulls_toward_mean():
    games = make_games()
    r = rolling_ratings(games, make_stats(games), season_carry=0.1).set_index(["game_id", "team"])
    # A's rating entering the new season (g3) is closer to the league mean (0) than at the end of 2020.
    end_2020 = r.loc[("g2", "A"), "off_epa"]
    assert abs(r.loc[("g3", "A"), "off_epa"]) < abs(end_2020) + 0.05


def test_elo_is_pregame():
    games = make_games()
    elo = elo_ratings(games).set_index("game_id")
    assert elo.loc["g1", "elo_home"] == elo.loc["g1", "elo_away"] == 1505.0
    assert elo.loc["g2", "elo_away"] > 1505.0  # A won g1
    flipped = games.copy()
    flipped.loc[0, "result"] = -7
    elo2 = elo_ratings(flipped).set_index("game_id")
    assert elo2.loc["g1", "elo_home"] == 1505.0
    assert elo2.loc["g2", "elo_away"] < 1505.0


def test_qb_rating_uses_only_prior_games_and_flags_new_starter():
    games = make_games()
    stats = pd.DataFrame({"game_id": ["g1", "g2"], "qb_id": ["qa", "qa"],
                          "epa_sum": [20.0, 500.0], "dropbacks": [30.0, 30.0]})
    r = qb_ratings(games, stats).set_index("game_id")
    first = r.loc["g2", "away_qb_rating"]
    # g2's huge EPA must not leak into qa's g2 rating.
    stats2 = stats.assign(epa_sum=[20.0, -500.0])
    assert qb_ratings(games, stats2).set_index("game_id").loc["g2", "away_qb_rating"] == pytest.approx(first)
    # A unseen starter gets the prior and is flagged as a change.
    assert r.loc["g3", "home_qb_rating"] == pytest.approx(-0.05)
    assert r.loc["g3", "home_qb_new"] == 1


def test_injury_impact_weights_position_status_and_prior_snaps():
    games = make_games()
    injuries = pl.DataFrame({
        "season": [2021, 2021, 2021], "week": [1, 1, 1], "team": ["A", "A", "B"],
        "gsis_id": ["wr1", "qb1", "cb1"], "position": ["WR", "QB", "CB"],
        "report_status": ["Out", "Out", "Questionable"],
    })
    snaps = pl.DataFrame({
        "game_id": ["g2", "g3"], "pfr_player_id": ["WR01", "WR01"],
        "offense_pct": [0.8, 0.0], "defense_pct": [0.0, 0.0],
    })
    players = pl.DataFrame({"gsis_id": ["wr1"], "pfr_id": ["WR01"]})
    out = injury_impact(injuries, snaps, players, games).set_index("team")
    # WR: weight 0.9 * Out 1.0 * prior snap share 0.8 (g3's own 0.0 must not count). QB adds nothing.
    assert out.loc["A", "inj_off"] == pytest.approx(0.9 * 0.8)
    # CB without snap history: 0.9 * Questionable 0.2 * default share 0.35.
    assert out.loc["B", "inj_def"] == pytest.approx(0.9 * 0.2 * 0.35)


def test_haversine_known_distance():
    nyc_to_la = haversine_km(40.8135, -74.0745, 33.9535, -118.3392)
    assert 3900 < nyc_to_la < 4000


def test_indoor_games_get_neutral_weather_without_network():
    games = pd.DataFrame({"game_id": ["x"], "stadium_id": ["NOR00"], "roof": ["dome"],
                          "gameday": ["2024-10-01"], "gametime": ["13:00"]})
    w = game_weather(games, verbose=False).iloc[0]
    assert (w["temp_f"], w["wind_mph"], w["precip_in"], w["is_indoor"]) == (70.0, 0.0, 0.0, 1)


def test_market_prob_removes_vig_and_falls_back_to_spread():
    df = pd.DataFrame({"home_moneyline": [-150, None], "away_moneyline": [130, None], "spread_line": [3.0, 0.0]})
    p = market_home_prob(df)
    assert 0.57 < p[0] < 0.59
    assert p[1] == pytest.approx(0.5)
