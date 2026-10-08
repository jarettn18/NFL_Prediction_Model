import pandas as pd
import pytest

from nflpred import betting, odds
from nflpred.config import ODDS_TEAM_NAMES, SPREAD_EARLY_HIT_RATE, SPREAD_SIGMA

BOOKS = ["draftkings", "fanduel", "betmgm", "williamhill_us"]


def bookmaker(key, spread_home=-7.5, total=41.5, over=-105, under=-115):
    return {"key": key, "markets": [
        {"key": "h2h", "outcomes": [{"name": "Jacksonville Jaguars", "price": -340},
                                    {"name": "Philadelphia Eagles", "price": 270}]},
        {"key": "spreads", "outcomes": [{"name": "Jacksonville Jaguars", "price": -110, "point": spread_home},
                                        {"name": "Philadelphia Eagles", "price": -110, "point": -spread_home}]},
        {"key": "totals", "outcomes": [{"name": "Over", "price": over, "point": total},
                                       {"name": "Under", "price": under, "point": total}]},
    ]}


EVENTS = [{
    "id": "e1", "commence_time": "2026-10-11T17:00:00Z",
    "home_team": "Jacksonville Jaguars", "away_team": "Philadelphia Eagles",
    "bookmakers": [bookmaker("draftkings"), bookmaker("williamhill_us", total=42.0), bookmaker("pointsbetus")],
}, {
    # Monday night game: 00:15 UTC Tuesday is Monday 20:15 Eastern.
    "id": "e2", "commence_time": "2026-10-13T00:15:00Z",
    "home_team": "Los Angeles Rams", "away_team": "Buffalo Bills",
    "bookmakers": [{"key": "fanduel", "markets": []}],
}]


def schedule():
    return pd.DataFrame({
        "game_id": ["2026_05_PHI_JAX", "2026_05_BUF_LA"], "season": [2026, 2026], "week": [5, 5],
        "home_team": ["JAX", "LA"], "away_team": ["PHI", "BUF"],
        "kickoff": pd.to_datetime(["2026-10-11 13:00", "2026-10-12 20:15"]),
    })


def test_team_names_cover_all_32_teams():
    assert len(set(ODDS_TEAM_NAMES.values())) == 32


def test_parse_odds_one_row_per_book_with_nflverse_spread_sign():
    lines = odds.parse_odds(EVENTS, BOOKS)
    assert sorted(lines["book"]) == ["draftkings", "fanduel", "williamhill_us"]  # pointsbetus not requested
    dk = lines[lines["book"] == "draftkings"].iloc[0]
    assert (dk["home_team"], dk["away_team"]) == ("JAX", "PHI")
    assert dk["spread_line"] == 7.5  # home favored by 7.5 -> positive
    assert (dk["total_line"], dk["over_price"], dk["under_price"]) == (41.5, -105, -115)
    assert (dk["home_moneyline"], dk["away_moneyline"]) == (-340, 270)
    assert lines.loc[lines["book"] == "williamhill_us", "total_line"].item() == 42.0


def test_match_games_handles_utc_kickoffs():
    matched, unmatched = odds.match_games(odds.parse_odds(EVENTS, BOOKS), schedule())
    assert len(matched) == 3 and unmatched.empty
    assert matched.loc[matched["book"] == "fanduel", "game_id"].item() == "2026_05_BUF_LA"


def test_opening_and_latest_lines_per_book(tmp_path, monkeypatch):
    monkeypatch.setattr(odds, "ODDS_DIR", tmp_path)
    first = odds.match_games(odds.parse_odds(EVENTS, BOOKS, fetched_at=pd.Timestamp("2026-10-06", tz="UTC")),
                             schedule())[0]
    later = first.assign(total_line=first["total_line"] + 2, fetched_at=pd.Timestamp("2026-10-09", tz="UTC"))
    odds.snapshot(later)
    odds.snapshot(first)  # written out of order on purpose
    # The FanDuel row for BUF @ LA has no lines and is skipped.
    opening = odds.opening_lines(2026, 5).set_index("book")["total_line"]
    latest = odds.latest_lines(2026, 5).set_index("book")["total_line"]
    assert opening.to_dict() == {"draftkings": 41.5, "williamhill_us": 42.0}
    assert latest.to_dict() == {"draftkings": 43.5, "williamhill_us": 44.0}


def test_reference_lines_prefers_ref_book_then_median():
    offers = pd.DataFrame({
        "game_id": ["g1", "g1", "g2", "g2"], "book": ["draftkings", "fanduel", "fanduel", "betmgm"],
        "fetched_at": pd.Timestamp("2026-10-06", tz="UTC"), "spread_line": [3.0, 3.5, -1.0, -2.0],
        **{c: 41.0 for c in odds.LINE_COLUMNS if c != "spread_line"}})
    ref = odds.reference_lines(offers, "draftkings").set_index("game_id")
    assert ref.loc["g1", "spread_line"] == 3.0 and ref.loc["g1", "book"] == "draftkings"
    assert ref.loc["g2", "spread_line"] == -1.5 and ref.loc["g2", "book"] == "median"


def test_ev_and_kelly_math():
    assert betting.american_to_decimal(-110) == pytest.approx(1.9091, abs=1e-4)
    assert betting.american_to_decimal(150) == 2.5
    assert betting.expected_value(0.5238, -110) == pytest.approx(0.0, abs=1e-3)  # break-even
    assert betting.expected_value(0.55, -110) == pytest.approx(0.55 * 10 / 11 - 0.45)
    # full Kelly at even money with p=0.55 is 10%; quarter Kelly 2.5%, capped at 2%
    assert betting.kelly_stake(0.55, 100, 1000, 0.25, 1.0) == pytest.approx(25.0)
    assert betting.kelly_stake(0.55, 100, 1000, 0.25, 0.02) == pytest.approx(20.0)
    assert betting.kelly_stake(0.45, -110, 1000, 0.25, 0.02) == 0.0


def scored_game(**kw):
    """One game scored on the reference line: JAX -7.5, total 41.5."""
    row = dict(season=2026, week=5, game_id="g1", home_team="JAX", away_team="PHI",
               kickoff=pd.Timestamp("2026-10-11 13:00"), lines_fetched_at=pd.Timestamp("2026-10-06", tz="UTC"),
               spread_line=7.5, total_line=41.5, pred_margin=4.5, pred_total=38.0)
    row.update(kw)
    return pd.DataFrame([row])


def offers(*books, game_id="g1"):
    """Book offers; each item is (book, overrides) on top of JAX -7.5 / total 41.5, all at -110."""
    base = dict(game_id=game_id, spread_line=7.5, spread_home_price=-110, spread_away_price=-110,
                total_line=41.5, over_price=-110, under_price=-110, home_moneyline=-340, away_moneyline=270)
    return pd.DataFrame([{**base, "book": b, **kw} for b, kw in books])


def by_market(c):
    return c.set_index("market")


def test_totals_go_to_best_number_then_best_price():
    # Model total 38 -> Under. FanDuel's 42.5 beats DraftKings' 41.5.
    c = by_market(betting.candidates(scored_game(), offers(("draftkings", {}), ("fanduel", {"total_line": 42.5}))))
    assert c.loc["total", "side"] == "Under" and c.loc["total", "book"] == "fanduel"
    assert c.loc["total", "line"] == 42.5 and c.loc["total", "edge"] == pytest.approx(4.5)
    assert c.loc["total", "n_books"] == 2
    # Same number, better price wins.
    c = by_market(betting.candidates(scored_game(), offers(("draftkings", {}), ("betmgm", {"under_price": -105}))))
    assert c.loc["total", "book"] == "betmgm"


def test_spreads_only_in_early_weeks():
    o = offers(("draftkings", {}))
    assert "spread" not in betting.candidates(scored_game(week=5), o)["market"].tolist()
    assert "spread" in betting.candidates(scored_game(week=3), o)["market"].tolist()


def test_spread_probability_uses_early_hit_rate_shifted_by_the_line():
    # Model has JAX by 4.5 vs JAX -7.5 -> take PHI. Caesars has PHI +8.
    c = by_market(betting.candidates(scored_game(week=2), offers(("draftkings", {}), ("williamhill_us", {"spread_line": 8.0}))))
    s = c.loc["spread"]
    assert (s["side"], s["book"], s["line"], s["edge"]) == ("PHI", "williamhill_us", 8.0, pytest.approx(3.5))
    shift = betting.cover_prob(3.5, SPREAD_SIGMA) - betting.cover_prob(3.0, SPREAD_SIGMA)
    assert s["p_win"] == pytest.approx(SPREAD_EARLY_HIT_RATE + shift)
    # At the reference book's own number, p is exactly the early-season rate.
    dk = by_market(betting.candidates(scored_game(week=2), offers(("draftkings", {}))))
    assert dk.loc["spread", "p_win"] == pytest.approx(SPREAD_EARLY_HIT_RATE)


def test_select_bets_respects_caps_and_min_edges():
    cfg = dict(min_edge_pts=1.5, spread_min_edge_pts=0.5, min_ev=0.02, kelly_fraction=0.25, max_stake_pct=0.02,
               max_week_exposure_pct=0.03, markets=["total", "spread"], spread_weeks=(1, 4))
    games = pd.concat([scored_game(game_id=f"g{i}", pred_total=30.0) for i in range(3)], ignore_index=True)
    o = pd.concat([offers(("draftkings", {}), game_id=f"g{i}") for i in range(3)], ignore_index=True)
    bets = betting.select_bets(betting.candidates(games, o, cfg), 1000, cfg=cfg)
    assert (bets["stake"] <= 20).all() and bets["stake"].sum() <= 30
    # Rounding never pushes the week over its cap: six $20 bets scaled to fit $100 are $16.67 each,
    # which would round up to $17 ($102 total).
    six = pd.concat([scored_game(game_id=f"h{i}", pred_total=30.0) for i in range(6)], ignore_index=True)
    o6 = pd.concat([offers(("draftkings", {}), game_id=f"h{i}") for i in range(6)], ignore_index=True)
    cfg10 = {**cfg, "max_week_exposure_pct": 0.10}
    assert betting.select_bets(betting.candidates(six, o6, cfg10), 1000, cfg=cfg10)["stake"].sum() == 96
    # A 0.5-pt total edge is too small...
    small = betting.candidates(scored_game(pred_total=41.0), offers(("draftkings", {})), cfg)
    assert betting.select_bets(small, 1000, cfg=cfg).empty
    # ...and so is a 0.2-pt spread edge, even though the early-season rate makes its EV positive.
    tiny = betting.candidates(scored_game(week=2, pred_margin=7.3, pred_total=41.5), offers(("draftkings", {})), cfg)
    assert (tiny.set_index("market").loc["spread", "ev"] > 0.02)
    assert betting.select_bets(tiny, 1000, cfg=cfg).empty


def test_ledger_is_idempotent_and_grades(tmp_path):
    cfg = dict(min_edge_pts=1.5, spread_min_edge_pts=0.5, min_ev=0.0, kelly_fraction=0.25, max_stake_pct=0.02,
               max_week_exposure_pct=1.0, markets=["total", "spread"], spread_weeks=(1, 4))
    cands = betting.candidates(scored_game(week=3), offers(("draftkings", {})), cfg)
    bets = betting.select_bets(cands, 1000, cfg=cfg)
    ledger, added = betting.add_bets(betting.load_ledger(tmp_path / "ledger.csv"), bets)
    ledger, added_again = betting.add_bets(ledger, bets)
    assert len(ledger) == 2 and added_again.empty
    assert (ledger["book"] == "draftkings").all()

    # Final JAX 24 PHI 17 (won by 7, PHI +7.5 covers); total 41 (Under 41.5 wins).
    # Closing lines: JAX -8.5, total 40.5.
    final = pd.DataFrame({"game_id": ["g1"], "home_team": ["JAX"], "result": [7.0], "total": [41.0],
                          "spread_line": [8.5], "total_line": [40.5]})
    graded, n = betting.grade(ledger, final)
    g = graded.set_index("market")
    assert n == 2
    assert g.loc["total", "outcome"] == "W" and g.loc["total", "clv_pts"] == pytest.approx(1.0)
    assert g.loc["spread", "outcome"] == "W" and g.loc["spread", "clv_pts"] == pytest.approx(-1.0)
    assert g.loc["total", "pnl"] == pytest.approx(g.loc["total", "stake"] * 10 / 11)
    summary = betting.summarize_ledger(graded)
    assert {"ALL", "ALL spread", "ALL total"} <= set(summary.index)


def test_old_ledger_without_book_column_loads(tmp_path):
    path = tmp_path / "ledger.csv"
    old_cols = [c for c in betting.LEDGER_COLUMNS if c != "book"]
    pd.DataFrame([{c: None for c in old_cols} | {"bet_id": "g1_total", "stake": 10}]).to_csv(path, index=False)
    ledger = betting.load_ledger(path)
    assert list(ledger.columns) == betting.LEDGER_COLUMNS and ledger.loc[0, "book"] == "draftkings"


def test_push_grades_to_zero():
    bets = betting.candidates(scored_game(), offers(("draftkings", {}))).assign(stake=10.0)
    ledger, _ = betting.add_bets(pd.DataFrame(columns=betting.LEDGER_COLUMNS), bets)
    final = pd.DataFrame({"game_id": ["g1"], "home_team": ["JAX"], "result": [3.0], "total": [41.5],
                          "spread_line": [7.5], "total_line": [41.5]})
    graded, _ = betting.grade(ledger, final)
    assert graded.loc[0, "outcome"] == "P" and graded.loc[0, "pnl"] == 0
