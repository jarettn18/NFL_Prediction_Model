"""Turn model predictions vs sportsbook lines into sized bets, and keep a paper-trading ledger.

Bet lines are stored from the bettor's side: a spread bet's `line` is that team's handicap
(home -3.5 / away +3.5), a total bet's `line` is the total. CLV is positive when the closing
line moved in our favor.

The model runs once per game on a reference line; each bet then goes to whichever book offers
the best expected value for the side the model leans to.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import norm

from nflpred.config import BETTING, LEDGER_PATH, SPREAD_EARLY_HIT_RATE, SPREAD_SIGMA, TOTAL_SIGMA
from nflpred.features.build import market_home_prob
from nflpred.models import GameModels
from nflpred.odds import LINE_COLUMNS

LEDGER_COLUMNS = ["bet_id", "created_at", "season", "week", "game_id", "matchup", "market", "side", "line", "price",
                  "book", "stake", "model_line", "edge", "p_win", "ev", "lines_fetched_at",
                  "close_line", "clv_pts", "outcome", "pnl"]

CANDIDATE_COLUMNS = ["season", "week", "game_id", "matchup", "kickoff", "lines_fetched_at", "market", "side", "line",
                     "price", "book", "n_books", "model_line", "edge", "p_win", "ev"]


def american_to_decimal(price: float) -> float:
    return 1 + (price / 100 if price > 0 else 100 / -price)


def cover_prob(edge_pts: float, sigma: float) -> float:
    """Chance a side beats the line, given the model's edge for that side (negative = model is against it),
    treating the result as Normal around the model's line."""
    return float(norm.cdf(edge_pts / sigma))


def expected_value(p: float, price: float) -> float:
    """Expected profit per $1 staked."""
    return p * (american_to_decimal(price) - 1) - (1 - p)


def kelly_stake(p: float, price: float, bankroll: float, fraction: float, max_pct: float) -> float:
    b = american_to_decimal(price) - 1
    f = max((p * b - (1 - p)) / b, 0.0)
    return float(min(bankroll * fraction * f, bankroll * max_pct))


def score_against_lines(games: pd.DataFrame, lines: pd.DataFrame, model: GameModels) -> pd.DataFrame:
    """Re-run the model with the given sportsbook lines in place of the schedule's lines."""
    g = games.drop(columns=[c for c in LINE_COLUMNS if c in games.columns])
    g = g.merge(lines[["game_id", "fetched_at", *LINE_COLUMNS]].rename(columns={"fetched_at": "lines_fetched_at"}),
                on="game_id", how="inner")
    g["mkt_home_prob"] = market_home_prob(g)
    return pd.concat([g, model.predict(g)], axis=1)


def in_spread_window(week: int, cfg: dict = BETTING) -> bool:
    lo, hi = cfg["spread_weeks"]
    return lo <= week <= hi


def _side_offers(r, offers: pd.DataFrame, market: str, side_home_or_over: bool):
    """(book, line, price, edge) for one side at each book. Lines are from the bettor's side; edge is the
    model's expected margin over that line."""
    for _, o in offers.iterrows():
        if market == "total":
            line, price = o["total_line"], o["over_price" if side_home_or_over else "under_price"]
            edge = (r["pred_total"] - line) if side_home_or_over else (line - r["pred_total"])
        else:
            line = -o["spread_line"] if side_home_or_over else o["spread_line"]
            price = o["spread_home_price" if side_home_or_over else "spread_away_price"]
            edge = (r["pred_margin"] if side_home_or_over else -r["pred_margin"]) + line
        if pd.notna(line) and pd.notna(price):
            yield o["book"], line, price, edge


def candidates(scored: pd.DataFrame, offers: pd.DataFrame, cfg: dict = BETTING) -> pd.DataFrame:
    """The side the model leans to in each game and market, at the book with the best EV.

    `scored` has one row per game (model run on the reference line); `offers` has one row per game per book.
    Totals: win probability from the edge at that book's line. Spreads (only in `spread_weeks`): the backtest's
    early-season ATS hit rate, shifted by how much better or worse the book's number is than the reference."""
    rows = []
    by_game = dict(tuple(offers.groupby("game_id")))
    for _, r in scored.iterrows():
        h, a = r["home_team"], r["away_team"]
        game_offers = by_game.get(r["game_id"], offers.iloc[0:0])
        base = {"season": r["season"], "week": r["week"], "game_id": r["game_id"], "matchup": f"{a} @ {h}",
                "kickoff": r["kickoff"], "lines_fetched_at": r["lines_fetched_at"]}
        markets = [m for m in cfg["markets"] if m != "spread" or in_spread_window(r["week"], cfg)]
        for market in markets:
            ref_line = r["total_line"] if market == "total" else r["spread_line"]
            if pd.isna(ref_line):
                continue
            ref_edge = (r["pred_total"] - ref_line) if market == "total" else (r["pred_margin"] - ref_line)
            first = ref_edge > 0  # Over, or the home team
            if market == "total":
                side, model_line = ("Over" if first else "Under"), r["pred_total"]
            else:
                side, model_line = (h if first else a), (-r["pred_margin"] if first else r["pred_margin"])
            best = None
            n = 0
            for book, line, price, edge in _side_offers(r, game_offers, market, first):
                n += 1
                if market == "total":
                    p = cover_prob(edge, TOTAL_SIGMA)
                else:
                    p = SPREAD_EARLY_HIT_RATE + cover_prob(edge, SPREAD_SIGMA) - cover_prob(abs(ref_edge), SPREAD_SIGMA)
                ev = expected_value(p, price)
                if best is None or ev > best["ev"]:
                    best = {"book": book, "line": line, "price": price, "edge": edge, "p_win": p, "ev": ev}
            if best:
                rows.append({**base, "market": market, "side": side, "model_line": model_line, "n_books": n, **best})
    c = pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)
    return c.sort_values("ev", ascending=False).reset_index(drop=True)


def select_bets(cands: pd.DataFrame, bankroll: float, already_staked: float = 0.0, cfg: dict = BETTING) -> pd.DataFrame:
    """Bets that clear the edge and EV bars, Kelly-sized, scaled down to fit the weekly exposure cap."""
    min_edge = np.where(cands["market"] == "spread", cfg["spread_min_edge_pts"], cfg["min_edge_pts"])
    bets = cands[(cands["edge"] >= min_edge) & (cands["ev"] >= cfg["min_ev"])].copy()
    bets["stake"] = [kelly_stake(p, price, bankroll, cfg["kelly_fraction"], cfg["max_stake_pct"])
                     for p, price in zip(bets["p_win"], bets["price"])]
    room = max(bankroll * cfg["max_week_exposure_pct"] - already_staked, 0.0)
    total = bets["stake"].sum()
    if total > room:
        bets["stake"] *= room / total
    bets["stake"] = np.floor(bets["stake"] + 1e-9)  # whole dollars, never over the caps
    return bets[bets["stake"] > 0].reset_index(drop=True)


# --- ledger -------------------------------------------------------------------------------------

def load_ledger(path=LEDGER_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    ledger = pd.read_csv(path)
    if "book" not in ledger.columns:  # ledgers from before line shopping were all DraftKings
        ledger["book"] = "draftkings"
    return ledger[LEDGER_COLUMNS]


def save_ledger(ledger: pd.DataFrame, path=LEDGER_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ledger[LEDGER_COLUMNS].to_csv(path, index=False)


def add_bets(ledger: pd.DataFrame, bets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Append bets not already in the ledger (one bet per game per market). Returns (ledger, added)."""
    if bets.empty:
        return ledger, bets
    new = bets.assign(bet_id=bets["game_id"] + "_" + bets["market"],
                      created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    new = new[~new["bet_id"].isin(ledger["bet_id"])]
    for c in LEDGER_COLUMNS:
        if c not in new.columns:
            new[c] = np.nan
    new = new[LEDGER_COLUMNS]
    out = new if ledger.empty else pd.concat([ledger, new], ignore_index=True)
    return out, new


def current_bankroll(ledger: pd.DataFrame, start: float = BETTING["bankroll"]) -> float:
    return start + float(pd.to_numeric(ledger["pnl"], errors="coerce").fillna(0).sum())


def _grade_row(bet, game) -> dict:
    """Closing line, CLV and outcome for one bet, from the nflverse closing lines and final score."""
    if bet["market"] == "total":
        close = game["total_line"]
        over = bet["side"] == "Over"
        clv = (close - bet["line"]) if over else (bet["line"] - close)
        margin = (game["total"] - bet["line"]) if over else (bet["line"] - game["total"])
    else:
        home = bet["side"] == game["home_team"]
        close = -game["spread_line"] if home else game["spread_line"]
        clv = bet["line"] - close
        margin = (game["result"] if home else -game["result"]) + bet["line"]
    out = {"close_line": close, "clv_pts": clv}
    if pd.notna(margin):
        out["outcome"] = "W" if margin > 0 else ("L" if margin < 0 else "P")
        out["pnl"] = {"W": bet["stake"] * (american_to_decimal(bet["price"]) - 1), "L": -bet["stake"], "P": 0.0}[out["outcome"]]
    return out


def grade(ledger: pd.DataFrame, games: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Fill closing line, CLV, outcome and P&L for ungraded bets whose games are final."""
    ledger = ledger.copy()
    ledger["outcome"] = ledger["outcome"].astype(object)
    by_id = games.set_index("game_id")
    graded = 0
    for i, bet in ledger[ledger["outcome"].isna()].iterrows():
        if bet["game_id"] not in by_id.index:
            continue
        game = by_id.loc[bet["game_id"]]
        if pd.isna(game["result"]):
            continue
        for k, v in _grade_row(bet, game).items():
            ledger.at[i, k] = v
        graded += 1
    return ledger, graded


def summarize_ledger(ledger: pd.DataFrame) -> pd.DataFrame:
    """Record, ROI and average CLV by week, plus lifetime rows overall and per market."""
    done = ledger[ledger["outcome"].notna()].copy()
    if done.empty:
        return pd.DataFrame()

    def stats(g: pd.DataFrame) -> dict:
        w, l, p = (g["outcome"] == "W").sum(), (g["outcome"] == "L").sum(), (g["outcome"] == "P").sum()
        staked = g["stake"].sum()
        return {"bets": len(g), "record": f"{w}-{l}" + (f"-{p}" if p else ""),
                "win_pct": w / (w + l) if w + l else float("nan"), "staked": staked, "pnl": g["pnl"].sum(),
                "roi": g["pnl"].sum() / staked if staked else float("nan"), "avg_clv": g["clv_pts"].mean(),
                "beat_close": (g["clv_pts"] > 0).mean()}

    rows = {f"{int(s)} wk {int(w)}": stats(g) for (s, w), g in done.groupby(["season", "week"])}
    for market, g in done.groupby("market"):
        rows[f"ALL {market}"] = stats(g)
    rows["ALL"] = stats(done)
    return pd.DataFrame(rows).T


def ready_for_real_money(ledger: pd.DataFrame, min_bets: int = 100) -> tuple[bool, str]:
    done = ledger[ledger["outcome"].notna()]
    n = len(done)
    clv = done["clv_pts"].mean() if n else float("nan")
    roi = done["pnl"].sum() / done["stake"].sum() if n and done["stake"].sum() else float("nan")
    ok = n >= min_bets and clv > 0 and roi > 0
    return ok, f"{n}/{min_bets} graded bets, avg CLV {clv:+.2f} pts, ROI {roi:+.1%}"
