"""Project-wide paths and settings."""

from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
STADIUMS_CSV = DATA_DIR / "stadiums.csv"

# 2009 is loaded only to warm up the rolling ratings; training starts the year after.
FIRST_SEASON = 2009
FIRST_TRAIN_SEASON = 2010


def current_season(today: date | None = None) -> int:
    today = today or date.today()
    return today.year if today.month >= 3 else today.year - 1


# Cached files for the in-progress season are refetched once they are older than this.
CURRENT_SEASON_CACHE_HOURS = 6

# Franchise relocations: map old codes to current ones so ratings carry over.
TEAM_ALIASES = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

# Standard deviation of NFL game margins around the spread, used to turn a margin into a win probability.
MARGIN_SIGMA = 13.5

# Deliberately conservative: the market is hard to beat, and in walk-forward tests (2015-2026)
# bigger models overfit and did worse than the Vegas line. Shallow, slow trees beat it slightly.
LGBM_PARAMS = dict(
    n_estimators=150,
    learning_rate=0.01,
    num_leaves=4,
    min_child_samples=200,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_lambda=5.0,
    verbose=-1,
)

ODDS_DIR = DATA_DIR / "odds"
LEDGER_PATH = DATA_DIR / "bets" / "ledger.csv"

# SD of results around the model's line, for turning a point edge into a win probability.
# Spread uses the conservative league-wide margin SD; total is the walk-forward residual SD
# of total - pred_total (2015-2026 backtest: 13.2).
SPREAD_SIGMA = MARGIN_SIGMA
TOTAL_SIGMA = 13.2

# ATS hit rate used as the win probability for early-season spread bets. The closing-line backtest
# (2015-2026) hit 56.5% ATS in weeks 1-4 (743 games, above break-even in 11 of 12 seasons) but only
# 52.3% overall and 48.9% from week 13 on. Shrunk toward 50% with a 400-game prior:
# (0.565 * 743 + 0.5 * 400) / 1143 = 0.542.
SPREAD_EARLY_HIT_RATE = 0.54

# Paper-trading rules. Totals with a 1.5+ point edge hit 56.0% over 486 bets in the backtest.
# Spreads are bet only in `spread_weeks`, the one window where the model has beaten the spread.
BETTING = dict(
    books=["draftkings", "fanduel", "betmgm", "williamhill_us"],  # williamhill_us = Caesars
    ref_book="draftkings",      # whose lines feed the model; the median across books if it has none
    markets=["total", "spread"],
    min_edge_pts=1.5,           # totals
    spread_weeks=(1, 4),
    spread_min_edge_pts=0.5,
    min_ev=0.02,                # expected profit per $1 staked
    bankroll=1000.0,
    kelly_fraction=0.25,
    max_stake_pct=0.02,         # of bankroll, per bet
    max_week_exposure_pct=0.10, # of bankroll, all bets in a week
)

# Odds API bookmaker keys -> names for reports.
BOOK_NAMES = {"draftkings": "DraftKings", "fanduel": "FanDuel", "betmgm": "BetMGM", "williamhill_us": "Caesars"}

# The Odds API full team names -> nflverse codes.
ODDS_TEAM_NAMES = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LA", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}
