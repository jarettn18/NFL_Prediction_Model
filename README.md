# nflpred

Predicts NFL game outcomes (**win probability**, **spread**, and **total**) from team strength, starting QBs, injuries, rest/travel/location, weather, and Vegas lines. It uses only free data: [nflverse](https://github.com/nflverse) via `nflreadpy` and [Open-Meteo](https://open-meteo.com) for weather.

## Setup
```
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

## Usage
```
nflpred ingest                 # download 2009-present into data/cache and build the feature table
nflpred backtest --start 2015  # walk-forward test vs Vegas closing lines
nflpred train                  # train on all completed games, save models/models.pkl
nflpred predict                # next week's games (refreshes injuries, lines, QBs, weather forecast)
nflpred predict --season 2026 --week 5
```
Reports are written to `reports/` as CSV and Markdown.

## Weekly betting agent (paper trading)
Each week the agent pulls lines from DraftKings, FanDuel, BetMGM and Caesars via [The Odds API](https://the-odds-api.com). It scores them with the model, sends each bet to the book with the best number and price, sizes the bets, and records them in a paper-trading ledger. **It never places real bets.** Sportsbooks have no betting API and their terms forbid automated wagering, so you place real bets yourself, at the book named on the slip, and only once the paper record justifies it.

Setup: get a free API key and store it as a user environment variable:
```
setx ODDS_API_KEY "your-key"
```

```
nflpred odds          # fetch current lines from every book into data/odds/{season}_week{wk}.parquet (3 API credits)
nflpred bets          # refresh, retrain, fetch odds, write reports/{season}_week{wk}_bets.md, add paper bets
nflpred bets --dry-run --lines latest   # preview against the newest lines without touching the ledger
nflpred ledger        # grade finished bets vs final scores and closing lines; show record, ROI, CLV
```
- **Opening lines** are the first snapshot the agent sees for each game, so run it soon after lines post.
- **Rules** live in `BETTING` in [config.py](src/nflpred/config.py). All bets need EV of at least 2%, are sized at quarter Kelly, and are capped at 2% of bankroll per bet and 10% per week.
  - **Totals:** every week, with an edge of at least 1.5 points. The backtest hit 56.0% over 486 bets.
  - **Spreads:** weeks 1–4 only, with an edge of at least 0.5 points. Overall the model doesn't beat the spread (52.3% vs 52.4% break-even), but in weeks 1–4 it hit 56.5% over 743 games and beat break-even in 11 of 12 seasons. From week 13 on it hit only 48.9%. This window was found by slicing the same backtest data, so treat it as a hypothesis for paper trading to confirm. Its win probability is a shrunk version of the backtest rate (`SPREAD_EARLY_HIT_RATE`, 54%).
- **Line shopping:** the model runs once per game on DraftKings' line, or the median across books if DraftKings has none. Each bet then goes to whichever book offers the best expected value for that side: a better number first, then a better price. Up to 10 books cost the same 3 credits per fetch. The Odds API currently returns no Caesars lines, so those bets come from the other three books.
- **Ledger** is `data/bets/ledger.csv`, one bet per game per market, so reruns never duplicate a bet. The paper bankroll is the starting bankroll plus graded P&L.
- **CLV** (closing-line value) is how many points the closing line moved toward your bet. Consistently positive CLV is the best early sign of a real edge, well before win/loss records mean anything. `nflpred ledger` reports "ready for real money" only after 100+ graded bets with positive average CLV and ROI.

Schedule it to run every Tuesday at 10:00. Lines open Sunday night or Monday, and Monday night results are in nflverse by Tuesday. Run [scripts/weekly_bets.ps1](scripts/weekly_bets.ps1) by hand once first, then register it (replace the path with your checkout's):
```
schtasks /create /tn "nflpred weekly bets" /sc weekly /d TUE /st 10:00 /tr "powershell -NoProfile -ExecutionPolicy Bypass -File C:\path\to\nflpred\scripts\weekly_bets.ps1"
```
Logs go to `logs/weekly_{date}.log`.

**Limitation:** at the open, the final injury reports and confirmed QB starters aren't out yet. The model uses whatever reports exist, so it has less information than in the closing-line backtest. CLV tracking is what shows whether the opening-line bets still hold up.

## How it works
- **Team strength** ([features/team_strength.py](src/nflpred/features/team_strength.py)): Elo, plus exponentially weighted offensive and defensive EPA/play (pass and rush), success rate, scoring, and pace. Ratings regress toward the league mean each offseason.
- **QB** ([features/qb.py](src/nflpred/features/qb.py)): the listed starter's shrunk EPA/dropback and experience, plus a flag when the starter changes. This is where a backup QB starting shows up.
- **Injuries** ([features/injuries.py](src/nflpred/features/injuries.py)): players listed Out, Doubtful, or Questionable on the final report, weighted by position and by snap share over their previous 8 games.
- **Situational** ([features/situational.py](src/nflpred/features/situational.py)): rest, travel distance, time-zone shift, body-clock kickoff hour, altitude, divisional, primetime, neutral site, playoffs.
- **Weather** ([weather.py](src/nflpred/weather.py)): temperature, wind, and precipitation at kickoff from Open-Meteo (archive for history, forecast for upcoming games). Domes and retractable roofs count as indoor.
- **Models** ([models.py](src/nflpred/models.py)): LightGBM models that learn *adjustments to the Vegas line*. The margin model predicts `result - spread`, the total model predicts `total - total_line`, and the win classifier is boosted from the market's implied log-odds. A separate "pure football" model with no market inputs covers games whose lines aren't posted yet.

Every feature is point-in-time: a game's features use only games that kicked off earlier (see [tests/](tests/test_features.py)).

## Conventions
- `spread_line` follows nflverse: **positive means the home team is favored** by that many points.
- In reports, `*` marks a model-vs-Vegas edge of at least 1.5 points.
- Betting against the spread at -110 odds needs a 52.4% win rate to break even. Beating closing lines consistently is very hard, so read the backtest before trusting any edge.
