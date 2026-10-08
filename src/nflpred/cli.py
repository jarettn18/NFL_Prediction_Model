"""Command-line interface: ingest, train, backtest, predict."""

from __future__ import annotations

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from nflpred import betting, ingest, odds
from nflpred.backtest import BREAKEVEN, calibration_table, summarize, walk_forward, walk_forward_weeks
from nflpred.config import (BETTING, BOOK_NAMES, CACHE_DIR, FIRST_SEASON, FIRST_TRAIN_SEASON, LEDGER_PATH, MODELS_DIR, REPORTS_DIR,
                            current_season)
from nflpred.features.build import build_dataset
from nflpred.models import GameModels

app = typer.Typer(add_completion=False, help="NFL game predictions: win probability, spread, and total.")
console = Console()
FEATURES_CACHE = CACHE_DIR / "features.parquet"
MODEL_PATH = MODELS_DIR / "models.pkl"


def _parse_seasons(spec: str | None) -> list[int]:
    if not spec:
        return list(range(FIRST_SEASON, current_season() + 1))
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(s) for s in spec.split(",")]


def get_dataset(refresh: bool = False, rebuild: bool = False) -> pd.DataFrame:
    """Feature table for every game. Reuses the cached build unless asked to refresh or rebuild."""
    if FEATURES_CACHE.exists() and not (refresh or rebuild):
        return pd.read_parquet(FEATURES_CACHE)
    console.print("[bold]Loading data...[/bold]")
    data = ingest.load_all(refresh_current=refresh)
    console.print("[bold]Building features...[/bold]")
    df = build_dataset(data)
    df.to_parquet(FEATURES_CACHE, index=False)
    return df


@app.command("ingest")
def ingest_cmd(
    seasons: str = typer.Option(None, help="e.g. 2010-2026 or 2024,2025 (default: all)"),
    refresh: bool = typer.Option(False, help="Re-download even if cached"),
):
    """Download nflverse data into the local parquet cache."""
    for name in ("schedules", "pbp", "injuries", "snap_counts"):
        df = ingest.load(name, _parse_seasons(seasons), refresh=refresh)
        console.print(f"{name}: {df.height:,} rows")
    ingest.load_players(refresh=refresh)
    get_dataset(rebuild=True)
    console.print(f"[green]Feature table cached at {FEATURES_CACHE}[/green]")


@app.command()
def train(
    kind: str = typer.Option("lgbm", help="lgbm or ridge"),
    refresh: bool = typer.Option(False, help="Refresh current-season data first"),
):
    """Train on every completed game and save the models."""
    df = get_dataset(refresh=refresh)
    df = df[df["season"] >= FIRST_TRAIN_SEASON]
    model = GameModels(kind=kind).fit(df)
    model.save(MODEL_PATH)
    console.print(f"[green]Trained through {model.trained_through}; saved to {MODEL_PATH}[/green]")
    imp = model.feature_importance("margin").head(15)
    t = Table(title="Top features: spread adjustment model")
    t.add_column("feature")
    t.add_column("importance", justify="right")
    for name, val in imp.items():
        t.add_row(name, f"{val:.1f}")
    console.print(t)


@app.command()
def backtest(
    start: int = typer.Option(2015, help="First test season"),
    end: int = typer.Option(None, help="Last test season (default: current)"),
    kind: str = typer.Option("lgbm", help="lgbm or ridge"),
    rebuild: bool = typer.Option(False, help="Rebuild the feature table first"),
):
    """Walk-forward backtest against Vegas closing lines."""
    df = get_dataset(rebuild=rebuild)
    preds = walk_forward(df, start, end or current_season(), kind=kind)
    REPORTS_DIR.mkdir(exist_ok=True)
    preds.to_csv(REPORTS_DIR / f"backtest_{kind}.csv", index=False)
    s = summarize(preds)

    t = Table(title=f"Walk-forward backtest ({kind}); lower log loss / MAE is better")
    cols = [("games", "games", "{:.0f}"), ("ll_model", "LogLoss model", "{:.4f}"), ("ll_vegas", "LogLoss Vegas", "{:.4f}"),
            ("ll_pure", "LogLoss pure", "{:.4f}"), ("acc_model", "Acc", "{:.3f}"),
            ("mae_margin_model", "MAE spr", "{:.2f}"), ("mae_margin_vegas", "MAE spr Vegas", "{:.2f}"),
            ("mae_total_model", "MAE tot", "{:.2f}"), ("mae_total_vegas", "MAE tot Vegas", "{:.2f}"),
            ("ats", "ATS", "{:.3f}"), ("ats_1.5", "ATS edge>1.5", "{:.3f}"), ("ats_1.5_n", "n", "{:.0f}"),
            ("ou", "O/U", "{:.3f}"), ("ou_1.5", "O/U edge>1.5", "{:.3f}"), ("ou_1.5_n", "n", "{:.0f}")]
    t.add_column("season")
    for _, label, _ in cols:
        t.add_column(label, justify="right")
    for season, row in s.iterrows():
        t.add_row(str(season), *[fmt.format(row[key]) if pd.notna(row[key]) else "-" for key, _, fmt in cols],
                  style="bold" if season == "ALL" else None)
    console.print(t)
    console.print(f"ATS / O/U break-even at -110 odds is {BREAKEVEN:.1%}.")

    cal = calibration_table(preds)
    ct = Table(title="Win probability calibration (all test seasons)")
    for c in ("bin", "games", "predicted", "actual"):
        ct.add_column(c, justify="right")
    for interval, row in cal.iterrows():
        ct.add_row(str(interval), f"{row['n']:.0f}", f"{row['predicted']:.3f}", f"{row['actual']:.3f}")
    console.print(ct)


def _fmt_spread(team_home: str, team_away: str, home_margin: float) -> str:
    if pd.isna(home_margin):
        return "-"
    if abs(home_margin) < 0.25:
        return "PK"
    fav = team_home if home_margin > 0 else team_away
    return f"{fav} -{abs(home_margin):.1f}"


def _weather_str(r) -> str:
    if r.get("is_indoor") == 1:
        return "indoor"
    if pd.isna(r.get("temp_f")):
        return "n/a"
    s = f"{r['temp_f']:.0f}F {r['wind_mph']:.0f}mph"
    if r.get("precip_in", 0) and r["precip_in"] >= 0.05:
        s += f" {r['precip_in']:.2f}in"
    return s


def _target_week(df: pd.DataFrame, season: int | None, week: int | None) -> tuple[int, int]:
    season = season or current_season()
    if week is None:
        # The first week that is mostly unplayed (so a leftover Monday night game doesn't hold it back).
        by_week = df[df["season"] == season].groupby("week")["result"].apply(lambda r: r.isna().mean())
        open_weeks = by_week[by_week > 0.5]
        if open_weeks.empty:
            raise typer.BadParameter(f"No unplayed games found for {season}; pass --week.")
        week = int(open_weeks.index.min())
    return season, week


def _latest_model(df: pd.DataFrame, use_saved: bool = False) -> GameModels:
    if use_saved and MODEL_PATH.exists():
        return GameModels.load(MODEL_PATH)
    return GameModels().fit(df[df["season"] >= FIRST_TRAIN_SEASON])


@app.command()
def predict(
    season: int = typer.Option(None, help="Season (default: current)"),
    week: int = typer.Option(None, help="Week (default: next week with unplayed games)"),
    refresh: bool = typer.Option(True, help="Refresh current-season data (injuries, lines, QBs, weather)"),
    use_saved: bool = typer.Option(False, help="Use the saved model instead of retraining on the latest results"),
):
    """Predict a week's games and save the report to reports/."""
    df = get_dataset(refresh=refresh)
    season, week = _target_week(df, season, week)
    model = _latest_model(df, use_saved)
    games = df[(df["season"] == season) & (df["week"] == week)].sort_values("kickoff")
    if games.empty:
        raise typer.BadParameter(f"No games found for {season} week {week}.")
    p = pd.concat([games, model.predict(games)], axis=1)

    p["spread_edge"] = p["pred_margin"] - p["spread_line"]
    p["total_edge"] = p["pred_total"] - p["total_line"]
    rows = []
    for _, r in p.iterrows():
        h, a = r["home_team"], r["away_team"]
        fav_win = max(r["win_prob"], 1 - r["win_prob"])
        rows.append({
            "kickoff": r["kickoff"].strftime("%a %m/%d %H:%M"),
            "matchup": f"{a} @ {h}" + (" (N)" if r.get("neutral") == 1 else ""),
            "qbs": f"{r['away_qb_name']} / {r['home_qb_name']}",
            "pick": f"{h if r['win_prob'] >= 0.5 else a} {fav_win:.0%}",
            "vegas_win": f"{h if r['mkt_home_prob'] >= 0.5 else a} {max(r['mkt_home_prob'], 1 - r['mkt_home_prob']):.0%}" if pd.notna(r["spread_line"]) else "-",
            "model_spread": _fmt_spread(h, a, r["pred_margin"]),
            "vegas_spread": _fmt_spread(h, a, r["spread_line"]),
            "ats_lean": (h if r["spread_edge"] > 0 else a) + (" *" if abs(r["spread_edge"]) >= 1.5 else "") if pd.notna(r["spread_edge"]) else "-",
            "model_total": f"{r['pred_total']:.1f}",
            "vegas_total": f"{r['total_line']:.1f}" if pd.notna(r["total_line"]) else "-",
            "ou_lean": ("Over" if r["total_edge"] > 0 else "Under") + (" *" if abs(r["total_edge"]) >= 1.5 else "") if pd.notna(r["total_edge"]) else "-",
            "weather": _weather_str(r),
            "final": f"{a} {r['away_score']:.0f} - {h} {r['home_score']:.0f}" if pd.notna(r["result"]) else "",
        })
    report = pd.DataFrame(rows)
    if not report["final"].any():
        report = report.drop(columns="final")

    t = Table(title=f"{season} Week {week} predictions (model trained through {model.trained_through})")
    for c in report.columns:
        t.add_column(c, justify="left" if c in ("matchup", "qbs", "kickoff", "weather") else "right")
    for _, r in report.iterrows():
        t.add_row(*[str(v) for v in r])
    console.print(t)
    console.print("* = edge of 1.5+ points vs the Vegas line. Games without posted lines use the pure football model.")

    REPORTS_DIR.mkdir(exist_ok=True)
    stem = REPORTS_DIR / f"{season}_week{week:02d}"
    keep = ["game_id", "kickoff", "away_team", "home_team", "away_qb_name", "home_qb_name", "win_prob", "mkt_home_prob",
            "pred_margin", "spread_line", "spread_edge", "pred_total", "total_line", "total_edge",
            "pure_margin", "pure_total", "temp_f", "wind_mph", "precip_in", "is_indoor"]
    p[keep].to_csv(stem.with_suffix(".csv"), index=False)
    stem.with_suffix(".md").write_text(
        f"# {season} Week {week} predictions\n\nModel trained through {model.trained_through}.\n\n"
        + report.to_markdown(index=False) + "\n\n`*` = edge of 1.5+ points vs the Vegas line.\n", encoding="utf-8")
    console.print(f"[green]Saved {stem.with_suffix('.csv')} and {stem.with_suffix('.md')}[/green]")


def _record(hits: pd.Series) -> str:
    """W-L(-P) string from a series of True/False/None (None = push)."""
    w, l = int((hits == True).sum()), int((hits == False).sum())  # noqa: E712
    p = int(hits.isna().sum())
    pct = f" ({w / (w + l):.1%})" if w + l else ""
    return f"{w}-{l}" + (f"-{p}" if p else "") + pct


def _vs_line(edge: float, outcome: float):
    """True if the side the model leans to won vs the line, None on a push or no lean."""
    if pd.isna(edge) or pd.isna(outcome) or outcome == 0 or edge == 0:
        return None
    return bool((edge > 0) == (outcome > 0))


@app.command()
def review(
    season: int = typer.Option(None, help="Season (default: current)"),
    weeks: str = typer.Option("1-3", help="e.g. 1-3 or 1,2,5"),
):
    """Grade past weeks: predict each week as if before kickoff, then compare to final scores."""
    df = get_dataset()
    season = season or current_season()
    week_list = _parse_seasons(weeks)
    p = walk_forward_weeks(df, season, week_list)
    if p.empty:
        raise typer.BadParameter(f"No completed games for {season} weeks {weeks}.")

    p["su_model"] = [None if r == 0 else (w >= 0.5) == (r > 0) for w, r in zip(p["win_prob"], p["result"])]
    p["su_vegas"] = [None if r == 0 else (m >= 0.5) == (r > 0) for m, r in zip(p["mkt_home_prob"], p["result"])]
    p["ats_hit"] = [_vs_line(e, o) for e, o in zip(p["pred_margin"] - p["spread_line"], p["result"] - p["spread_line"])]
    p["ou_hit"] = [_vs_line(e, o) for e, o in zip(p["pred_total"] - p["total_line"], p["total"] - p["total_line"])]
    mark = lambda v: "-" if v is None else ("[green]✓[/green]" if v else "[red]✗[/red]")

    rows = []
    for week, g in p.groupby("week"):
        t = Table(title=f"{season} Week {week} (model trained through {g['trained_through'].iloc[0]})")
        for c in ("matchup", "final", "pick", "SU", "model spr", "vegas spr", "ATS lean", "ATS",
                  "model tot", "vegas tot", "actual", "O/U lean", "O/U"):
            t.add_column(c, justify="left" if c in ("matchup", "final") else "right")
        for _, r in g.sort_values("kickoff").iterrows():
            h, a = r["home_team"], r["away_team"]
            pick = h if r["win_prob"] >= 0.5 else a
            se, te = r["pred_margin"] - r["spread_line"], r["pred_total"] - r["total_line"]
            t.add_row(
                f"{a} @ {h}", f"{a} {r['away_score']:.0f} - {h} {r['home_score']:.0f}",
                f"{pick} {max(r['win_prob'], 1 - r['win_prob']):.0%}", mark(r["su_model"]),
                _fmt_spread(h, a, r["pred_margin"]), _fmt_spread(h, a, r["spread_line"]),
                (h if se > 0 else a) + (" *" if abs(se) >= 1.5 else ""), mark(r["ats_hit"]),
                f"{r['pred_total']:.1f}", f"{r['total_line']:.1f}", f"{r['total']:.0f}",
                ("Over" if te > 0 else "Under") + (" *" if abs(te) >= 1.5 else ""), mark(r["ou_hit"]),
            )
        console.print(t)
        rows.append((f"Week {week}", g))
    rows.append(("Total", p))

    s = Table(title=f"{season} weeks {weeks}: summary")
    for c in ("", "games", "winners: model", "winners: Vegas fav", "ATS", "O/U", "O/U edge 1.5+",
              "spread MAE model", "spread MAE Vegas", "total MAE model", "total MAE Vegas"):
        s.add_column(c, justify="right")
    for label, g in rows:
        big = (g["pred_total"] - g["total_line"]).abs() >= 1.5
        s.add_row(
            label, str(len(g)), _record(g["su_model"]), _record(g["su_vegas"]), _record(g["ats_hit"]),
            _record(g["ou_hit"]), _record(g.loc[big, "ou_hit"]),
            f"{(g['pred_margin'] - g['result']).abs().mean():.2f}", f"{(g['spread_line'] - g['result']).abs().mean():.2f}",
            f"{(g['pred_total'] - g['total']).abs().mean():.2f}", f"{(g['total_line'] - g['total']).abs().mean():.2f}",
            style="bold" if label == "Total" else None,
        )
    console.print(s)
    console.print("Lines are closing lines. ATS/O-U records exclude pushes; break-even at -110 is 52.4%.")

    REPORTS_DIR.mkdir(exist_ok=True)
    out = REPORTS_DIR / f"review_{season}_weeks_{weeks.replace(',', '_')}.csv"
    p[["game_id", "week", "away_team", "home_team", "away_score", "home_score", "win_prob", "mkt_home_prob",
       "pred_margin", "spread_line", "result", "pred_total", "total_line", "total",
       "su_model", "su_vegas", "ats_hit", "ou_hit"]].to_csv(out, index=False)
    console.print(f"[green]Saved {out}[/green]")


def _fetch_and_snapshot(df: pd.DataFrame) -> None:
    raw, remaining = odds.fetch_odds(BETTING["books"])
    matched, unmatched = odds.match_games(raw, df)
    for path in odds.snapshot(matched):
        console.print(f"Snapshot saved to {path}")
    per_book = ", ".join(f"{BOOK_NAMES.get(b, b)} {n}" for b, n in matched["book"].value_counts().items())
    console.print(f"Fetched {raw['event_id'].nunique() if len(raw) else 0} games, matched "
                  f"{matched['event_id'].nunique() if len(matched) else 0} (lines per book: {per_book or 'none'}). "
                  f"API credits remaining: {remaining}")
    for _, r in unmatched.drop_duplicates("event_id").iterrows():
        console.print(f"[yellow]  unmatched: {r['away_team']} @ {r['home_team']} {r['commence_time']}[/yellow]")


@app.command("odds")
def odds_cmd(refresh: bool = typer.Option(False, help="Refresh the schedule first")):
    """Fetch current sportsbook lines and append them to this week's snapshot (needs ODDS_API_KEY)."""
    _fetch_and_snapshot(get_dataset(refresh=refresh))


def _fmt_line(market: str, line: float) -> str:
    return f"{line:.1f}" if market == "total" else ("PK" if line == 0 else f"{line:+.1f}")


@app.command()
def bets(
    season: int = typer.Option(None, help="Season (default: current)"),
    week: int = typer.Option(None, help="Week (default: next week with unplayed games)"),
    lines: str = typer.Option("opening", help="Score against the 'opening' (first seen) or 'latest' snapshot"),
    fetch: bool = typer.Option(True, help="Fetch fresh odds first (uses ~3 API credits)"),
    refresh: bool = typer.Option(True, help="Refresh current-season data first"),
    dry_run: bool = typer.Option(False, help="Show the bet slip without writing to the ledger"),
):
    """Pick and size bets against sportsbook lines, write a bet slip, and paper-trade them in the ledger."""
    df = get_dataset(refresh=refresh)
    season, week = _target_week(df, season, week)
    if fetch:
        _fetch_and_snapshot(df)
    offers = odds.opening_lines(season, week) if lines == "opening" else odds.latest_lines(season, week)
    offers = offers[offers["book"].isin(BETTING["books"])] if not offers.empty else offers
    if offers.empty:
        raise typer.BadParameter(f"No odds snapshots for {season} week {week}; run `nflpred odds` first.")

    now_et = pd.Timestamp.now(tz="America/New_York").tz_localize(None)
    games = df[(df["season"] == season) & (df["week"] == week) & (df["kickoff"] > now_et)]
    model = _latest_model(df)
    scored = betting.score_against_lines(games, odds.reference_lines(offers), model)
    cands = betting.candidates(scored, offers)

    ledger = betting.load_ledger()
    bankroll = betting.current_bankroll(ledger)
    this_week = ledger[(ledger["season"] == season) & (ledger["week"] == week)]
    fresh = cands[~(cands["game_id"] + "_" + cands["market"]).isin(ledger["bet_id"])]
    picks = betting.select_bets(fresh, bankroll, already_staked=this_week["stake"].sum())

    books = ", ".join(BOOK_NAMES.get(b, b) for b in BETTING["books"])
    t = Table(title=f"{season} Week {week}: best of {books} ({lines} lines) "
                    f"(model trained through {model.trained_through}, bankroll ${bankroll:,.0f})")
    for c in ("kickoff", "matchup", "market", "bet", "price", "book", "model", "edge", "win %", "EV", "stake"):
        t.add_column(c, justify="left" if c in ("kickoff", "matchup", "bet", "book") else "right")
    staked = picks.set_index(picks["game_id"] + "_" + picks["market"])["stake"]
    rows = []
    for _, c in cands.sort_values(["kickoff", "market"]).iterrows():
        bid = c["game_id"] + "_" + c["market"]
        stake = staked.get(bid, 0.0)
        prior = this_week.loc[this_week["bet_id"] == bid, "stake"]
        row = {"kickoff": c["kickoff"].strftime("%a %m/%d %H:%M"), "matchup": c["matchup"], "market": c["market"],
               "bet": f"{c['side']} {_fmt_line(c['market'], c['line'])}", "price": f"{c['price']:+.0f}",
               "book": BOOK_NAMES.get(c["book"], c["book"]), "model": _fmt_line(c["market"], c["model_line"]), "edge": f"{c['edge']:.1f}",
               "win %": f"{c['p_win']:.1%}", "EV": f"{c['ev']:+.1%}",
               "stake": f"${stake:,.0f}" if stake else (f"(${prior.iloc[0]:,.0f} placed)" if len(prior) else "-")}
        rows.append(row)
        t.add_row(*row.values(), style="bold green" if stake else None)
    console.print(t)
    lo, hi = BETTING["spread_weeks"]
    spread_rule = (f"spreads in weeks {lo}-{hi} with edge >= {BETTING['spread_min_edge_pts']} pts"
                   + ("" if betting.in_spread_window(week) else " (not this week)"))
    console.print(f"Rules: totals with edge >= {BETTING['min_edge_pts']} pts; {spread_rule}; EV >= {BETTING['min_ev']:.0%}; "
                  f"{BETTING['kelly_fraction']:.0%} Kelly, max {BETTING['max_stake_pct']:.0%}/bet, "
                  f"{BETTING['max_week_exposure_pct']:.0%}/week.")
    if picks.empty:
        console.print("[yellow]No new bets this run.[/yellow]")

    REPORTS_DIR.mkdir(exist_ok=True)
    stem = REPORTS_DIR / f"{season}_week{week:02d}_bets"
    pd.DataFrame(rows).to_csv(stem.with_suffix(".csv"), index=False)
    stem.with_suffix(".md").write_text(
        f"# {season} Week {week} bet slip (best of {books}, {lines} lines)\n\n"
        f"Model trained through {model.trained_through}. Paper bankroll ${bankroll:,.0f}.\n\n"
        + (pd.DataFrame(rows).to_markdown(index=False) if rows else "No games with lines.")
        + "\n\nRows with a stake are this run's bets. Paper trades only: nothing is placed automatically.\n",
        encoding="utf-8")
    console.print(f"[green]Saved {stem.with_suffix('.md')}[/green]")

    if not dry_run and not picks.empty:
        ledger, added = betting.add_bets(ledger, picks)
        betting.save_ledger(ledger)
        console.print(f"[green]Added {len(added)} paper bets (${added['stake'].sum():,.0f}) to {LEDGER_PATH}[/green]")


@app.command()
def ledger(refresh: bool = typer.Option(True, help="Refresh current-season results first")):
    """Grade paper bets against final scores and closing lines, and show the track record."""
    df = get_dataset(refresh=refresh)  # always refresh, so a following `bets --no-refresh` sees current data
    led = betting.load_ledger()
    if led.empty:
        console.print("Ledger is empty; run `nflpred bets` first.")
        return
    led, n = betting.grade(led, df)
    betting.save_ledger(led)
    console.print(f"Graded {n} bets.")

    s = betting.summarize_ledger(led)
    if s.empty:
        console.print("No graded bets yet.")
        return
    t = Table(title="Paper-trading record (CLV = points the closing line moved our way)")
    for c in ("", "bets", "record", "win %", "staked", "P&L", "ROI", "avg CLV", "beat close"):
        t.add_column(c, justify="right")
    for label, r in s.iterrows():
        t.add_row(label, f"{r['bets']}", r["record"], f"{r['win_pct']:.1%}", f"${r['staked']:,.0f}",
                  f"${r['pnl']:+,.0f}", f"{r['roi']:+.1%}", f"{r['avg_clv']:+.2f}", f"{r['beat_close']:.0%}",
                  style="bold" if label == "ALL" else ("italic" if label.startswith("ALL ") else None))
    console.print(t)
    ok, why = betting.ready_for_real_money(led)
    console.print(f"Paper bankroll: ${betting.current_bankroll(led):,.0f}. Break-even at -110 is {BREAKEVEN:.1%}.")
    console.print(("[green]Ready for real money: [/green]" if ok else "[yellow]Not ready for real money: [/yellow]") + why)


if __name__ == "__main__":
    app()
