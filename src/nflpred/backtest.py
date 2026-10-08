"""Walk-forward evaluation: for each season S, train on everything before S and predict S."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nflpred.config import FIRST_TRAIN_SEASON
from nflpred.models import GameModels

BREAKEVEN = 0.5238  # win rate needed to profit at -110


def walk_forward(df: pd.DataFrame, start: int, end: int, kind: str = "lgbm", verbose: bool = True) -> pd.DataFrame:
    df = df[df["season"] >= FIRST_TRAIN_SEASON]
    preds = []
    for season in range(start, end + 1):
        train = df[(df["season"] < season) & df["result"].notna()]
        test = df[(df["season"] == season) & df["result"].notna()]
        if test.empty:
            continue
        if verbose:
            print(f"  {season}: train {len(train)} games, test {len(test)}")
        model = GameModels(kind=kind).fit(train)
        preds.append(pd.concat([test, model.predict(test)], axis=1))
    return pd.concat(preds, ignore_index=True)


def walk_forward_weeks(df: pd.DataFrame, season: int, weeks: list[int], kind: str = "lgbm") -> pd.DataFrame:
    """Predict each week with a model trained only on games that kicked off before that week started."""
    df = df[df["season"] >= FIRST_TRAIN_SEASON]
    preds = []
    for week in weeks:
        test = df[(df["season"] == season) & (df["week"] == week) & df["result"].notna()]
        if test.empty:
            continue
        train = df[df["result"].notna() & (df["kickoff"] < test["kickoff"].min())]
        model = GameModels(kind=kind).fit(train)
        preds.append(pd.concat([test, model.predict(test)], axis=1).assign(trained_through=model.trained_through))
    return pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()


def _log_loss(y, p) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _pick_rate(edge: pd.Series, outcome: pd.Series, min_edge: float = 0.0) -> tuple[float, int]:
    """Hit rate of betting the side the model favors vs the line, excluding pushes."""
    mask = (edge.abs() > min_edge) & (outcome != 0) & edge.notna()
    if not mask.any():
        return float("nan"), 0
    return float((np.sign(edge[mask]) == np.sign(outcome[mask])).mean()), int(mask.sum())


def metrics(p: pd.DataFrame) -> dict:
    mkt = p[p["has_market"] == 1]
    decided = mkt[mkt["home_win"] != 0.5]
    y = decided["home_win"].to_numpy()
    cover = mkt["result"] - mkt["spread_line"]
    over = mkt["total"] - mkt["total_line"]
    ats, n_ats = _pick_rate(mkt["pred_margin"] - mkt["spread_line"], cover)
    ats_big, n_big = _pick_rate(mkt["pred_margin"] - mkt["spread_line"], cover, min_edge=1.5)
    pure_ats, _ = _pick_rate(mkt["pure_margin"] - mkt["spread_line"], cover)
    ou, _ = _pick_rate(mkt["pred_total"] - mkt["total_line"], over)
    ou_big, n_ou_big = _pick_rate(mkt["pred_total"] - mkt["total_line"], over, min_edge=1.5)
    return {
        "games": len(mkt),
        "ll_model": _log_loss(y, decided["win_prob"]),
        "ll_margin": _log_loss(y, decided["win_prob_margin"]),
        "ll_pure": _log_loss(y, decided["pure_win_prob"]),
        "ll_vegas": _log_loss(y, decided["mkt_home_prob"]),
        "brier_model": float(np.mean((decided["win_prob"] - y) ** 2)),
        "brier_vegas": float(np.mean((decided["mkt_home_prob"] - y) ** 2)),
        "acc_model": float(((decided["win_prob"] > 0.5) == (y == 1)).mean()),
        "acc_vegas": float(((decided["mkt_home_prob"] > 0.5) == (y == 1)).mean()),
        "mae_margin_model": float((mkt["pred_margin"] - mkt["result"]).abs().mean()),
        "mae_margin_pure": float((mkt["pure_margin"] - mkt["result"]).abs().mean()),
        "mae_margin_vegas": float(cover.abs().mean()),
        "mae_total_model": float((mkt["pred_total"] - mkt["total"]).abs().mean()),
        "mae_total_vegas": float(over.abs().mean()),
        "ats": ats, "ats_n": n_ats, "ats_1.5": ats_big, "ats_1.5_n": n_big, "ats_pure": pure_ats,
        "ou": ou, "ou_1.5": ou_big, "ou_1.5_n": n_ou_big,
    }


def summarize(preds: pd.DataFrame) -> pd.DataFrame:
    rows = {season: metrics(g) for season, g in preds.groupby("season")}
    rows["ALL"] = metrics(preds)
    return pd.DataFrame(rows).T


def calibration_table(preds: pd.DataFrame, col: str = "win_prob", bins: int = 10) -> pd.DataFrame:
    d = preds[(preds["has_market"] == 1) & (preds["home_win"] != 0.5)]
    cut = pd.cut(d[col], np.linspace(0, 1, bins + 1))
    return d.groupby(cut, observed=True).agg(n=("home_win", "size"), predicted=(col, "mean"), actual=("home_win", "mean"))
