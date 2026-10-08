"""Three prediction targets (margin, total, win probability) in two flavors.

Market-anchored models learn *adjustments* to the Vegas line:
  - margin: predicts (home margin - spread_line), then adds the spread back
  - total:  predicts (total - total_line), then adds the total back
  - win:    a classifier boosted from the market's implied log-odds (LightGBM init_score)
With strong regularization they stay close to Vegas unless the features justify moving.

Pure football models use no market inputs. They cover games whose lines aren't posted yet
and give an independent second opinion.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nflpred.config import LGBM_PARAMS, MARGIN_SIGMA
from nflpred.features.build import ALL_FEATURES, FOOTBALL_FEATURES


def _regressor(kind: str, objective: str = "l2"):
    """`objective="l1"` for residuals vs the line: robust to blowouts, and it tested best on MAE.
    Pure margin models keep L2 because their output feeds a win probability (a median would compress it)."""
    if kind == "ridge":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=200.0))
    return lgb.LGBMRegressor(**LGBM_PARAMS, objective=objective)


def _clip_prob(p) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)


@dataclass
class GameModels:
    kind: str = "lgbm"  # "lgbm" or "ridge"
    models: dict = field(default_factory=dict)
    trained_through: str = ""

    def fit(self, df: pd.DataFrame) -> "GameModels":
        played = df[df["result"].notna()]
        mkt = played[played["spread_line"].notna() & played["total_line"].notna()]

        self.models["margin"] = _regressor(self.kind, "l1").fit(mkt[ALL_FEATURES], mkt["result"] - mkt["spread_line"])
        self.models["total"] = _regressor(self.kind, "l1").fit(mkt[ALL_FEATURES], mkt["total"] - mkt["total_line"])
        self.models["pure_margin"] = _regressor(self.kind).fit(played[FOOTBALL_FEATURES], played["result"])
        self.models["pure_total"] = _regressor(self.kind).fit(played[FOOTBALL_FEATURES], played["total"])

        if self.kind == "lgbm":
            decided = mkt[mkt["home_win"] != 0.5]
            clf = lgb.LGBMClassifier(**LGBM_PARAMS)
            clf.fit(decided[ALL_FEATURES], decided["home_win"].astype(int),
                    init_score=logit(_clip_prob(decided["mkt_home_prob"])))
            self.models["win"] = clf

        last = played.sort_values("kickoff").iloc[-1]
        self.trained_through = f"{last['season']} week {last['week']}"
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        has_mkt = df["spread_line"].notna() & df["total_line"].notna()

        out["pure_margin"] = self.models["pure_margin"].predict(df[FOOTBALL_FEATURES])
        out["pure_total"] = self.models["pure_total"].predict(df[FOOTBALL_FEATURES])
        out["pure_win_prob"] = norm.cdf(out["pure_margin"] / MARGIN_SIGMA)

        X = df[ALL_FEATURES]
        out["pred_margin"] = np.where(has_mkt, df["spread_line"] + self.models["margin"].predict(X), out["pure_margin"])
        out["pred_total"] = np.where(has_mkt, df["total_line"] + self.models["total"].predict(X), out["pure_total"])
        out["win_prob_margin"] = norm.cdf(out["pred_margin"] / MARGIN_SIGMA)
        if "win" in self.models:
            raw = self.models["win"].predict(X, raw_score=True) + logit(_clip_prob(df["mkt_home_prob"].fillna(0.5)))
            out["win_prob_clf"] = expit(raw)
            # Blend with the margin-implied probability: equally accurate in backtests, and it keeps
            # the pick consistent with the predicted spread.
            out["win_prob"] = np.where(has_mkt, (out["win_prob_clf"] + out["win_prob_margin"]) / 2, out["pure_win_prob"])
        else:
            out["win_prob"] = out["win_prob_margin"]
        out["has_market"] = has_mkt.astype(int)
        return out

    def feature_importance(self, target: str = "margin") -> pd.Series:
        model = self.models[target]
        cols = ALL_FEATURES if target in ("margin", "total", "win") else FOOTBALL_FEATURES
        if self.kind == "lgbm":
            imp = model.booster_.feature_importance(importance_type="gain")
        else:
            imp = np.abs(model[-1].coef_)
        return pd.Series(imp, index=cols).sort_values(ascending=False)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "GameModels":
        with open(path, "rb") as f:
            return pickle.load(f)
