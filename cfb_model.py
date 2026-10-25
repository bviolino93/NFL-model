"""
College football model: how far the result should land from the betting
line, from team ratings the market may not have fully priced.

The college counterpart of nfl_combined.py. Features (cfb_ratings.py):

  spreads  points rating margin, opponent-adjusted EPA/play, success rate,
           pass and rush EPA matchups, Elo difference, home field
  totals   points rating total, both offenses' and defenses' EPA/play,
           success rate, pass and rush EPA, and pace (plays per game)

A ridge on standardised features (plus the line itself) predicts
result-minus-line, so it only moves the fair line by what history says
those ratings add on top of the market. Trained against the OPENING line:
picks come from the Sunday-to-Wednesday runs, when lines are fresh.

Backtest (python cfb_model.py), season-by-season walk-forward, FBS games:

  * vs the closing line nothing holds up (correlation with result minus
    close ~0.01): the close already knows these ratings.
  * vs the opening line, totals: the model's lean predicts which way the
    total moves by kickoff (corr ~0.15 in 2017-21 and 2022-25) and picks
    went ~52-53% at a 2+ point lean; spreads ~54-56% at 2+ points on a few
    hundred games. Thin, so stakes stay small and the dashboard tracks it.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge

import cfb_data
import cfb_ratings

SPREAD_FEATS = ["pred_margin", "net_epa", "net_sr", "net_pass_epa",
                "net_rush_epa", "elo_diff", "home_flag"]
TOTAL_FEATS = ["pred_total", "sum_epa", "sum_sr", "sum_pass_epa",
               "sum_rush_epa", "sum_plays"]
ALPHA = 2000.0
FIRST_TRAIN = 2015


def features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    g["home_flag"] = (~g.neutral_site.astype(bool)).astype(float)
    for s in ("epa", "sr", "pass_epa", "rush_epa"):
        # home offense vs away defense, minus the reverse
        g[f"net_{s}"] = (g[f"h_off_{s}"] - g[f"a_def_{s}"]) - (g[f"a_off_{s}"] - g[f"h_def_{s}"])
        g[f"sum_{s}"] = g[f"h_off_{s}"] + g[f"a_def_{s}"] + g[f"a_off_{s}"] + g[f"h_def_{s}"]
    g["sum_plays"] = g.h_off_plays + g.a_off_plays + g.h_def_plays + g.a_def_plays
    g["elo_diff"] = g.home_pregame_elo - g.away_pregame_elo
    return g


class LineModel:
    """Predicts result minus market in two steps, so each piece reads as a
    football claim. First, three opinions of the game in points:

      eff     play-by-play efficiency (EPA, success rate, pass and rush
              matchups, pace for totals), fitted to results
      scores  the points rating (cfb_ratings.walk_forward)
      elo     Elo difference, for spreads

    then a ridge on how far each opinion sits from the market. kind is
    "spread" (home margin) or "total"."""

    def __init__(self, kind: str):
        self.kind = kind
        self.feats = SPREAD_FEATS if kind == "spread" else TOTAL_FEATS
        if kind == "spread":
            self.eff_feats = ["net_epa", "net_sr", "net_pass_epa", "net_rush_epa", "home_flag"]
            self.opinions = ["eff", "scores", "elo"]
        else:
            self.eff_feats = ["sum_epa", "sum_sr", "sum_pass_epa", "sum_rush_epa", "sum_plays"]
            self.opinions = ["eff", "scores"]

    def views(self, d: pd.DataFrame) -> pd.DataFrame:
        """Each opinion of the home margin (or total), in points."""
        v = pd.DataFrame(index=d.index)
        v["eff"] = self.eff.predict(d[self.eff_feats].values.astype(float))
        v["scores"] = d["pred_margin" if self.kind == "spread" else "pred_total"].values
        if self.kind == "spread":
            v["elo"] = self.elo.predict(d[["elo_diff", "home_flag"]].values.astype(float))
        return v

    def _gaps(self, d, market):
        return self.views(d).values - np.asarray(market, float)[:, None]

    def fit(self, d: pd.DataFrame, market: np.ndarray, result: np.ndarray):
        self.eff = LinearRegression().fit(d[self.eff_feats].values.astype(float), result)
        if self.kind == "spread":
            self.elo = LinearRegression().fit(d[["elo_diff", "home_flag"]].values.astype(float), result)
        X = self._gaps(d, market)
        self.r = Ridge(alpha=ALPHA).fit(X, result - market)
        return self

    def predict(self, d: pd.DataFrame, market: np.ndarray) -> np.ndarray:
        return self.r.predict(self._gaps(d, market))

    def parts(self, row: pd.Series, market: float) -> dict:
        """Each opinion's push on the prediction in points, with the
        opinion itself: {"eff": (push, eff_points), ...}."""
        d = row.to_frame().T
        v = self.views(d).iloc[0]
        return {k: (float(c * (v[k] - market)), float(v[k]))
                for k, c in zip(self.opinions, self.r.coef_)}


def market_home_margin(spread: pd.Series) -> pd.Series:
    """Betting spread (home -7) to expected home margin (+7)."""
    return -spread


def training_rows(g, kind, line="open"):
    col = f"{line}_{kind}"
    tgt = "margin" if kind == "spread" else "points"
    feats = SPREAD_FEATS if kind == "spread" else TOTAL_FEATS
    d = g.dropna(subset=feats + [col, tgt])
    mkt = market_home_margin(d[col]) if kind == "spread" else d[col]
    return d, mkt.values.astype(float), d[tgt].values.astype(float)


def fit_models(g: pd.DataFrame, line="open") -> dict:
    out = {}
    for kind in ("spread", "total"):
        d, mkt, y = training_rows(g[g.season >= FIRST_TRAIN], kind, line)
        out[kind] = LineModel(kind).fit(d, mkt, y)
    return out


def load_features(seasons=None) -> pd.DataFrame:
    cur = cfb_data.current_season()
    seasons = list(seasons or range(cfb_data.FIRST_SEASON, cur + 1))
    games = cfb_data.load_games(seasons)
    pbp = cfb_data.load_pbp_games(range(max(seasons[0], cfb_data.PBP_FIRST), seasons[-1] + 1))
    return features(cfb_ratings.ratings(games, pbp))


def backtest(g: pd.DataFrame, train_line="open", test_line="open", first_test=2017):
    rows = []
    for kind in ("spread", "total"):
        d, mkt, y = training_rows(g, kind, test_line)
        dt, mt, yt = training_rows(g, kind, train_line)
        pred = np.full(len(d), np.nan)
        for S in range(first_test, int(d.season.max()) + 1):
            te = (d.season == S).values
            tr = (dt.season < S).values & (dt.season >= FIRST_TRAIN).values
            if te.sum() == 0 or tr.sum() < 500:
                continue
            m = LineModel(kind).fit(dt[tr], mt[tr], yt[tr])
            pred[te] = m.predict(d[te], mkt[te])
        resid = y - mkt
        for era, (a, b) in {"2017-21": (2017, 2021), "2022-25": (2022, 2025)}.items():
            e = d.season.between(a, b).values & ~np.isnan(pred) & (resid != 0)
            r = {"kind": kind, "era": era, "n": int(e.sum()),
                 "corr": round(float(np.corrcoef(pred[e], resid[e])[0, 1]), 3)}
            for th in (1, 1.5, 2, 3):
                m = e & (np.abs(pred) >= th)
                win = (np.sign(pred[m]) == np.sign(resid[m])).mean() if m.any() else np.nan
                r[f">={th}"] = f"{win:.3f}/{m.sum()}"
            rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    g = load_features(range(2006, 2026))
    for tr, te in (("open", "open"), ("open", "close"), ("close", "close")):
        print(f"\ntrained vs {tr}, scored vs {te}")
        print(backtest(g, tr, te).to_string(index=False))
    if len(sys.argv) > 1:
        g.to_parquet(sys.argv[1])
