"""
College football team ratings, refit every week from games already played.

Two ratings, both weighted ridge fits with recent games counting most
(half-life in days, plus a discount per season back, since rosters turn
over every year):

walk_forward()      points: a margin rating per team (home field fitted),
                    and an offense/defense points rating for totals.
epa_walk_forward()  play-by-play: opponent-adjusted offense and defense
                    EPA per play, success rate, pass and rush EPA, plays.

Every game gets the ratings as they stood before its week (weeks start on
Tuesday), so the same code serves the backtest and next Saturday's games.
Settings from a sweep over 2009-2021 (cfb_model.py backtest): points
half-life 240 days, 0.6 per season back, light shrinkage, margins capped
at 45.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from scipy.sparse.linalg import lsqr

STATS = ("epa", "sr", "pass_epa", "rush_epa", "plays")


def _prep(games: pd.DataFrame) -> pd.DataFrame:
    g = games.copy()
    g["date"] = pd.to_datetime(g.start_date, utc=True).dt.tz_localize(None)
    g["slate"] = (g.date - pd.to_timedelta((g.date.dt.dayofweek - 1) % 7, unit="D")
                  ).dt.normalize()
    return g.sort_values("date").reset_index(drop=True)


def _weights(slate, slates, seasons, cur, hl_days, offseason):
    age = (np.datetime64(slate) - slates).astype("timedelta64[D]").astype(float)
    return 0.5 ** (age / hl_days) * offseason ** (cur - seasons)


def walk_forward(games: pd.DataFrame, hl_days=240, offseason=0.6, alpha=0.3,
                 cap=45, start_season=2009, lookback=3) -> pd.DataFrame:
    """Adds pred_margin (home), pred_home_pts, pred_away_pts, pred_total."""
    g = _prep(games)
    teams = pd.Index(pd.unique(pd.concat([g.home_id, g.away_id])))
    nt = len(teams)
    hi, ai = teams.get_indexer(g.home_id), teams.get_indexer(g.away_id)
    played = g.margin.notna().values
    home_all = (~g.neutral_site.values.astype(bool)).astype(float)
    res = np.full((len(g), 3), np.nan)
    for slate, idx in g[g.season >= start_season].groupby("slate").groups.items():
        idx = np.asarray(idx)
        cur = g.season.values[idx[0]]
        tr = np.where(played & (g.slate.values < np.datetime64(slate))
                      & (g.season.values >= cur - lookback))[0]
        if len(tr) < 200:
            continue
        w = _weights(slate, g.slate.values[tr], g.season.values[tr], cur,
                     hl_days, offseason)
        sw, n, home = np.sqrt(w), len(tr), home_all[tr]
        # margin = r_home - r_away + hfa * home
        X = csr_matrix((np.r_[np.ones(n), -np.ones(n)],
                        (np.r_[np.arange(n), np.arange(n)], np.r_[hi[tr], ai[tr]])),
                       shape=(n, nt))
        X = hstack([X, csr_matrix(home[:, None])]).tocsr()
        y = np.clip(g.margin.values[tr], -cap, cap)
        r = lsqr(X.multiply(sw[:, None]).tocsr(), y * sw, damp=np.sqrt(alpha))[0]
        # points scored = mu + offense + opposing defense + half of hfa
        X2 = csr_matrix((np.ones(4 * n), (np.r_[np.arange(2 * n), np.arange(2 * n)],
                                          np.r_[hi[tr], ai[tr], nt + ai[tr], nt + hi[tr]])),
                        shape=(2 * n, 2 * nt))
        X2 = hstack([X2, csr_matrix(np.r_[home, -home][:, None] * 0.5)]).tocsr()
        y2 = np.r_[g.home_points.values[tr], g.away_points.values[tr]].astype(float)
        sw2 = np.r_[sw, sw]
        mu = np.average(y2, weights=np.r_[w, w])
        p = lsqr(X2.multiply(sw2[:, None]).tocsr(), (y2 - mu) * sw2,
                 damp=np.sqrt(alpha))[0]
        h = home_all[idx]
        res[idx, 0] = r[hi[idx]] - r[ai[idx]] + r[-1] * h
        res[idx, 1] = mu + p[hi[idx]] + p[nt + ai[idx]] + 0.5 * p[-1] * h
        res[idx, 2] = mu + p[ai[idx]] + p[nt + hi[idx]] - 0.5 * p[-1] * h
    g["pred_margin"], g["pred_home_pts"], g["pred_away_pts"] = res.T
    g["pred_total"] = g.pred_home_pts + g.pred_away_pts
    return g


def epa_walk_forward(games: pd.DataFrame, pbp: pd.DataFrame, hl_days=150,
                     offseason=0.5, alpha=3.0, start_season=2015,
                     lookback=3) -> pd.DataFrame:
    """Adds h_off_<stat>, h_def_<stat>, a_off_<stat>, a_def_<stat> for each
    stat in STATS (defense = what it allows, so higher is worse), plus each
    stat's league average as lg_<stat>."""
    g = _prep(games)
    p = pbp.merge(g[["game_id", "home_id", "away_id", "slate", "season",
                     "neutral_site"]], on="game_id", suffixes=("_p", ""))
    p["off_id"] = np.where(p.is_home, p.home_id, p.away_id)
    p["def_id"] = np.where(p.is_home, p.away_id, p.home_id)
    p["home_flag"] = np.where(p.neutral_site.astype(bool), 0.0,
                              np.where(p.is_home, 1.0, -1.0))
    p = p.dropna(subset=["epa"]).reset_index(drop=True)
    teams = pd.Index(pd.unique(pd.concat([g.home_id, g.away_id, p.off_id, p.def_id])))
    nt = len(teams)
    oi, di = teams.get_indexer(p.off_id), teams.get_indexer(p.def_id)
    hi, ai = teams.get_indexer(g.home_id), teams.get_indexer(g.away_id)
    cols = [f"{s}_{k}_{st}" for s in "ha" for k in ("off", "def") for st in STATS]
    res = {c: np.full(len(g), np.nan) for c in cols + [f"lg_{st}" for st in STATS]}
    ps, pseason = p.slate.values, p.season.values
    for slate, idx in g[g.season >= start_season].groupby("slate").groups.items():
        idx = np.asarray(idx)
        cur = g.season.values[idx[0]]
        tr = np.where((ps < np.datetime64(slate)) & (pseason >= cur - lookback))[0]
        if len(tr) < 500:
            continue
        w = _weights(slate, ps[tr], pseason[tr], cur, hl_days, offseason) \
            * np.sqrt(p.plays.values[tr])
        sw, n = np.sqrt(w), len(tr)
        X = csr_matrix((np.ones(2 * n), (np.r_[np.arange(n), np.arange(n)],
                                         np.r_[oi[tr], nt + di[tr]])), shape=(n, 2 * nt))
        X = hstack([X, csr_matrix(p.home_flag.values[tr][:, None])]).tocsr()
        Xw = X.multiply(sw[:, None]).tocsr()
        for st in STATS:
            y = p[st].values[tr].astype(float)
            ok = ~np.isnan(y)
            mu = np.average(y[ok], weights=w[ok])
            b = lsqr(Xw[ok], (y[ok] - mu) * sw[ok], damp=np.sqrt(alpha))[0]
            res[f"h_off_{st}"][idx] = mu + b[hi[idx]]
            res[f"h_def_{st}"][idx] = mu + b[nt + hi[idx]]
            res[f"a_off_{st}"][idx] = mu + b[ai[idx]]
            res[f"a_def_{st}"][idx] = mu + b[nt + ai[idx]]
            res[f"lg_{st}"][idx] = mu
    for k, v in res.items():
        g[k] = v
    return g


def ratings(games: pd.DataFrame, pbp: pd.DataFrame, start_season=2009) -> pd.DataFrame:
    """Both ratings on every game from start_season, one row per game."""
    g = walk_forward(games, start_season=start_season)
    e = epa_walk_forward(games, pbp, start_season=max(start_season, 2015))
    keep = ["game_id"] + [c for c in e.columns if c[:2] in ("h_", "a_", "lg")]
    return g.merge(e[keep], on="game_id", how="left")
