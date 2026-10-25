"""
Walk-forward backtest against CLOSING spreads and totals.

Every projection for week N is made from ratings fitted only on games
before week N, and from the quarterback nflverse lists as that game's
starter (knowable pregame from the injury report). Home field is the mean
home margin of non-neutral games in the previous three seasons.

The question asked of each model is the one that matters for betting:

    actual - close  ~  b * (model - close)

b is the share of the model's disagreement with the closing line that
showed up in results. b > 0 with a t-stat that clears 2 means the model
knows something the close does not. ATS / over-under records at edge
thresholds are reported alongside, against the 52.4% break-even at -110.

Usage
-----
    python backtest.py                     # default config, 2014-now
    python backtest.py --sweep             # tune on 2014-2021, test after
    python backtest.py --first 2016 --half-life 8 --no-qb
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import pandas as pd

from nfl_data import graded_games, load_games, load_team_games
from nfl_power import PowerConfig, fit_power

BREAK_EVEN = 110.0 / 210.0


def home_field_before(games: pd.DataFrame, season: int, week: int,
                      seasons_back: int = 3, default: float = 1.8) -> float:
    g = games[((games["season"] < season) |
               ((games["season"] == season) & (games["week"] < week)))
              & (games["season"] >= season - seasons_back)
              & ~games["neutral"]]
    if len(g) < 100:
        return default
    return float(g["margin"].mean())


def project(tg: pd.DataFrame, games: pd.DataFrame, seasons,
            cfg: PowerConfig) -> pd.DataFrame:
    """One row per graded game in `seasons` with model and market numbers."""
    out = []
    target = games[games["season"].isin(seasons)]
    for (s, w), wk in target.groupby(["season", "week"], sort=True):
        hfa = home_field_before(games, s, w)
        try:
            r = fit_power(tg, int(s), int(w), hfa, cfg)
        except ValueError:
            continue
        for g in wk.itertuples():
            if g.home_team not in r.teams or g.away_team not in r.teams:
                continue
            hq = g.home_qb_id if isinstance(g.home_qb_id, str) else None
            aq = g.away_qb_id if isinstance(g.away_qb_id, str) else None
            out.append({
                "season": s, "week": w, "game_id": g.game_id,
                "game_type": g.game_type,
                "home": g.home_team, "away": g.away_team,
                "spread": g.spread_line, "total": g.total_line,
                "margin": g.margin, "points": g.points,
                "hfa": 0.0 if g.neutral else hfa,
                "model_edge": r.projected_margin(
                    g.home_team, g.away_team, neutral=True,
                    home_qb=hq, away_qb=aq),
                "epa_sum": r.projected_epa_sum(g.home_team, g.away_team,
                                               hq, aq),
            })
    return pd.DataFrame(out)


def calibrate(fr: pd.DataFrame) -> pd.DataFrame:
    """
    Put the ratings on the points scale with maps fitted on PRIOR seasons
    only, so the calibration itself is walk-forward:

        margin - hfa ~ k * model_edge      (EPA understates point margins,
                                            and the ridge shrinks; k fixes
                                            both)
        points       ~ a + b * epa_sum

    The first season has no prior and is left ungraded.
    """
    fr = fr.copy()
    fr["model_margin"] = np.nan
    fr["model_total"] = np.nan
    for s in sorted(fr["season"].unique()):
        past = fr[fr["season"] < s]
        if len(past) < 200:
            continue
        e = past["model_edge"].to_numpy(float)
        k = float(e @ (past["margin"] - past["hfa"]).to_numpy(float) / (e @ e))
        b, a = np.polyfit(past["epa_sum"], past["points"], 1)
        m = fr["season"] == s
        fr.loc[m, "model_margin"] = fr.loc[m, "hfa"] + k * fr.loc[m, "model_edge"]
        fr.loc[m, "model_total"] = a + b * fr.loc[m, "epa_sum"]
    return fr.dropna(subset=["model_margin"])


def _slope(edge, resid):
    """resid = b * edge, no intercept; returns (b, t)."""
    e = np.asarray(edge, float)
    r = np.asarray(resid, float)
    ok = np.isfinite(e) & np.isfinite(r)
    e, r = e[ok], r[ok]
    if len(e) < 30 or (e @ e) == 0:
        return float("nan"), float("nan")
    b = float(e @ r / (e @ e))
    s2 = float(((r - b * e) ** 2).sum()) / max(len(e) - 1, 1)
    se = math.sqrt(s2 / (e @ e))
    return b, b / se if se > 0 else float("nan")


def _record(edge, resid, thr):
    e = np.asarray(edge, float)
    r = np.asarray(resid, float)
    m = np.isfinite(e) & np.isfinite(r) & (np.abs(e) >= thr) & (r != 0)
    if not m.any():
        return 0, 0, float("nan")
    win = int(((np.sign(e[m]) * r[m]) > 0).sum())
    n = int(m.sum())
    return win, n - win, win / n


def score(fr: pd.DataFrame, thresholds=(0, 1, 2, 3, 4)) -> dict:
    """Spread and totals verdicts against the close."""
    res = {"n_games": len(fr)}
    se = fr["model_margin"] - fr["spread"]
    sr = fr["margin"] - fr["spread"]
    res["spread_b"], res["spread_t"] = _slope(se, sr)
    res["spread_mae_model"] = float((fr["model_margin"] - fr["margin"]).abs().mean())
    res["spread_mae_close"] = float(sr.abs().mean())
    for t in thresholds:
        w, l, p = _record(se, sr, t)
        res[f"ats_{t}"] = (w, l, p)
    if "model_total" in fr and fr["model_total"].notna().sum() > 100:
        f2 = fr.dropna(subset=["model_total"])
        te = f2["model_total"] - f2["total"]
        tr = f2["points"] - f2["total"]
        res["total_b"], res["total_t"] = _slope(te, tr)
        res["total_mae_model"] = float((f2["model_total"] - f2["points"]).abs().mean())
        res["total_mae_close"] = float(tr.abs().mean())
        for t in thresholds:
            w, l, p = _record(te, tr, t)
            res[f"ou_{t}"] = (w, l, p)
    return res


def report(res: dict, label: str = "") -> str:
    def rec(k):
        w, l, p = res.get(k, (0, 0, float("nan")))
        return f"{w}-{l} ({p:.1%})" if w + l else "-"
    lines = [f"== {label} ({res['n_games']:,} games)" if label else ""]
    lines.append(
        f"SPREAD  b {res['spread_b']:+.3f}  t {res['spread_t']:+.2f}   "
        f"MAE model {res['spread_mae_model']:.2f} vs close "
        f"{res['spread_mae_close']:.2f}")
    lines.append("        ATS by |edge|: " + "  ".join(
        f">={t}: {rec(f'ats_{t}')}" for t in (0, 1, 2, 3, 4)))
    if "total_b" in res:
        lines.append(
            f"TOTAL   b {res['total_b']:+.3f}  t {res['total_t']:+.2f}   "
            f"MAE model {res['total_mae_model']:.2f} vs close "
            f"{res['total_mae_close']:.2f}")
        lines.append("        O/U by |edge|: " + "  ".join(
            f">={t}: {rec(f'ou_{t}')}" for t in (0, 1, 2, 3, 4)))
    return "\n".join(lines)


def run(cfg: PowerConfig, first: int, last: int, train_from: int):
    games = graded_games(load_games())
    tg = load_team_games(range(train_from, last + 1))
    fr = project(tg, games, range(first, last + 1), cfg)
    return calibrate(fr)


def sweep(first: int, split: int, last: int, train_from: int):
    """Tune on seasons < split, then score the winner once on >= split."""
    games = graded_games(load_games())
    tg = load_team_games(range(train_from, last + 1))
    grid = []
    for hl in (4, 6, 10, 16):
        for off in (8, 16, 30):
            for alpha in (300, 600, 1200):
                for qb in (True, False):
                    grid.append(PowerConfig(half_life=hl, offseason_weeks=off,
                                            alpha=alpha, alpha_qb=alpha,
                                            use_qb=qb))
    rows = []
    t0 = time.time()
    for i, cfg in enumerate(grid):
        fr = calibrate(project(tg, games, range(first, last + 1), cfg))
        tr, te = fr[fr["season"] < split], fr[fr["season"] >= split]
        a, b = score(tr), score(te)
        rows.append({"half_life": cfg.half_life, "offseason": cfg.offseason_weeks,
                     "alpha": cfg.alpha, "qb": cfg.use_qb,
                     "train_mae": a["spread_mae_model"], "train_b": a["spread_b"],
                     "train_t": a["spread_t"], "test_mae": b["spread_mae_model"],
                     "test_b": b["spread_b"], "test_t": b["spread_t"],
                     "train_tot_mae": a.get("total_mae_model"),
                     "test_tot_mae": b.get("total_mae_model"),
                     "test_tot_b": b.get("total_b"), "test_tot_t": b.get("total_t")})
        print(f"[{i + 1}/{len(grid)} {time.time() - t0:.0f}s] {rows[-1]}",
              flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=2014)
    ap.add_argument("--last", type=int, default=2026)
    ap.add_argument("--train-from", type=int, default=2012)
    ap.add_argument("--split", type=int, default=2022)
    ap.add_argument("--half-life", type=float, default=PowerConfig.half_life)
    ap.add_argument("--offseason", type=float, default=PowerConfig.offseason_weeks)
    ap.add_argument("--alpha", type=float, default=PowerConfig.alpha)
    ap.add_argument("--no-qb", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.sweep:
        df = sweep(a.first, a.split, a.last, a.train_from)
        df.to_csv(a.out or "sweep.csv", index=False)
        print(df.sort_values("train_mae").head(15).to_string())
        return
    cfg = PowerConfig(half_life=a.half_life, offseason_weeks=a.offseason,
                      alpha=a.alpha, alpha_qb=a.alpha, use_qb=not a.no_qb)
    fr = run(cfg, a.first, a.last, a.train_from)
    if a.out:
        fr.to_csv(a.out, index=False)
    print(report(score(fr), f"{a.first}-{a.last} all"))
    print(report(score(fr[fr["season"] < a.split]), f"{a.first}-{a.split - 1}"))
    print(report(score(fr[fr["season"] >= a.split]), f"{a.split}-{a.last}"))


if __name__ == "__main__":
    main()
