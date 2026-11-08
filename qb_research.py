"""
Backup quarterbacks and this-season-only ratings (Brian, 2026-11-08).

The ratings kept rating teams with a barely-played backup (Eagles with
McKee/Milton, Ravens with Fagnano) close to their starter-led selves, and
were slow on teams collapsing mid-season. This scores, walk-forward on
2016-2025 with the actual starting QBs:

  * spread MAE of the calibrated ratings vs final margins
  * how far off they are, on average, for a team starting a QB with under
    BACKUP_DB career dropbacks (the backup penalty needed)
  * betting the ratings' side at 3+ points off the closing line

for the current config, ratings from the current season only, and both
with a stronger QB term.

    python qb_research.py

Results (run 2026-11-08):
  current config     MAE 10.216, backups off by -3.6 pts (t -6.0), 3+ gap 51.2% ATS
  this season only   MAE 10.454, 3+ gap 47.2% ATS (worse, with or without
                     the backup penalty: ~48%)
  QB term x2         no better than current
  backup penalty -3.3 (fit on 2016-20) applied to 2021-25: MAE 10.22 -> 10.11,
                     3+ gap 51.3% -> ~54% ATS. Live in paper.BACKUP_PTS.
"""

from __future__ import annotations

import sys
from dataclasses import replace

import numpy as np
import pandas as pd

from backtest import calibrate, project
from nfl_data import graded_games, load_games, load_team_games
from nfl_power import PowerConfig

BACKUP_DB = 150
FIRST, LAST = 2015, 2025
CONFIGS = {
    "current": PowerConfig(),
    "this season only": PowerConfig(offseason_weeks=10_000.0),
    "current, QB term x2": PowerConfig(alpha_qb=300.0, alpha=1200.0),
    "this season, QB term x2": PowerConfig(offseason_weeks=10_000.0, alpha_qb=300.0,
                                           alpha=1200.0),
}


def career_dropbacks(tg: pd.DataFrame) -> pd.DataFrame:
    """Each QB's dropbacks in team-games before each game (all seasons)."""
    t = tg[["game_id", "season", "week", "team", "qb_id", "n_db"]].dropna(subset=["qb_id"])
    t = t.sort_values(["season", "week"])
    t["before"] = t.groupby("qb_id")["n_db"].cumsum() - t["n_db"]
    return t


def score(fr: pd.DataFrame, g: pd.DataFrame, cdb: pd.DataFrame) -> dict:
    fr = fr.merge(g[["game_id", "home_qb_id", "away_qb_id"]], on="game_id")
    b = cdb.set_index(["game_id", "qb_id"])["before"]
    for s in ("home", "away"):
        fr[f"{s}_db"] = [b.get((i, q), 0.0) if isinstance(q, str) else np.nan
                         for i, q in zip(fr.game_id, fr[f"{s}_qb_id"])]
    fr = fr[fr.season >= FIRST + 1]
    res = fr["margin"] - fr["model_margin"]
    out = {"games": len(fr), "MAE": round(float(res.abs().mean()), 3)}
    # backup side, from that team's view (one backup only)
    hb = fr.home_db < BACKUP_DB
    ab = fr.away_db < BACKUP_DB
    one = hb ^ ab
    side = np.where(hb, 1, -1)
    r = (res * side)[one]
    out["backup games"] = int(one.sum())
    out["backup team vs model (pts)"] = round(float(r.mean()), 2)
    out["t"] = round(float(r.mean() / (r.std() / np.sqrt(len(r)))), 2)
    gap = fr["model_margin"] - fr["spread"]
    big = gap.abs() >= 3
    ats = np.sign(gap[big]) * (fr["margin"] - fr["spread"])[big]
    out["3+ gap ATS"] = f"{(ats > 0).sum()}-{(ats < 0).sum()} ({(ats > 0).sum() / max(1, (ats != 0).sum()):.1%})"
    gb = big & one
    atsb = np.sign(gap[gb]) * (fr["margin"] - fr["spread"])[gb]
    out["3+ gap ATS, backup games"] = f"{(atsb > 0).sum()}-{(atsb < 0).sum()}"
    return out, fr


if __name__ == "__main__":
    games = load_games()
    g = graded_games(games[games.game_type == "REG"])
    g = g[(g.season >= FIRST) & (g.season <= LAST)]
    tg = load_team_games(range(FIRST - 2, LAST + 1))
    cdb = career_dropbacks(load_team_games(range(2006, LAST + 1)))
    pick = sys.argv[1:] or list(CONFIGS)
    for name in pick:
        fr = calibrate(project(tg, g, sorted(g.season.unique()), CONFIGS[name]))
        o, fr2 = score(fr, g, cdb)
        fr2.to_pickle(f"/tmp/qbr_{name.replace(' ', '_').replace(',', '')}.pkl")
        print(name, o, flush=True)
