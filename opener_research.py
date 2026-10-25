"""
Do the ratings beat the OPENING spread, and does the line move their way?

The closing spread is too sharp for public ratings (backtest.py,
spread_research.py). Openers are softer: they are set before the market
has bet into them. If the model disagrees with an opener and the line then
moves toward the model, betting early is an edge even though the close is
efficient (closing line value).

Needs historical opening lines, which nflverse does not have. This reads
the free AusSportsBetting NFL file (aussportsbetting.com/historical_data/
nfl.xlsx; opening lines from 2014). The site blocks the sandbox and GitHub
Actions, so download it by hand; it is not committed.

Result (2026-10-25): no. Betting the model's side of the opener went about
50-53% in 2014-17 and 2018-21 at every gap size. Lines do move toward the
model in 2022-26, but that era's openers are often lookahead lines posted
before the previous week was played, so the move is the market catching up
on results the model had already seen; with ratings one week staler the
bets go about 50%. Starting QBs are not passed in (they are only known at
kickoff, not when the line opens).

    python opener_research.py path/to/nfl.xlsx
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from backtest import calibrate, project
from nfl_data import graded_games, load_games, load_team_games
from nfl_power import PowerConfig
from picks import ODDS_TEAM

ERAS = {"2014-17": (2014, 2017), "2018-21": (2018, 2021),
        "2022-26": (2022, 2026)}
ALIASES = {"Washington Redskins": "WAS", "Washington Football Team": "WAS",
           "Washington": "WAS", "Oakland Raiders": "LV",
           "San Diego Chargers": "LAC", "St. Louis Rams": "LA",
           "Los Angeles Rams": "LA"}


def load_ausb(path: str) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None)
    hdr = raw.index[raw.apply(lambda r: r.astype(str).str.contains(
        "Home Team").any(), axis=1)][0]
    d = pd.read_excel(path, header=hdr)
    d.columns = [str(c).strip() for c in d.columns]
    print("columns:", list(d.columns))
    name = {**ODDS_TEAM, **ALIASES}
    d["home"] = d["Home Team"].map(name)
    d["away"] = d["Away Team"].map(name)
    miss = d.loc[d.home.isna() | d.away.isna(), ["Home Team", "Away Team"]]
    if len(miss):
        print("unmapped teams:", miss.stack().unique()[:20])
    d["date"] = pd.to_datetime(d["Date"], errors="coerce").dt.normalize()
    # Home Line is the home team's handicap (-3.5 = home favoured); nflverse
    # uses home margin (+3.5), so flip the sign.
    d["open"] = -pd.to_numeric(d["Home Line Open"], errors="coerce")
    d["close_ausb"] = -pd.to_numeric(d["Home Line Close"], errors="coerce")
    return d[["date", "home", "away", "open", "close_ausb"]].dropna()


def frame(ausb: pd.DataFrame, stale_weeks: int = 0) -> pd.DataFrame:
    """`stale_weeks=1` rates each game with data through two weeks before,
    matching what was known when a lookahead line opened."""
    games = load_games()
    games = games[games.game_type == "REG"]
    g = graded_games(games)
    g = g[g.season >= 2012]
    g["home_qb_id"] = np.nan
    g["away_qb_id"] = np.nan
    tg = load_team_games(range(2011, int(g.season.max()) + 1))
    gg = g.assign(week=(g["week"] - stale_weeks).clip(lower=1))
    fr = calibrate(project(tg, gg, sorted(gg.season.unique()), PowerConfig()))
    fr = fr.drop(columns=["week"]).merge(
        g[["game_id", "gameday", "week"]], on="game_id")
    fr["date"] = pd.to_datetime(fr["gameday"]).dt.normalize()
    m = fr.merge(ausb, on=["home", "away"], how="inner")
    m = m[(m.date_x - m.date_y).abs() <= pd.Timedelta(days=1)]
    m = m.drop_duplicates("game_id")
    print(f"matched {len(m)} of {len(fr)} games; close agrees within 0.5 "
          f"on {((m.close_ausb - m.spread).abs() <= 0.5).mean():.0%}")
    m["gap_open"] = m["model_margin"] - m["open"]
    m["gap_close"] = m["model_margin"] - m["spread"]
    m["move"] = m["spread"] - m["open"]
    m["res_open"] = m["margin"] - m["open"]
    m["res_close"] = m["margin"] - m["spread"]
    return m


def slope(x, y):
    x = np.asarray(x, float) - np.mean(x)
    y = np.asarray(y, float)
    b = (x * (y - y.mean())).sum() / (x ** 2).sum()
    r = y - y.mean() - b * x
    se = np.sqrt((r ** 2).sum() / (len(y) - 2) / (x ** 2).sum())
    return round(b, 3), round(b / se, 2)


def report(m: pd.DataFrame) -> None:
    for era, (a, b) in ERAS.items():
        d = m[(m.season >= a) & (m.season <= b)]
        print(f"\n== {era}: {len(d)} games ==")
        print("  line move on model-vs-open gap   (slope, t):",
              slope(d.gap_open, d.move))
        print("  result-vs-open on model gap      (slope, t):",
              slope(d.gap_open, d.res_open))
        print("  result-vs-close on model gap     (slope, t):",
              slope(d.gap_close, d.res_close))
        for k in (1.0, 2.0, 3.0, 4.0):
            s = d[d.gap_open.abs() >= k]
            side = np.sign(s.gap_open)
            ro = side * s.res_open
            rc = side * s.res_close
            mv = side * s.move
            n = int((ro != 0).sum())
            w = int((ro > 0).sum())
            print(f"  |gap|>={k}: {len(s):4d} bets, vs open {w}-{n - w} "
                  f"({w / n if n else float('nan'):.3f}), vs close "
                  f"{(rc > 0).sum()}-{(rc < 0).sum()}, line moved our way "
                  f"{(mv > 0).mean():.0%} / against {(mv < 0).mean():.0%}, "
                  f"avg {mv.mean():+.2f} pts")


if __name__ == "__main__":
    lines = load_ausb(sys.argv[1])
    for stale in (0, 1):
        print(f"\n######## ratings {stale} week(s) staler than the game")
        report(frame(lines, stale))
