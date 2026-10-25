"""
College football data from the cfbfastR / sportsdataverse files on GitHub.

collegefootballdata.com is the usual source, but it needs an API key and is
blocked from some networks; cfbfastR republishes the same data as plain
files on GitHub, so read those and cache them on disk like nfl_data.py.

What comes out
--------------
load_games()       one row per game (FBS at home or away) with final scores,
                   neutral site, and the median opening and closing spread
                   and total across books (spread from the home side:
                   -7 means home favoured by 7, the betting convention).
load_pbp_games()   one row per offense per game: plays, EPA/play, success
                   rate, pass and rush EPA, outside garbage time.

Lines run 2006-2025 (cfbfastR's betting file; opening lines from 2012, none
for 2020 and about 60% of 2022-25). Play-by-play from 2014.
"""

from __future__ import annotations

import os
import time
import urllib.request

import numpy as np
import pandas as pd

RAW = "https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main"
SCHED_URL = f"{RAW}/schedules/csv/cfb_schedules_{{season}}.csv"
LINES_URL = f"{RAW}/betting/csv/cfb_line_odds.csv.gz"
PBP_URL = ("https://github.com/sportsdataverse/sportsdataverse-data/releases/"
           "download/cfbfastR_cfb_pbp/play_by_play_{season}.parquet")

CACHE_DIR = os.environ.get("CFB_CACHE_DIR", ".cache_cfb")
FIRST_SEASON = 2006
PBP_FIRST = 2014
WP_LOW, WP_HIGH = 0.05, 0.95
PBP_COLS = ["game_id", "pos_team", "home", "EPA", "success", "rush", "pass",
            "wp_before", "period"]


def current_season(now=None) -> int:
    now = pd.Timestamp(now or pd.Timestamp.now(tz="UTC"))
    return now.year if now.month >= 8 else now.year - 1


def _fetch(url: str, name: str, max_age_hours: float | None) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, name)
    fresh = os.path.exists(path) and (
        max_age_hours is None
        or time.time() - os.path.getmtime(path) < max_age_hours * 3600)
    if not fresh:
        tmp = path + ".part"
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, path)
    return path


def load_schedules(seasons) -> pd.DataFrame:
    cur = current_season()
    out = []
    for y in seasons:
        p = _fetch(SCHED_URL.format(season=y), f"sched_{y}.csv",
                   6 if y >= cur else None)
        out.append(pd.read_csv(p))
    s = pd.concat(out, ignore_index=True).drop_duplicates("game_id")
    s["neutral_site"] = s.neutral_site.fillna(False).astype(bool)
    return s


def _home_rows(lines: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    """Spread rows for the home side. The file names teams by abbreviation
    before 2020 and by school after, so map each label to the team id it
    shows up with in (nearly) all of its games."""
    g = sched.set_index("game_id")[["home_id", "away_id"]]
    sp = lines[lines.market_type == "spread"].join(g, on="game_id")
    sp = sp.dropna(subset=["home_id"])
    co = pd.concat([sp[["abbr", "game_id", "home_id"]].rename(columns={"home_id": "tid"}),
                    sp[["abbr", "game_id", "away_id"]].rename(columns={"away_id": "tid"})])
    co = co.drop_duplicates()
    cnt = co.groupby(["abbr", "tid"]).size().rename("n").reset_index()
    tot = co.groupby("abbr").game_id.nunique()
    best = cnt.sort_values("n").groupby("abbr").tail(1).set_index("abbr")
    best = best[best.n >= 0.8 * tot.reindex(best.index)]
    sp["tid"] = sp.abbr.map(best.tid)
    return sp[sp.tid == sp.home_id]


def load_games(seasons=None) -> pd.DataFrame:
    cur = current_season()
    seasons = list(seasons or range(FIRST_SEASON, cur + 1))
    s = load_schedules(seasons)
    lines = pd.read_csv(_fetch(LINES_URL, "lines.csv.gz", 24 * 7))
    lines = lines[lines.season.isin(seasons)
                  & lines.market_type.isin(["spread", "total"])]
    fbs = (s.home_division == "fbs") | (s.away_division == "fbs")
    s = s[fbs].copy()
    sp = _home_rows(lines, s)
    ov = lines[(lines.market_type == "total")
               & (lines.abbr.astype(str).str.lower() == "over")]
    out = s.set_index("game_id")
    out["close_spread"] = sp.groupby("game_id").lines.median()
    out["open_spread"] = sp.groupby("game_id").opening_lines.median()
    out["close_total"] = ov.groupby("game_id").lines.median()
    out["open_total"] = ov.groupby("game_id").opening_lines.median()
    out = out.reset_index()
    # A few opening lines in the file are typos (e.g. -299); a real open
    # rarely sits more than a touchdown or two from the close.
    out.loc[(out.open_spread - out.close_spread).abs() > 10, "open_spread"] = np.nan
    out.loc[(out.open_total - out.close_total).abs() > 15, "open_total"] = np.nan
    done = out.completed.fillna(False).astype(bool) & out.home_points.notna()
    out["margin"] = np.where(done, out.home_points - out.away_points, np.nan)
    out["points"] = np.where(done, out.home_points + out.away_points, np.nan)
    return out


def _pbp_season(path: str) -> pd.DataFrame:
    d = pd.read_parquet(path, columns=PBP_COLS)
    d = d[(d.rush == 1) | (d["pass"] == 1)].dropna(subset=["EPA"])
    garbage = ((d.wp_before < WP_LOW) | (d.wp_before > WP_HIGH)) & (d.period >= 3)
    d = d[~garbage].assign(pass_epa=lambda x: x.EPA.where(x["pass"] == 1),
                           rush_epa=lambda x: x.EPA.where(x.rush == 1))
    a = d.groupby(["game_id", "pos_team"]).agg(
        plays=("EPA", "size"), epa=("EPA", "mean"), sr=("success", "mean"),
        pass_epa=("pass_epa", "mean"), rush_epa=("rush_epa", "mean")).reset_index()
    home = d.groupby("game_id").home.first()
    a["is_home"] = a.pos_team.values == home.reindex(a.game_id).values
    return a


def load_pbp_games(seasons) -> pd.DataFrame:
    """Per-offense game stats, cached as a small file per season (the raw
    play-by-play is ~120 MB a season and is deleted once collapsed)."""
    cur = current_season()
    out = []
    for y in seasons:
        if y < PBP_FIRST:
            continue
        small = os.path.join(CACHE_DIR, f"pbp_games_{y}.parquet")
        stale = y >= cur and (not os.path.exists(small) or
                              time.time() - os.path.getmtime(small) > 12 * 3600)
        if stale or not os.path.exists(small):
            raw = _fetch(PBP_URL.format(season=y), f"pbp_{y}.parquet", 0)
            _pbp_season(raw).assign(season=y).to_parquet(small)
            os.remove(raw)
        out.append(pd.read_parquet(small))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
