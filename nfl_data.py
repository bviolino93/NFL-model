"""
nflverse data, read straight from the release files.

Why not nfl_data_py
-------------------
nfl_data_py pins an old pandas and does not install on current Python
(3.12+), and nflverse has deprecated it in favour of nflreadpy. Everything
this project needs is published as plain CSV/parquet on the nflverse-data
GitHub releases, so read those directly and cache them on disk. Completed
seasons never change; the current season is re-pulled when the cached copy
is older than `max_age_hours`.

What comes out
--------------
load_games()       one row per game, with the CLOSING spread, total and
                   moneylines (nflverse states spread_line from the home
                   side: +3 means home favoured by 3).
team_games()       one row per team per game: offensive EPA split into
                   dropbacks and designed runs, with and without garbage
                   time, plus the quarterback who took most of the dropbacks.

The team-game table is the unit the ratings are fitted on. A ridge on plays
and a weighted ridge on per-game play means have the same normal equations
when every play in a game shares its offense/defense/QB indicators, so
collapsing ~48k plays a season to ~570 rows loses nothing for a team model
and makes a weekly walk-forward over a decade run in seconds.
"""

from __future__ import annotations

import os
import time
import urllib.request

import pandas as pd

RELEASES = "https://github.com/nflverse/nflverse-data/releases/download"
GAMES_URL = f"{RELEASES}/schedules/games.csv"
PBP_URL = f"{RELEASES}/pbp/play_by_play_{{season}}.parquet"

CACHE_DIR = os.environ.get("NFL_CACHE_DIR", ".cache_nflverse")

# Garbage time: outside this win-probability band the football stops being
# representative (prevent defense, running clock, backups).
WP_LOW, WP_HIGH = 0.05, 0.95

PBP_COLS = ["game_id", "season", "week", "season_type", "game_date",
            "posteam", "defteam", "home_team", "away_team", "epa", "wp",
            "play_type", "pass", "rush", "qb_dropback", "success", "id",
            "two_point_attempt"]

# One code per franchise across seasons, so relocated teams keep their
# history (the ratings decay old games anyway).
TEAM_CANON = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA"}


def _canon(s: pd.Series) -> pd.Series:
    return s.astype(str).replace(TEAM_CANON)


def _fetch(url: str, path: str, max_age_hours: float | None) -> str:
    if os.path.exists(path):
        if max_age_hours is None:
            return path
        if (time.time() - os.path.getmtime(path)) / 3600.0 < max_age_hours:
            return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, path)
    return path


def _current_season() -> int:
    t = time.gmtime()
    # The NFL season is named for the year it starts; Jan-Feb games belong
    # to the previous one.
    return t.tm_year if t.tm_mon >= 3 else t.tm_year - 1


def load_games(cache_dir: str = CACHE_DIR, max_age_hours: float = 6.0,
               regular_only: bool = False) -> pd.DataFrame:
    path = _fetch(GAMES_URL, os.path.join(cache_dir, "games.csv"),
                  max_age_hours)
    g = pd.read_csv(path, low_memory=False)
    for c in ("home_team", "away_team"):
        g[c] = _canon(g[c])
    if regular_only:
        g = g[g["game_type"] == "REG"]
    g["gameday"] = pd.to_datetime(g["gameday"], errors="coerce")
    return g.reset_index(drop=True)


def load_pbp(seasons, cache_dir: str = CACHE_DIR) -> pd.DataFrame:
    cur = _current_season()
    frames = []
    for yr in seasons:
        path = os.path.join(cache_dir, f"pbp_{int(yr)}.parquet")
        age = 6.0 if int(yr) >= cur else None
        try:
            _fetch(PBP_URL.format(season=int(yr)), path, age)
        except Exception:
            if not os.path.exists(path):
                continue
        df = pd.read_parquet(path, columns=PBP_COLS)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=PBP_COLS)
    return pd.concat(frames, ignore_index=True)


def team_games(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (game, offense): play counts and EPA sums for dropbacks and
    designed runs, all plays and non-garbage plays, and the starting QB.

    Kneels, spikes, two-point tries and special teams carry no offensive
    signal and are dropped. Scrambles and sacks count as dropbacks, which is
    what they are.
    """
    d = pbp[pbp["play_type"].isin(["pass", "run"])
            & (pbp["two_point_attempt"].fillna(0) == 0)].copy()
    d = d.dropna(subset=["posteam", "defteam", "epa"])
    d = d[d["epa"].abs() <= 10.0]
    for c in ("posteam", "defteam", "home_team", "away_team"):
        d[c] = _canon(d[c])
    d["db"] = d["qb_dropback"].fillna(d["pass"]).fillna(0).astype(float)
    d["ru"] = 1.0 - d["db"]
    wp = pd.to_numeric(d["wp"], errors="coerce")
    d["live"] = (wp.between(WP_LOW, WP_HIGH) | wp.isna()).astype(float)
    d["succ"] = pd.to_numeric(d["success"], errors="coerce").fillna(0.0)

    e = d["epa"].astype(float)
    cols = {
        "n": 1.0, "epa": e, "succ": d["succ"],
        "n_db": d["db"], "epa_db": e * d["db"],
        "n_ru": d["ru"], "epa_ru": e * d["ru"],
    }
    for k, v in list(cols.items()):
        d[k] = v
        d[k + "_live"] = d[k] * d["live"]
    keep = [k for k in cols] + [k + "_live" for k in cols]
    keys = ["game_id", "season", "week", "season_type", "posteam", "defteam",
            "home_team", "away_team"]
    agg = d.groupby(keys, dropna=False)[keep].sum().reset_index()

    # Starting QB = most dropbacks for that offense in that game.
    qb = (d[d["db"] > 0].dropna(subset=["id"])
          .groupby(["game_id", "posteam", "id"]).size().rename("qb_db")
          .reset_index().sort_values("qb_db", ascending=False)
          .drop_duplicates(["game_id", "posteam"]))
    agg = agg.merge(qb.rename(columns={"id": "qb_id"}),
                    on=["game_id", "posteam"], how="left")
    agg["is_home"] = (agg["posteam"] == agg["home_team"]).astype(float)
    agg = agg.rename(columns={"posteam": "team", "defteam": "opp"})
    return agg


def load_team_games(seasons, cache_dir: str = CACHE_DIR) -> pd.DataFrame:
    """team_games() for many seasons, with completed seasons cached."""
    cur = _current_season()
    out = []
    for yr in seasons:
        path = os.path.join(cache_dir, f"team_games_{int(yr)}.parquet")
        fresh = os.path.exists(path) and (
            int(yr) < cur or
            (time.time() - os.path.getmtime(path)) / 3600.0 < 6.0)
        if fresh:
            out.append(pd.read_parquet(path))
            continue
        pbp = load_pbp([yr], cache_dir)
        if pbp.empty:
            continue
        tg = team_games(pbp)
        os.makedirs(cache_dir, exist_ok=True)
        tg.to_parquet(path, index=False)
        out.append(tg)
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)


def graded_games(games: pd.DataFrame) -> pd.DataFrame:
    """Completed games with a closing spread and total."""
    g = games.dropna(subset=["home_score", "away_score", "spread_line",
                             "total_line"]).copy()
    g["margin"] = g["home_score"] - g["away_score"]
    g["points"] = g["home_score"] + g["away_score"]
    g["neutral"] = (g["location"].astype(str).str.lower() == "neutral")
    return g.reset_index(drop=True)


if __name__ == "__main__":
    import sys
    yrs = [int(a) for a in sys.argv[1:]] or [_current_season()]
    tg = load_team_games(yrs)
    print(tg.groupby("season").size())
    print(tg.head())
