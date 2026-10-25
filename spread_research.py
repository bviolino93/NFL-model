"""
Search for spread factors the closing line misprices.

Every candidate is built from information known before kickoff and put in
the home team's terms, so a positive value means "the home side should do
better than the line says". Each is scored by its slope on (margin - line)
in three separate eras:

    discovery   1999-2011
    validation  2012-2021
    holdout     2022-2026  (never used to choose anything)

A factor is only worth using if it points the same way, with real size, in
all three. Twenty-odd candidates means one or two will look good in any
single era by luck, which is why one era proves nothing.

    python spread_research.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_data import graded_games, load_games, load_team_games

ERAS = {"discovery": (1999, 2011), "validation": (2012, 2021),
        "holdout": (2022, 2026)}

# Standard-time UTC offsets; what matters is the difference.
TZ = {"ARI": -7, "ATL": -5, "BAL": -5, "BUF": -5, "CAR": -5, "CHI": -6,
      "CIN": -5, "CLE": -5, "DAL": -6, "DEN": -7, "DET": -5, "GB": -6,
      "HOU": -6, "IND": -5, "JAX": -5, "KC": -6, "LV": -8, "LAC": -8,
      "LA": -8, "MIA": -5, "MIN": -6, "NE": -5, "NO": -6, "NYG": -5,
      "NYJ": -5, "PHI": -5, "PIT": -5, "SF": -8, "SEA": -8, "TB": -5,
      "TEN": -6, "WAS": -5}


def team_rows(g: pd.DataFrame, tg: pd.DataFrame | None) -> pd.DataFrame:
    """One row per team per game, from that team's side."""
    rows = []
    for side, other, sign in (("home", "away", 1), ("away", "home", -1)):
        rows.append(pd.DataFrame({
            "game_id": g.game_id, "season": g.season, "week": g.week,
            "team": g[f"{side}_team"], "opp": g[f"{other}_team"],
            "margin": sign * g.margin, "line": sign * g.spread_line,
            "won": (sign * g.margin > 0).astype(float),
            "pf": g[f"{side}_score"], "pa": g[f"{other}_score"],
        }))
    t = pd.concat(rows, ignore_index=True)
    t["ats"] = t["margin"] - t["line"]
    t["one_score"] = (t["margin"].abs() <= 8).astype(float)
    t["close_net"] = t["one_score"] * np.sign(t["margin"])
    if tg is not None:
        e = tg.groupby(["game_id", "team"])["epa"].sum()
        off = [e.get((i, a), np.nan) for i, a in zip(t.game_id, t.team)]
        de = [e.get((i, b), np.nan) for i, b in zip(t.game_id, t.opp)]
        t["epa_margin"] = np.array(off) - np.array(de)
        t["luck"] = t["margin"] - t["epa_margin"]
    else:
        t["luck"] = np.nan
    return t.sort_values(["team", "season", "week"])


def prior_features(t: pd.DataFrame) -> pd.DataFrame:
    """Season-to-date numbers known before each game (shrunk toward 0)."""
    t = t.copy()
    grp = t.groupby(["team", "season"])
    n = grp.cumcount()
    for c in ("ats", "luck", "close_net", "margin"):
        cs = grp[c].cumsum() - t[c]
        t[f"{c}_std"] = cs / (n + 4)
    t["ats_prev"] = grp["ats"].shift(1).fillna(0)
    t["mov_prev"] = grp["margin"].shift(1).fillna(0)
    # Last season, for the first weeks when this season says little.
    last = (t.groupby(["team", "season"])
            .agg(ls_ats=("ats", "mean"), ls_luck=("luck", "mean"),
                 ls_close=("close_net", "sum"), ls_win=("won", "mean"),
                 ls_pf=("pf", "sum"), ls_pa=("pa", "sum"))
            .reset_index())
    pyth = last.ls_pf ** 2.37 / (last.ls_pf ** 2.37 + last.ls_pa ** 2.37)
    last["ls_pyth_luck"] = last.ls_win - pyth
    last["season"] += 1
    t = t.merge(last.drop(columns=["ls_win", "ls_pf", "ls_pa"]),
                on=["team", "season"], how="left")
    early = (t["week"] <= 4).astype(float)
    for c in ("ls_ats", "ls_luck", "ls_close", "ls_pyth_luck"):
        t[c] = t[c].fillna(0) * early
    return t


def build(first: int = 1999, last: int = 2026,
          use_epa: bool = True) -> pd.DataFrame:
    games = load_games()
    games = games[games.game_type == "REG"]
    g = graded_games(games)
    g = g[(g.season >= first) & (g.season <= last) & ~g.neutral]
    tg = load_team_games(range(first, last + 1)) if use_epa else None
    t = prior_features(team_rows(g, tg))
    cols = ["ats_std", "luck_std", "close_net_std", "margin_std", "ats_prev",
            "mov_prev", "ls_ats", "ls_luck", "ls_close", "ls_pyth_luck"]
    h = t.set_index(["game_id", "team"])[cols]
    for c in cols:
        hv = h[c].reindex(list(zip(g.game_id, g.home_team))).to_numpy()
        av = h[c].reindex(list(zip(g.game_id, g.away_team))).to_numpy()
        g[c] = hv - av

    g["s_res"] = g["margin"] - g["spread_line"]
    fav = np.sign(g["spread_line"]).replace(0, 1)
    outdoor = g["roof"].isin(["outdoors", "open"]).astype(float)
    wind = pd.to_numeric(g["wind"], errors="coerce").fillna(0) * outdoor
    hour = pd.to_numeric(g["gametime"].astype(str).str[:2], errors="coerce")
    prime = ((hour >= 19) | (g["weekday"] != "Sunday")).astype(float)
    early_kick = ((hour <= 13) & (g["weekday"] == "Sunday")).astype(float)
    tz_shift = (g.home_team.map(TZ) - g.away_team.map(TZ)).fillna(0)

    f = pd.DataFrame(index=g.index)
    f["home_dog"] = (g["spread_line"] < 0).astype(float)
    f["big_fav"] = fav * (g["spread_line"].abs() >= 7)
    f["huge_fav"] = fav * (g["spread_line"].abs() >= 10)
    f["dog_low_total"] = -fav * (g["total_line"] <= 41)
    f["div_dog"] = -fav * g["div_game"].fillna(0)
    f["wind_dog"] = -fav * (wind >= 15)
    f["prime_fav"] = fav * prime
    # West-coast team at a 1 PM ET start, and travel east generally
    f["west_early"] = early_kick * (g.away_team.map(TZ) <= -7)
    f["tz_east"] = tz_shift.clip(0, 3)
    f["rest_diff"] = (g.home_rest - g.away_rest).clip(-7, 7).fillna(0)
    f["late_fav"] = fav * (g["week"] >= 15)
    # Momentum and luck: positive means the HOME team looks better than
    # it is (or worse), with the sign to test whether the market overreacts
    f["ats_std"] = g["ats_std"]
    f["ats_prev"] = g["ats_prev"]
    f["mov_prev"] = g["mov_prev"]
    f["margin_std"] = g["margin_std"]
    f["luck_std"] = g["luck_std"]
    f["close_net_std"] = g["close_net_std"]
    f["ls_ats"] = g["ls_ats"]
    f["ls_luck"] = g["ls_luck"]
    f["ls_close"] = g["ls_close"]
    f["ls_pyth_luck"] = g["ls_pyth_luck"]
    out = pd.concat([g[["game_id", "season", "week", "home_team", "away_team",
                        "spread_line", "total_line", "margin", "s_res",
                        "home_spread_odds", "away_spread_odds"]], f], axis=1)
    return out


FEATS = ["home_dog", "big_fav", "huge_fav", "dog_low_total", "div_dog",
         "wind_dog", "prime_fav", "west_early", "tz_east", "rest_diff",
         "late_fav", "ats_std", "ats_prev", "mov_prev", "margin_std",
         "luck_std", "close_net_std", "ls_ats", "ls_luck", "ls_close",
         "ls_pyth_luck"]


def slope(x: np.ndarray, y: np.ndarray):
    x = x - x.mean()
    if (x ** 2).sum() == 0:
        return np.nan, np.nan
    b = (x * (y - y.mean())).sum() / (x ** 2).sum()
    resid = y - y.mean() - b * x
    se = np.sqrt((resid ** 2).sum() / (len(y) - 2) / (x ** 2).sum())
    return b, b / se


def screen(fr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in FEATS:
        row = {"feature": c}
        for era, (a, b) in ERAS.items():
            d = fr[(fr.season >= a) & (fr.season <= b)].dropna(subset=[c])
            sl, t = slope(d[c].to_numpy(float), d.s_res.to_numpy(float))
            row[f"{era}_b"] = sl
            row[f"{era}_t"] = t
        ts = [row[f"{e}_t"] for e in ERAS]
        row["same_sign"] = len({np.sign(x) for x in ts}) == 1
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    fr = build()
    pd.set_option("display.width", 200)
    print(len(fr), "games")
    print(screen(fr).round(3).to_string(index=False))
