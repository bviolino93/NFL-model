"""
Player yardage projections for props: receiving yards, receptions and
rushing yards.

Why a separate model
--------------------
Game lines are the sharpest market there is, and the backtests in this repo
say public ratings add nothing to the close. Props are set by the same books
with far less money behind each line, and their inputs are mostly public:
how much of the offense a player gets, how efficient he is with it, how
much offense his team will run, and how good the defense is. That is
exactly what this models.

    projection = share x team_volume x efficiency, then adjusted
                 for opponent and game script by a walk-forward regression

  share       target share / carry share, an exponential average over the
              player's own previous games (carrying across seasons)
  efficiency  yards per target / carry, shrunk toward the position mean so
              ten great targets do not make a star
  volume      the team's recent targets / carries per game
  context     the opponent's yards allowed per target / carry, and the
              market's implied team total and spread (both known pregame)

Every feature for a game uses only earlier games, and the final regression
for season S is fitted on seasons before S. Scored against two naive
baselines (season-to-date mean and last-4 mean); there are no historical
prop LINES in public data, so this measures projection accuracy, not edge.

    python nfl_props.py              # fit and score 2016-now, test >= 2022
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from nfl_data import CACHE_DIR, TEAM_CANON, _fetch, load_games

STATS_URL = ("https://github.com/nflverse/nflverse-data/releases/download/"
             "stats_player/stats_player_week_{season}.parquet")
SKILL = {"WR", "TE", "RB", "FB"}
COLS = ["player_id", "player_display_name", "position", "season", "week",
        "season_type", "game_id", "team", "opponent_team", "targets",
        "receptions", "receiving_yards", "carries", "rushing_yards"]

# Set by judgment, not swept; the walk-forward regression in project()
# absorbs most mis-scaling. Worth sweeping on 2016-2021 MAE.
SHARE_HALFLIFE = 4.0      # games
EFF_HALFLIFE = 12.0       # games; efficiency is noisier than volume
TEAM_HALFLIFE = 6.0       # games
OPP_HALFLIFE = 8.0        # games
EFF_PRIOR_TARGETS = 40.0  # shrinkage weight toward the position mean
EFF_PRIOR_CARRIES = 60.0


def load_player_weeks(seasons, cache_dir: str = CACHE_DIR) -> pd.DataFrame:
    out = []
    for yr in seasons:
        path = os.path.join(cache_dir, f"stats_player_week_{int(yr)}.parquet")
        try:
            _fetch(STATS_URL.format(season=int(yr)), path,
                   6.0 if yr >= pd.Timestamp.now().year else None)
        except Exception:
            if not os.path.exists(path):
                continue
        out.append(pd.read_parquet(path, columns=COLS))
    d = pd.concat(out, ignore_index=True)
    for c in ("team", "opponent_team"):
        d[c] = d[c].astype(str).replace(TEAM_CANON)
    for c in ("targets", "receptions", "receiving_yards", "carries",
              "rushing_yards"):
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
    return d


def _ewm_prior(df, by, col, halflife):
    """Exponential mean of `col` over each group's PREVIOUS rows."""
    return (df.groupby(by, sort=False)[col]
              .transform(lambda s: s.shift(1).ewm(halflife=halflife,
                                                  min_periods=1).mean()))


def _ewm_sum_prior(df, by, col, halflife):
    """Exponentially decayed SUM over previous rows (for shrinkage counts)."""
    a = 0.5 ** (1.0 / halflife)

    def f(s):
        out, acc = np.empty(len(s)), 0.0
        for i, v in enumerate(s.to_numpy(float)):
            out[i] = acc
            acc = acc * a + v
        return pd.Series(out, index=s.index)
    return df.groupby(by, sort=False)[col].transform(f)


def add_upcoming(pw: pd.DataFrame, games: pd.DataFrame, season: int,
                 week: int) -> pd.DataFrame:
    """
    Append a zero-stat row for every skill player who appeared in either of
    his team's last two games, for each unplayed game in (season, week).
    Features only look backward, so these rows get projections like any
    other; `upcoming` marks them so nothing fits on their zeros.
    """
    up = games[(games["season"] == season) & (games["week"] == week)
               & games["home_score"].isna()]
    rows = []
    for g in up.itertuples():
        for team, opp in ((g.home_team, g.away_team),
                          (g.away_team, g.home_team)):
            t = pw[pw["team"] == team]
            last = t.drop_duplicates("game_id", keep="last") \
                .sort_values(["season", "week"]).tail(2)["game_id"]
            pl = t[t["game_id"].isin(last)].drop_duplicates("player_id",
                                                             keep="last")
            for r in pl.itertuples():
                rows.append({"player_id": r.player_id,
                             "player_display_name": r.player_display_name,
                             "position": r.position, "season": season,
                             "week": week, "season_type": "REG",
                             "game_id": g.game_id, "team": team,
                             "opponent_team": opp})
    if not rows:
        return pw.assign(upcoming=False)
    new = pd.DataFrame(rows)
    for c in ("targets", "receptions", "receiving_yards", "carries",
              "rushing_yards"):
        new[c] = 0.0
    return pd.concat([pw.assign(upcoming=False), new.assign(upcoming=True)],
                     ignore_index=True)


def build_features(pw: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    if "upcoming" not in pw:
        pw = pw.assign(upcoming=False)
    d = pw.sort_values(["season", "week"]).reset_index(drop=True)
    d["order"] = d["season"] * 100 + d["week"]

    # Team volume per game, and shares.
    team = (d.groupby(["game_id", "season", "week", "team", "opponent_team"])
              [["targets", "carries", "receiving_yards", "rushing_yards"]]
              .sum().reset_index().sort_values(["season", "week"]))
    team = team.rename(columns={"targets": "tm_tgt", "carries": "tm_car",
                                "receiving_yards": "tm_ryd",
                                "rushing_yards": "tm_rush"})
    team["tm_tgt_ewm"] = _ewm_prior(team, "team", "tm_tgt", TEAM_HALFLIFE)
    team["tm_car_ewm"] = _ewm_prior(team, "team", "tm_car", TEAM_HALFLIFE)
    # Defence: yards allowed per target / carry, from the opponent's view.
    team["ypt_allowed"] = team["tm_ryd"] / team["tm_tgt"].replace(0, np.nan)
    team["ypc_allowed"] = team["tm_rush"] / team["tm_car"].replace(0, np.nan)
    team = team.sort_values(["season", "week"])
    team["opp_ypt"] = _ewm_prior(team, "opponent_team", "ypt_allowed",
                                 OPP_HALFLIFE)
    team["opp_ypc"] = _ewm_prior(team, "opponent_team", "ypc_allowed",
                                 OPP_HALFLIFE)
    # Relative to the league over the trailing season, so the factor is ~1.
    lg = team.groupby("season")[["ypt_allowed", "ypc_allowed"]].mean()
    team = team.join(lg.shift(1).rename(columns=lambda c: "lg_" + c),
                     on="season")
    team["opp_ypt_rel"] = team["opp_ypt"] / team["lg_ypt_allowed"]
    team["opp_ypc_rel"] = team["opp_ypc"] / team["lg_ypc_allowed"]
    d = d.merge(team[["game_id", "team", "tm_tgt", "tm_car", "tm_tgt_ewm",
                      "tm_car_ewm", "opp_ypt_rel", "opp_ypc_rel"]],
                on=["game_id", "team"], how="left")

    d["tshare"] = d["targets"] / d["tm_tgt"].replace(0, np.nan)
    d["cshare"] = d["carries"] / d["tm_car"].replace(0, np.nan)
    d = d.sort_values(["season", "week"]).reset_index(drop=True)
    pid = "player_id"
    d["tshare_ewm"] = _ewm_prior(d, pid, "tshare", SHARE_HALFLIFE)
    d["cshare_ewm"] = _ewm_prior(d, pid, "cshare", SHARE_HALFLIFE)
    d["n_prior"] = d.groupby(pid).cumcount()

    # Shrunk efficiency.
    pos_ypt = (d.groupby("position")["receiving_yards"].sum()
               / d.groupby("position")["targets"].sum()).fillna(7.0)
    pos_ypc = (d.groupby("position")["rushing_yards"].sum()
               / d.groupby("position")["carries"].sum()).fillna(4.2)
    pos_cr = (d.groupby("position")["receptions"].sum()
              / d.groupby("position")["targets"].sum()).fillna(0.65)
    for num, den, prior, pos_mean, out in (
            ("receiving_yards", "targets", EFF_PRIOR_TARGETS, pos_ypt, "ypt"),
            ("receptions", "targets", EFF_PRIOR_TARGETS, pos_cr, "catch"),
            ("rushing_yards", "carries", EFF_PRIOR_CARRIES, pos_ypc, "ypc")):
        sn = _ewm_sum_prior(d, pid, num, EFF_HALFLIFE)
        sd = _ewm_sum_prior(d, pid, den, EFF_HALFLIFE)
        m = d["position"].map(pos_mean).fillna(pos_mean.mean())
        d[out] = (sn + prior * m) / (sd + prior)

    # Market context: implied team points and spread from the team's side.
    g = games[["game_id", "home_team", "away_team", "spread_line",
               "total_line"]]
    d = d.merge(g, on="game_id", how="left")
    home = d["team"] == d["home_team"]
    d["team_spread"] = np.where(home, d["spread_line"], -d["spread_line"])
    d["implied"] = d["total_line"] / 2 + d["team_spread"] / 2

    # Base projections.
    d["base_tgt"] = d["tshare_ewm"] * d["tm_tgt_ewm"]
    d["base_car"] = d["cshare_ewm"] * d["tm_car_ewm"]
    d["base_recyd"] = d["base_tgt"] * d["ypt"]
    d["base_rec"] = d["base_tgt"] * d["catch"]
    d["base_rushyd"] = d["base_car"] * d["ypc"]

    # Naive baselines.
    for col in ("receiving_yards", "receptions", "rushing_yards"):
        d[f"naive_std_{col}"] = (d.groupby([pid, "season"])[col]
                                 .transform(lambda s: s.shift(1).expanding()
                                            .mean()))
        d[f"naive_l4_{col}"] = (d.groupby(pid)[col]
                                .transform(lambda s: s.shift(1).rolling(
                                    4, min_periods=1).mean()))
    return d


TARGETS = {
    "receiving_yards": ("base_recyd", "opp_ypt_rel"),
    "receptions": ("base_rec", "opp_ypt_rel"),
    "rushing_yards": ("base_rushyd", "opp_ypc_rel"),
}


def _design(d, base, opp):
    b = d[base].to_numpy(float)
    o = d[opp].fillna(1.0).to_numpy(float) - 1.0
    imp = (d["implied"].fillna(22.0).to_numpy(float) - 22.0) / 7.0
    sp = d["team_spread"].fillna(0.0).to_numpy(float) / 7.0
    return np.column_stack([np.ones(len(d)), b, b * o, b * imp, b * sp])


def project(d: pd.DataFrame, first_test: int) -> pd.DataFrame:
    """
    Walk-forward: for each season S >= first_test, fit the combination on
    seasons before S, predict S. Earlier seasons are used only to fit.
    """
    d = d.copy()
    for tgt, (base, opp) in TARGETS.items():
        d[f"proj_{tgt}"] = np.nan
        ok = d[base].notna() & (d["n_prior"] >= 3) & ~d["upcoming"]
        for s in sorted(d.loc[d["season"] >= first_test, "season"].unique()):
            tr = d[ok & (d["season"] < s) & (d["season"] >= s - 6)]
            te = d[base].notna() & (d["n_prior"] >= 3) & (d["season"] == s)
            beta, *_ = np.linalg.lstsq(_design(tr, base, opp),
                                       tr[tgt].to_numpy(float), rcond=None)
            d.loc[te, f"proj_{tgt}"] = np.clip(
                _design(d[te], base, opp) @ beta, 0, None)
    return d


def prob_over(d: pd.DataFrame, tgt: str, proj, line, season: int) -> float:
    """
    P(actual > line) for a projection, from how actuals were spread around
    similar projections in the three seasons before `season`. Yardage is
    skewed (a median well below the mean), so this matters: a 50-yard mean
    projection is not a 50-50 shot at over 49.5.
    """
    p = f"proj_{tgt}"
    h = d[d[p].notna() & ~d["upcoming"] & (d["season"] < season)
          & (d["season"] >= season - 3)]
    near = h[(h[p] - proj).abs() <= max(0.15 * proj, 1.0)]
    if len(near) < 50:
        return float("nan")
    # Scale each historical outcome to this projection, then count overs.
    ratio = near[tgt] / near[p].clip(lower=0.1)
    return float((ratio * proj > line).mean())


def upcoming_table(d: pd.DataFrame) -> pd.DataFrame:
    """Projections, and the median outcome, for the upcoming rows."""
    u = d[d["upcoming"]].copy()
    if u.empty:
        return u
    season = int(u["season"].iloc[0])
    keep = ["player_display_name", "position", "team", "opponent_team"]
    for tgt in TARGETS:
        p = f"proj_{tgt}"
        h = d[d[p].notna() & ~d["upcoming"] & (d["season"] < season)
              & (d["season"] >= season - 3)]
        # Median outcome as a function of the projection: skew is much
        # worse for a 10-yard projection than for an 80-yard one.
        h = h[h[p] > 0.5]
        q = pd.qcut(h[p], 20, duplicates="drop")
        grp = h.groupby(q, observed=True)
        x = grp[p].median().to_numpy(float)
        y = grp[tgt].median().to_numpy(float) / x
        u[f"median_{tgt}"] = u[p] * np.interp(u[p], x, y)
        keep += [p, f"median_{tgt}"]
    u = u[keep].rename(columns=lambda c: c.replace("proj_", "mean_"))
    return u.sort_values("mean_receiving_yards", ascending=False)


def score(d: pd.DataFrame, min_proj: dict | None = None) -> pd.DataFrame:
    """
    MAE of the model against the naive baselines, on the player-games a
    book would actually hang a line on (projection above a floor).
    """
    min_proj = min_proj or {"receiving_yards": 25.0, "receptions": 2.5,
                            "rushing_yards": 25.0}
    rows = []
    for tgt in TARGETS:
        p = f"proj_{tgt}"
        m = d[p].notna() & (d[p] >= min_proj[tgt]) & \
            d[f"naive_l4_{tgt}"].notna() & d[f"naive_std_{tgt}"].notna()
        x = d[m & ~d["upcoming"]]
        y = x[tgt]
        rows.append({
            "market": tgt, "n": int(m.sum()),
            "mae_model": float((x[p] - y).abs().mean()),
            "mae_season_avg": float((x[f"naive_std_{tgt}"] - y).abs().mean()),
            "mae_last4": float((x[f"naive_l4_{tgt}"] - y).abs().mean()),
            "corr_model": float(np.corrcoef(x[p], y)[0, 1]),
            "mean_proj": float(x[p].mean()), "mean_actual": float(y.mean()),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=2013)
    ap.add_argument("--last", type=int, default=2026)
    ap.add_argument("--test", type=int, default=2016)
    ap.add_argument("--split", type=int, default=2022)
    ap.add_argument("--out", default="")
    ap.add_argument("--week", type=int, default=0,
                    help="project this week's unplayed games of --last")
    a = ap.parse_args()
    all_games = load_games()
    pw = load_player_weeks(range(a.first, a.last + 1))
    pw = pw[pw["position"].isin(SKILL)]
    if a.week:
        pw = add_upcoming(pw, all_games, a.last, a.week)
    d = project(build_features(pw, all_games), a.test)
    if a.week:
        t = upcoming_table(d)
        pd.set_option("display.width", 200)
        print(t.round(1).to_string(index=False))
        if a.out:
            t.to_csv(a.out, index=False)
        return
    if a.out:
        d.to_parquet(a.out, index=False)
    pd.set_option("display.width", 200)
    print(f"== {a.test}-{a.split - 1}")
    print(score(d[d["season"].between(a.test, a.split - 1)]).round(3)
          .to_string(index=False))
    print(f"== {a.split}-{a.last} (held out)")
    print(score(d[d["season"] >= a.split]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
