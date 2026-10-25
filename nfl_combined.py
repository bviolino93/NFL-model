"""
One model that looks at everything and says how likely each side is to win.

The market line is the starting point, not one feature among many: it is
the best single forecast there is, and treating it as just another input
would let a noisy signal argue with it on equal terms. So the model
predicts how far the RESULT lands from the line,

    margin - spread  ~  w . features          (spreads)
    points - total   ~  w . features          (totals)

and the win chance of any bet comes from the empirical result
distribution centred on (line + prediction), pushes included.

Features (all knowable before kickoff)
--------------------------------------
  model_gap    EPA power rating (nfl_power) minus the line
  qb change    either side starting a different QB from last game
  injuries     snap-weighted non-QB players out/doubtful/questionable
  rest         rest-day difference, byes
  division, late season (weeks 17-18)
  weather      wind and cold for outdoor games (totals)

Weights are fitted by ridge regression on PRIOR seasons only and refitted
every season, so the backtest is honest. A feature that has not predicted
results beyond the line gets a weight near zero and cannot inflate the
win chance. That is the point: the model is allowed to use everything, and
the data decides what everything is worth.

    python nfl_combined.py        # walk-forward backtest, records by confidence
"""

from __future__ import annotations

import argparse
import os
import re

import numpy as np
import pandas as pd

from backtest import calibrate, project
from nfl_data import CACHE_DIR, TEAM_CANON, fix_starters, graded_games, \
    load_games, load_rosters, load_team_games
from nfl_power import PowerConfig
from selector import build_outcome_models

SPREAD_FEATS = ["model_gap", "rest_diff", "h_bye", "a_bye", "h_qbchg",
                "a_qbchg", "div", "late", "inj_off", "inj_def"]
TOTAL_FEATS = ["model_gap_t", "wind", "cold", "short", "div", "late",
               "qbchg_any", "inj_off_sum", "inj_def_sum"]
# Ridge penalty in units of games: strong, because the prior is "the
# market is right" and every weight has to earn its way off zero.
ALPHA = 3000.0
# Observed wind in the training data is sharper than any forecast; the app
# measured live forecasts at about 40% of the observed effect.
WIND_FORECAST_DISCOUNT = 0.40


# ----------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------
def _nk(s):
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", str(s).lower())
    return re.sub(r"[^a-z]", "", s)


def injury_loads(seasons, cache_dir: str = CACHE_DIR,
                 include_ir: bool = False) -> pd.DataFrame:
    """
    Per (season, week, team): sum over listed non-QB players of
    P(miss) x recent snap share, split offense/defense. Snap share is the
    mean over the team's previous four games, so only players who have
    actually been playing count.

    `include_ir` also counts players on reserve lists (IR and the like),
    who are not on the injury report at all. They count fully, and fade
    out on their own as their recent snap share falls. The totals model
    was fitted without them, so it keeps the default.
    """
    from nfl_data import _fetch
    base = "https://github.com/nflverse/nflverse-data/releases/download"
    canon = {**TEAM_CANON, "LAR": "LA", "JAC": "JAX"}
    sn, inj = [], []
    for y in seasons:
        for kind, out in (("snap_counts", sn), ("injuries", inj)):
            p = os.path.join(cache_dir, f"{kind}_{y}.parquet")
            try:
                _fetch(f"{base}/{kind}/{kind}_{y}.parquet", p,
                       6.0 if y >= pd.Timestamp.now().year else None)
                out.append(pd.read_parquet(p))
            except Exception:
                continue
    if not sn or not inj:
        return pd.DataFrame(columns=["season", "week", "team", "off", "def"])
    sn = pd.concat(sn, ignore_index=True)
    inj = pd.concat(inj, ignore_index=True)
    sn["team"] = sn["team"].astype(str).replace(canon)
    sn["k"] = sn["player"].map(_nk)
    op, dp = sn["offense_pct"].fillna(0), sn["defense_pct"].fillna(0)
    sn["pct"] = np.maximum(op, dp)
    sn["side"] = np.where(op >= dp, "off", "def")
    inj["team"] = inj["team"].astype(str).replace(canon)
    inj["k"] = inj["full_name"].map(_nk)
    inj["p"] = inj["report_status"].map(
        {"Out": 1.0, "Doubtful": 0.9, "Questionable": 0.25}).fillna(0.0)
    inj = inj[(inj["p"] > 0) & (inj["position"] != "QB")]
    if include_ir:
        keep = [inj[["season", "week", "team", "k", "p"]]]
        for y in seasons:
            try:
                r = load_rosters(y, cache_dir)
            except Exception:
                continue
            r = r[(r["status"] == "RES") & (r["position"] != "QB")]
            if r.empty:
                continue
            # The latest roster also stands for the coming week.
            r = pd.concat([r, r[r["week"] == r["week"].max()]
                           .assign(week=r["week"].max() + 1)])
            keep.append(pd.DataFrame({
                "season": r["season"], "week": r["week"], "team": r["team"],
                "k": r["full_name"].map(_nk), "p": 1.0}))
        inj = (pd.concat(keep, ignore_index=True).sort_values("p")
               .drop_duplicates(["season", "week", "team", "k"], keep="last"))

    rows = []
    for (s, t), grp in sn.groupby(["season", "team"]):
        weeks = sorted(grp["week"].unique())
        piv = (grp.pivot_table(index="week", columns="k", values="pct",
                               aggfunc="max").reindex(weeks).fillna(0.0))
        side = grp.groupby("k")["side"].agg(lambda x: x.mode()[0])
        imp = piv.rolling(4, min_periods=1).mean().shift(1)
        # Also cover the week after the last played one (upcoming games).
        nxt = imp.reindex(weeks + [weeks[-1] + 1])
        nxt.iloc[-1] = piv.tail(4).mean()
        for w, r in nxt.iloc[1:].iterrows():
            r = r[r > 0.25]
            for k, v in r.items():
                rows.append((s, w, t, k, v, side[k]))
    imp = pd.DataFrame(rows, columns=["season", "week", "team", "k", "imp",
                                      "side"])
    m = inj.merge(imp, on=["season", "week", "team", "k"])
    m["load"] = m["p"] * m["imp"]
    out = (m.pivot_table(index=["season", "week", "team"], columns="side",
                         values="load", aggfunc="sum").fillna(0.0)
           .reset_index())
    for c in ("off", "def"):
        if c not in out:
            out[c] = 0.0
    return out


def schedule_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values(["season", "week"]).copy()
    long = pd.concat([
        g[["game_id", "season", "week", "home_team", "home_qb_id"]]
        .rename(columns={"home_team": "t", "home_qb_id": "qb"}),
        g[["game_id", "season", "week", "away_team", "away_qb_id"]]
        .rename(columns={"away_team": "t", "away_qb_id": "qb"}),
    ]).sort_values(["season", "week"])
    long["prev"] = long.groupby("t")["qb"].shift(1)
    long["chg"] = ((long["qb"] != long["prev"]) & long["prev"].notna()
                   & long["qb"].notna()).astype(float)
    ch = long.set_index(["game_id", "t"])["chg"]
    g["h_qbchg"] = [ch.get((i, t), 0.0) for i, t in zip(g.game_id, g.home_team)]
    g["a_qbchg"] = [ch.get((i, t), 0.0) for i, t in zip(g.game_id, g.away_team)]
    g["qbchg_any"] = g["h_qbchg"] + g["a_qbchg"]
    g["rest_diff"] = (g["home_rest"] - g["away_rest"]).clip(-7, 7).fillna(0)
    g["h_bye"] = (g["home_rest"] >= 13).astype(float)
    g["a_bye"] = (g["away_rest"] >= 13).astype(float)
    g["short"] = ((g["home_rest"] <= 5) & (g["away_rest"] <= 5)).astype(float)
    out = g["roof"].isin(["outdoors", "open"]).astype(float)
    g["wind"] = pd.to_numeric(g["wind"], errors="coerce").fillna(0) * out
    g["cold"] = ((pd.to_numeric(g["temp"], errors="coerce") <= 32)
                 & (out == 1)).astype(float)
    g["div"] = g["div_game"].fillna(0).astype(float)
    g["late"] = ((g["week"] >= 17) & (g["game_type"] == "REG")).astype(float)
    return g


def build_frame(first: int = 2013, last: int = 2026,
                cfg: PowerConfig | None = None) -> pd.DataFrame:
    """Every graded game with the line, the result and every feature."""
    games = graded_games(load_games())
    tg = load_team_games(range(first - 1, last + 1))
    fr = calibrate(project(tg, games, range(first, last + 1),
                           cfg or PowerConfig()))
    g = schedule_features(games)
    fr = fr.merge(g[["game_id", "h_qbchg", "a_qbchg", "qbchg_any",
                     "rest_diff", "h_bye", "a_bye", "short", "wind", "cold",
                     "div", "late"]], on="game_id", how="left")
    L = injury_loads(range(first, last + 1))
    for side, pre in (("home", "h"), ("away", "a")):
        fr = fr.merge(L.rename(columns={"team": side, "off": f"{pre}_off",
                                        "def": f"{pre}_def"}),
                      on=["season", "week", side], how="left")
    for c in ("h_off", "h_def", "a_off", "a_def"):
        fr[c] = fr[c].fillna(0.0)
    fr["inj_off"] = fr["h_off"] - fr["a_off"]
    fr["inj_def"] = fr["h_def"] - fr["a_def"]
    fr["inj_off_sum"] = fr["h_off"] + fr["a_off"]
    fr["inj_def_sum"] = fr["h_def"] + fr["a_def"]
    fr["model_gap"] = fr["model_margin"] - fr["spread"]
    fr["model_gap_t"] = fr["model_total"] - fr["total"]
    fr["s_res"] = fr["margin"] - fr["spread"]
    fr["t_res"] = fr["points"] - fr["total"]
    return fr


# ----------------------------------------------------------------------
# Fit / predict
# ----------------------------------------------------------------------
def fit_weights(tr: pd.DataFrame, feats, target, alpha=ALPHA):
    X = tr[feats].to_numpy(float)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    y = tr[target].to_numpy(float)
    b0 = y.mean()
    w = np.linalg.solve(Z.T @ Z + alpha * np.eye(len(feats)),
                        Z.T @ (y - b0))
    # Weights in raw units; the intercept is NOT kept: a constant lean to
    # one side (e.g. unders) is the kind of thing that is real in one era
    # and gone in the next, and the books adjust to it.
    return pd.Series(w / sd, index=feats)


def walk_forward(fr: pd.DataFrame, first_test: int = 2016):
    fr = fr.copy()
    fr["pred_s"] = np.nan
    fr["pred_t"] = np.nan
    weights = {}
    for s in sorted(fr.loc[fr["season"] >= first_test, "season"].unique()):
        tr = fr[fr["season"] < s]
        ws = fit_weights(tr, SPREAD_FEATS, "s_res")
        wt = fit_weights(tr, TOTAL_FEATS, "t_res")
        m = fr["season"] == s
        fr.loc[m, "pred_s"] = fr.loc[m, SPREAD_FEATS].to_numpy(float) @ ws.values
        fr.loc[m, "pred_t"] = fr.loc[m, TOTAL_FEATS].to_numpy(float) @ wt.values
        weights[s] = (ws, wt)
    return fr.dropna(subset=["pred_s"]), weights


def win_probs(fr: pd.DataFrame, mm, tm) -> pd.DataFrame:
    """P(home covers), P(away covers), P(over), P(under) at the close."""
    fr = fr.copy()
    ph, pa, po, pu = [], [], [], []
    for r in fr.itertuples():
        w, p = mm.probs(r.spread + r.pred_s, r.spread)
        ph.append(w)
        pa.append(1 - w - p)
        w, p = tm.probs(r.total + r.pred_t, r.total)
        po.append(w)
        pu.append(1 - w - p)
    fr["p_home"], fr["p_away"], fr["p_over"], fr["p_under"] = ph, pa, po, pu
    return fr


def records(fr: pd.DataFrame, cuts=(0.50, 0.52, 0.53, 0.54, 0.55)):
    """
    Record and ROI at -110 when betting the side the model favours, by the
    model's no-push win chance.
    """
    out = []
    for mkt, a, b, res, sign in (
            ("spread", "p_home", "p_away", "s_res", 1),
            ("total", "p_over", "p_under", "t_res", 1)):
        pa, pb = fr[a], fr[b]
        side = np.where(pa >= pb, 1.0, -1.0)
        conf = np.maximum(pa, pb) / (pa + pb)
        r = fr[res].to_numpy(float) * sign
        for c in cuts:
            m = (conf >= c) & (r != 0)
            win = int(((side[m] * r[m]) > 0).sum())
            n = int(m.sum())
            roi = (win * (100 / 110) - (n - win)) / n if n else float("nan")
            out.append({"market": mkt, "min_chance": c, "bets": n,
                        "won": win, "win%": win / n if n else float("nan"),
                        "roi": roi})
    return pd.DataFrame(out)


# ----------------------------------------------------------------------
# Live use
# ----------------------------------------------------------------------
MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "combined_model.json")


def export_model(fr: pd.DataFrame, path: str = MODEL_FILE) -> dict:
    """
    Weights fitted on every graded game, plus the points calibration for
    the power rating, so the app needs only the last few seasons of plays
    to project a week.
    """
    import json
    ws = fit_weights(fr, SPREAD_FEATS, "s_res")
    wt = fit_weights(fr, TOTAL_FEATS, "t_res")
    e = fr["model_edge"].to_numpy(float)
    k = float(e @ (fr["margin"] - fr["hfa"]).to_numpy(float) / (e @ e))
    b, a0 = np.polyfit(fr["epa_sum"], fr["points"], 1)
    out = {"spread": ws.round(5).to_dict(), "total": wt.round(5).to_dict(),
           "k_margin": k, "total_a": float(a0), "total_b": float(b),
           "through_season": int(fr["season"].max()),
           "n_games": int(len(fr))}
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    return out


def predict_week(season: int, week: int, model: dict | None = None,
                 cfg: PowerConfig | None = None) -> pd.DataFrame:
    """
    Predictions for the unplayed games of (season, week): how far the
    result should land from the line, with each feature's contribution.

    Wind is left at zero here: the app prices the live FORECAST through its
    own measured wind rule, and counting it twice would double the bet.
    """
    import json
    from backtest import home_field_before
    from nfl_power import fit_power
    if model is None:
        with open(MODEL_FILE) as fh:
            model = json.load(fh)
    cfg = cfg or PowerConfig()
    games = fix_starters(load_games())
    up = games[(games["season"] == season) & (games["week"] == week)
               & games["home_score"].isna()].copy()
    if up.empty:
        return pd.DataFrame()
    tg = load_team_games(range(season - 2, season + 1))
    done = graded_games(games)
    r = fit_power(tg, season, week, home_field_before(done, season, week), cfg)
    feats = schedule_features(games)
    feats = feats[feats["game_id"].isin(up["game_id"])].set_index("game_id")
    L = injury_loads([season])
    L = L[(L["season"] == season) & (L["week"] == week)].set_index("team")
    rows = []
    for g in up.itertuples():
        if g.home_team not in r.teams or g.away_team not in r.teams:
            continue
        hq = g.home_qb_id if isinstance(g.home_qb_id, str) else None
        aq = g.away_qb_id if isinstance(g.away_qb_id, str) else None
        neutral = str(g.location).lower() == "neutral"
        hfa = 0.0 if neutral else r.home_field
        edge = r.projected_margin(g.home_team, g.away_team, neutral=True,
                                  home_qb=hq, away_qb=aq)
        mm = hfa + model["k_margin"] * edge
        mt = model["total_a"] + model["total_b"] * r.projected_epa_sum(
            g.home_team, g.away_team, hq, aq)
        f = feats.loc[g.game_id]
        lh = L.loc[g.home_team] if g.home_team in L.index else None
        la = L.loc[g.away_team] if g.away_team in L.index else None
        ho = 0.0 if lh is None else float(lh["off"])
        hd = 0.0 if lh is None else float(lh["def"])
        ao = 0.0 if la is None else float(la["off"])
        ad = 0.0 if la is None else float(la["def"])
        x = {"model_gap": mm - g.spread_line, "model_gap_t": mt - g.total_line,
             "rest_diff": f["rest_diff"], "h_bye": f["h_bye"],
             "a_bye": f["a_bye"], "h_qbchg": f["h_qbchg"],
             "a_qbchg": f["a_qbchg"], "qbchg_any": f["qbchg_any"],
             "div": f["div"], "late": f["late"], "short": f["short"],
             "wind": 0.0, "cold": 0.0,
             "inj_off": ho - ao, "inj_def": hd - ad,
             "inj_off_sum": ho + ao, "inj_def_sum": hd + ad}
        cs = {k: float(v) * float(x[k]) for k, v in model["spread"].items()}
        ct = {k: float(v) * float(x[k]) for k, v in model["total"].items()}
        rows.append({"game_id": g.game_id, "home": g.home_team,
                     "away": g.away_team, "spread_line": g.spread_line,
                     "total_line": g.total_line, "model_margin": mm,
                     "model_total": mt, "adj_spread": sum(cs.values()),
                     "adj_total": sum(ct.values()), "x": x,
                     "why_spread": cs, "why_total": ct})
    return pd.DataFrame(rows)


LABELS = {"model_gap": "team ratings", "model_gap_t": "team ratings",
          "rest_diff": "rest", "h_bye": "home off a bye",
          "a_bye": "away off a bye", "h_qbchg": "home QB change",
          "a_qbchg": "away QB change", "qbchg_any": "QB change",
          "div": "division game", "late": "late season", "short": "short week",
          "wind": "wind", "cold": "cold", "inj_off": "offensive injuries",
          "inj_def": "defensive injuries", "inj_off_sum": "offensive injuries",
          "inj_def_sum": "defensive injuries"}


def reasons(contrib: dict, n: int = 3, min_pts: float = 0.1) -> str:
    """Top contributions in words, e.g. 'QB change -0.4, injuries +0.3'."""
    items = sorted(contrib.items(), key=lambda kv: -abs(kv[1]))
    return ", ".join(f"{LABELS.get(k, k)} {v:+.1f}" for k, v in items[:n]
                     if abs(v) >= min_pts)


def _fact(k, x, home, away):
    """What is true about this game, in words, for feature k."""
    v = x.get(k, 0.0)
    side = home if v > 0 else away
    return {
        "model_gap": f"our power ratings rate {side} better than the line does",
        "model_gap_t": ("our ratings project more points than the total"
                        if v > 0 else
                        "our ratings project fewer points than the total"),
        "rest_diff": f"{side} has the rest advantage",
        "h_bye": f"{home} is coming off a bye",
        "a_bye": f"{away} is coming off a bye",
        "h_qbchg": f"{home} is starting a different QB than last game",
        "a_qbchg": f"{away} is starting a different QB than last game",
        "qbchg_any": "there's a quarterback change in this game",
        "div": "it's a division game",
        "late": "it's a late-season game",
        "short": "both teams are on a short week",
        "inj_off": f"{side}'s offense is missing more regulars",
        "inj_def": f"{side}'s defense is missing more regulars",
        "inj_off_sum": "both offenses are missing regulars",
        "inj_def_sum": "both defenses are missing regulars",
        "wind": "it's windy", "cold": "it's freezing",
    }.get(k, LABELS.get(k, k))


def explain(contrib: dict, x: dict, home: str, away: str, market: str,
            n: int = 3, min_pts: float = 0.1, toward: int = 0) -> str:
    """
    The biggest factors as sentences, e.g. "MIA is starting a different QB
    than last game, which has historically favored NYJ (+0.3 pts)."

    toward=+1 keeps only factors pushing the home margin / total UP, -1
    only those pushing it down, so a bet's reasons all point its way.
    """
    out = []
    items = [(k, v) for k, v in contrib.items()
             if abs(v) >= min_pts and (toward == 0 or v * toward > 0)]
    for k, v in sorted(items, key=lambda kv: -abs(kv[1]))[:n]:
        if market == "totals":
            helps = "the over" if v > 0 else "the under"
        else:
            helps = home if v > 0 else away
        f = _fact(k, x, home, away)
        out.append(f"{f[0].upper() + f[1:]}, which has historically favored "
                   f"{helps} ({abs(v):.1f} pts).")
    return " ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=int, default=2022)
    ap.add_argument("--export", action="store_true",
                    help=f"also fit on everything and write {MODEL_FILE}")
    ap.add_argument("--week", type=int, default=0,
                    help="print predictions for this week's unplayed games")
    a = ap.parse_args()
    if a.week:
        from nfl_data import _current_season
        p = predict_week(_current_season(), a.week)
        for r in p.itertuples():
            print(f"{r.away} @ {r.home}: spread {r.adj_spread:+.2f}: "
                  f"{explain(r.why_spread, r.x, r.home, r.away, 'spreads')}"
                  f"\n    total {r.adj_total:+.2f}: "
                  f"{explain(r.why_total, r.x, r.home, r.away, 'totals')}")
        return
    fr = build_frame()
    if a.export:
        print(export_model(fr))
    fr, weights = walk_forward(fr)
    mm, tm = build_outcome_models(load_games())
    fr = win_probs(fr, mm, tm)
    pd.set_option("display.width", 200)
    last = max(weights)
    print("Weights fitted through", last - 1, "(points per unit):")
    print(pd.concat(weights[last], axis=1, keys=["spread", "total"])
          .round(3).to_string())
    for lab, d in (("2016-%d" % (a.split - 1), fr[fr["season"] < a.split]),
                   ("%d-now (held out)" % a.split,
                    fr[fr["season"] >= a.split])):
        print(f"\n== {lab}: {len(d)} games")
        print(records(d).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
