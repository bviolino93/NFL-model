"""
Sunday Edge — NFL

Slate builder, bet freezer, and tracker for NFL spreads and totals.

Built with the measurement fixes Saturday Edge needed retrofitted:
separate official/watch ledgers, real closing-line capture, continuous
result margins, and a version string that tracks the selection math.

The model is an opponent-adjusted ridge power rating, walk-forward.
Backtested 2007-2025 (4,254 games) it did NOT beat the closing line:
model coefficient +0.099, t = +1.19. Those constants are in the header
below and shown in the app, because the honest thing is to let the
record accumulate against a stated prior rather than hide it.

v1.2: bets are priced at your book (consensus line at -110 by default),
the card and the tracker use one tiering rule (OFFICIAL = positive EV,
WATCH = big disagreement that does not beat the vig), closing lines are
the last PREGAME snapshot, and wins pay at the recorded price.
"""

import warnings
warnings.filterwarnings("ignore")

import html as _html
import json
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import Ridge

import requests

import nfl_data_py as nfl

import re
import types

# ======================================================================
# Injury model (formerly injuries.py, merged in so there is one file to
# deploy). The power rating has no idea who is playing; this prices who is
# on the injury report, measured on history, and nothing is applied until
# it has been measured. Quarterbacks are included as their own group and
# skipped automatically if the separate QB module (qb_adjustment.json) is on.
# ======================================================================
# Position groups. Specialists are deliberately absent.
GROUPS = {
    "QB":    {"QB"},
    "OL":    {"T", "OT", "G", "OG", "C", "OL", "LT", "RT", "LG", "RG"},
    "SKILL": {"WR", "TE", "RB", "FB", "HB"},
    "FRONT": {"DE", "DT", "NT", "DL", "LB", "ILB", "OLB", "MLB", "EDGE"},
    "DB":    {"CB", "S", "FS", "SS", "DB", "SAF"},
}
OFFENSE = ("OL", "SKILL")          # non-QB offense, one totals term
OFF_SIDE = ("QB", "OL", "SKILL")   # whose snap share is offensive snaps
DEFENSE = ("FRONT", "DB")
GROUP_LABEL = {"QB": "quarterback", "OL": "offensive line", "SKILL": "WR/TE/RB",
               "FRONT": "front seven", "DB": "secondary"}

# Chance each status means the player sits. Used ONLY until measure() has run;
# the measured rates in injury_adjustment.json replace these.
DEFAULT_MISS_PROB = {"out": 1.0, "doubtful": 0.9, "questionable": 0.25,
                     # On injured reserve / PUP: off the weekly report
                     # entirely, so found through the weekly roster instead.
                     "reserve": 1.0}

# Early in a season there are only a game or two of snaps. Blend in last
# season's share as if it were this many games of evidence.
PRIOR_GAMES = 4.0

# One code per franchise across every nflverse / PFR table.
TEAM_CANON = {
    "GNB": "GB", "KAN": "KC", "NWE": "NE", "NOR": "NO", "SFO": "SF",
    "TAM": "TB", "LVR": "LV", "OAK": "LV", "SDG": "LAC", "SD": "LAC",
    "STL": "LA", "LAR": "LA", "RAM": "LA", "JAC": "JAX", "WSH": "WAS",
    "HST": "HOU", "BLT": "BAL", "CLV": "CLE", "ARZ": "ARI",
}


def canon(team):
    t = str(team).upper().strip()
    return TEAM_CANON.get(t, t)


def group_of(pos):
    p = str(pos).upper().strip()
    for g, s in GROUPS.items():
        if p in s:
            return g
    return None


def name_key(name):
    """'Kenneth Walker III' and 'Kenneth Walker' should be the same man."""
    s = str(name).lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", s)
    return re.sub(r"[^a-z]", "", s)


# ----------------------------------------------------------------------
# Data prep
# ----------------------------------------------------------------------
def prep_injuries(inj):
    """nflverse weekly injury report -> one row per listed player-week."""
    if inj is None or len(inj) == 0:
        return pd.DataFrame(columns=["season", "week", "team", "player",
                                     "key", "group", "status"])
    d = inj.copy()
    name = d["full_name"] if "full_name" in d.columns else (
        d.get("first_name", "").astype(str) + " " + d.get("last_name", "").astype(str))
    d = pd.DataFrame({
        "season": pd.to_numeric(d["season"], errors="coerce"),
        "week": pd.to_numeric(d["week"], errors="coerce"),
        "team": d["team"].map(canon),
        "player": name.astype(str),
        "group": d["position"].map(group_of),
        "status": d["report_status"].astype(str).str.lower().str.strip(),
    })
    d = d[d["status"].isin(DEFAULT_MISS_PROB.keys()) & d["group"].notna()]
    d = d.dropna(subset=["season", "week"])
    d["key"] = d["player"].map(name_key)
    return d.drop_duplicates(["season", "week", "team", "key"])


def prep_reserve(ros):
    """
    Weekly rosters -> players on reserve (IR, PUP, NFI) that week, in the
    same shape as prep_injuries.

    This closes the injury report's biggest hole: a starter placed on IR
    drops off the report, so a season-ending injury was invisible. Only
    players who have taken snaps this season are counted (enforced in
    team_injury_loads), so reserve/retired names and camp casualties with
    no role on this year's team are not.
    """
    cols = ["season", "week", "team", "player", "key", "group", "status"]
    if ros is None or len(ros) == 0:
        return pd.DataFrame(columns=cols)
    d = ros
    stt = d["status"].astype(str).str.upper().str.strip() \
        if "status" in d.columns else pd.Series("", index=d.index)
    d = d[stt == "RES"]
    if d.empty:
        return pd.DataFrame(columns=cols)
    name = (d["full_name"] if "full_name" in d.columns
            else d["player_name"] if "player_name" in d.columns
            else d.get("first_name", "").astype(str) + " "
            + d.get("last_name", "").astype(str))
    out = pd.DataFrame({
        "season": pd.to_numeric(d["season"], errors="coerce"),
        "week": pd.to_numeric(d.get("week"), errors="coerce"),
        "team": d["team"].map(canon),
        "player": name.astype(str),
        "group": d["position"].map(group_of),
        "status": "reserve",
    })
    out = out.dropna(subset=["season", "week", "group"])
    out["key"] = out["player"].map(name_key)
    return out[cols].drop_duplicates(["season", "week", "team", "key"])


def combine_reports(inj, reserve):
    """Injury report plus reserve list. Anyone on both keeps the report
    entry, which is the more specific of the two."""
    if reserve is None or len(reserve) == 0:
        return inj
    both = pd.concat([inj, reserve[inj.columns]], ignore_index=True)
    return both.drop_duplicates(["season", "week", "team", "key"], keep="first")


def prep_snaps(sn):
    """nflverse snap counts -> player-game rows with the relevant share."""
    if sn is None or len(sn) == 0:
        return pd.DataFrame(columns=["season", "week", "game_id", "team",
                                     "key", "group", "share"])
    d = sn.copy()
    off = pd.to_numeric(d.get("offense_pct"), errors="coerce").fillna(0.0)
    dfn = pd.to_numeric(d.get("defense_pct"), errors="coerce").fillna(0.0)
    # Some versions store 95 rather than 0.95.
    if off.max() > 1.5:
        off = off / 100.0
    if dfn.max() > 1.5:
        dfn = dfn / 100.0
    grp = d["position"].map(group_of)
    share = np.where(grp.isin(OFF_SIDE), off, dfn)
    out = pd.DataFrame({
        "season": pd.to_numeric(d["season"], errors="coerce"),
        "week": pd.to_numeric(d["week"], errors="coerce"),
        "game_id": d["game_id"].astype(str),
        "team": d["team"].map(canon),
        "key": d["player"].map(name_key),
        "group": grp,
        "share": share,
    })
    return out.dropna(subset=["season", "week", "group"])


def _player_share(snaps, season, week, team_games_col="team"):
    """
    Each player's share of his team's snaps before this week, blended with
    last season's share early on.

    The share is LOCKED at its pre-absence level: it is measured over the
    team's games up to the last one he played, not over every game since.
    The ratings are fit net of injuries, so they describe each team at full
    strength; a starter who has been out a month is still missing a starter's
    worth, and letting his share decay would slowly stop charging for it.
    """
    cur = snaps[(snaps["season"] == season) & (snaps["week"] < week)]
    prev = snaps[snaps["season"] == season - 1]

    def _shares(df):
        if df.empty:
            return pd.Series(dtype=float), pd.Series(dtype=float)
        games = df[["team", "week", "game_id"]].drop_duplicates()
        cum = (games.groupby(["team", "week"]).size()
                    .groupby(level=0).cumsum().rename("n_thru").reset_index())
        last = (df.sort_values("week").groupby("key")
                  .agg(team=("team", "last"), week=("week", "max"))
                  .reset_index())
        last = last.merge(cum, on=["team", "week"], how="left")
        n = last.set_index("key")["n_thru"]
        tot = df.groupby("key")["share"].sum()
        n = n.reindex(tot.index).fillna(1.0).clip(lower=1)
        return tot, n

    s_cur, n_cur = _shares(cur)
    s_prev, n_prev = _shares(prev)
    keys = s_cur.index.union(s_prev.index)
    s_cur = s_cur.reindex(keys).fillna(0.0)
    n_cur = n_cur.reindex(keys).fillna(0.0)
    prev_rate = (s_prev / n_prev).reindex(keys).fillna(0.0).clip(0, 1)
    k = PRIOR_GAMES
    rate = (s_cur + k * prev_rate) / (n_cur + k)
    return rate.clip(0, 1)


def team_injury_loads(inj_week, snaps, season, week, miss_prob=None,
                      top_n=4):
    """
    {team: {"OL": x, "SKILL": x, "FRONT": x, "DB": x, "players": [...]}}

    A load of 1.0 in a group means one full-time player's worth of snaps is
    expected to be missing.
    """
    mp = dict(DEFAULT_MISS_PROB)
    if miss_prob:
        mp.update({k: float(v) for k, v in miss_prob.items() if k in mp})
    iw = inj_week[(inj_week["season"] == season) & (inj_week["week"] == week)]
    if iw.empty:
        return {}
    share = _player_share(snaps, season, week)
    # Reserve players count only if they have played for this team this
    # season. That keeps retired or long-gone names off the ledger.
    played_now = set(snaps.loc[(snaps["season"] == season)
                               & (snaps["week"] < week), "key"])
    iw = iw[(iw["status"] != "reserve") | iw["key"].isin(played_now)]
    if iw.empty:
        return {}
    iw = iw.assign(share=iw["key"].map(share).fillna(0.0),
                   p_miss=iw["status"].map(mp).fillna(0.0))
    iw = iw.assign(load=iw["share"] * iw["p_miss"])
    out = {}
    for team, d in iw.groupby("team"):
        rec = {g: float(d.loc[d["group"] == g, "load"].sum()) for g in GROUPS}
        top = d[d["load"] > 0.05].sort_values("load", ascending=False).head(top_n)
        rec["players"] = [
            {"player": r.player, "group": r.group, "status": r.status,
             "share": round(float(r.share), 2), "load": round(float(r.load), 2)}
            for r in top.itertuples()]
        out[str(team)] = rec
    return out


def season_loads(inj_week, snaps, season, miss_prob=None):
    """Every team-week of one season as a table: season, week, team, and a
    load column per position group. Used for the history the ratings are
    fit on, and by measure()."""
    rows = []
    iw = inj_week[inj_week["season"] == season]
    for week in sorted(iw["week"].dropna().unique()):
        loads = team_injury_loads(inj_week, snaps, int(season), int(week),
                                  miss_prob, top_n=0)
        for team, rec in loads.items():
            rows.append({"season": int(season), "week": int(week),
                         "team": team, **{g: rec[g] for g in GROUPS}})
    return pd.DataFrame(rows, columns=["season", "week", "team", *GROUPS])


# ----------------------------------------------------------------------
# Turning loads into points
# ----------------------------------------------------------------------
def usable(adj, section, name, expect_sign, min_t):
    """A coefficient is used only if it points the way football says it
    should AND it cleared the significance bar. Otherwise it is zero."""
    try:
        c = adj[section][name]
        coef, t = float(c["coef"]), float(c["t"])
    except Exception:
        return 0.0
    if not (math.isfinite(coef) and math.isfinite(t)):
        return 0.0
    if np.sign(coef) != expect_sign or abs(t) < min_t:
        return 0.0
    return coef


def _last(name):
    toks = [t for t in str(name).split()
            if t.strip(".").lower() not in {"jr", "sr", "ii", "iii", "iv", "v"}]
    return toks[-1] if toks else str(name)


def history_offsets(adj, table, games, min_t=2.0, cap=10.0, skip_qb=False):
    """
    For past games: how many points of each result were down to injuries.

    Subtracting these before the ratings are fit is what makes the weekly
    adjustment INCREMENTAL. A team that lost with three linemen out is not
    marked down as if it were simply worse, so its rating is its healthy
    strength; this week's report then subtracts only what is actually
    missing this week. An absence that is already in the ratings is not
    charged a second time.

    Returns (margin_offset, total_offset) aligned to games.index.
    """
    z = pd.Series(0.0, index=games.index)
    if not adj or table is None or len(table) == 0 or len(games) == 0:
        return z, z.copy()
    L = table.copy()
    # A table cached by an older version may lack a group (e.g. QB). Treat
    # a missing group as no injuries rather than crashing.
    for c in GROUPS:
        if c not in L.columns:
            L[c] = 0.0
    L["team"] = L["team"].map(canon)
    key = ["season", "week"]
    g = games[key].copy()
    g["season"] = pd.to_numeric(g["season"], errors="coerce")
    g["week"] = pd.to_numeric(g["week"], errors="coerce")
    g["home_team"] = games["home_team"].map(canon)
    g["away_team"] = games["away_team"].map(canon)
    g["_i"] = games.index
    for side in ("home", "away"):
        ren = {c: f"{side}_{c}" for c in GROUPS}
        ren["team"] = f"{side}_team"
        g = g.merge(L.rename(columns=ren), on=key + [f"{side}_team"],
                    how="left")
    g = g.drop_duplicates("_i").set_index("_i").reindex(games.index)
    for side in ("home", "away"):
        for c in GROUPS:
            g[f"{side}_{c}"] = pd.to_numeric(g[f"{side}_{c}"],
                                             errors="coerce").fillna(0.0)
    grps = [c for c in GROUPS if not (skip_qb and c == "QB")]
    dm = sum(usable(adj, "margin", c, -1, min_t)
             * (g[f"home_{c}"] - g[f"away_{c}"]) for c in grps)
    off = sum(g[f"home_{c}"] + g[f"away_{c}"] for c in OFFENSE)
    dfn = sum(g[f"home_{c}"] + g[f"away_{c}"] for c in DEFENSE)
    dt = (usable(adj, "total", "OFF", -1, min_t) * off
          + usable(adj, "total", "DEF", +1, min_t) * dfn)
    if not skip_qb:
        dt = dt + usable(adj, "total", "QB", -1, min_t) * (g["home_QB"] + g["away_QB"])
    return (pd.Series(np.clip(dm, -cap, cap), index=games.index),
            pd.Series(np.clip(dt, -cap, cap), index=games.index))


def game_deltas(adj, loads, home, away, min_t=2.0, cap=10.0, skip_qb=False):
    """
    Points added to the home margin and to the total, plus a short note.
    Returns (margin_delta, total_delta, note, detail).
    """
    if not adj or not loads:
        return 0.0, 0.0, None, {}
    zero = {g: 0.0 for g in GROUPS}
    lh = loads.get(canon(home), zero)
    la = loads.get(canon(away), zero)

    dm = 0.0
    for g in GROUPS:
        if skip_qb and g == "QB":
            continue
        coef = usable(adj, "margin", g, -1, min_t)
        dm += coef * (lh.get(g, 0.0) - la.get(g, 0.0))

    off = sum(lh.get(g, 0.0) + la.get(g, 0.0) for g in OFFENSE)
    dfn = sum(lh.get(g, 0.0) + la.get(g, 0.0) for g in DEFENSE)
    dt = (usable(adj, "total", "OFF", -1, min_t) * off
          + usable(adj, "total", "DEF", +1, min_t) * dfn)
    if not skip_qb:
        dt += usable(adj, "total", "QB", -1, min_t) * (
            lh.get("QB", 0.0) + la.get("QB", 0.0))

    # Safety rail, not a measurement.
    dm = float(np.clip(dm, -cap, cap))
    dt = float(np.clip(dt, -cap, cap))

    parts = []
    for team, l in ((away, la), (home, lh)):
        names = [_last(p["player"]) + (" (QB)" if p["group"] == "QB" else "")
                 for p in l.get("players", [])[:3]
                 if not (skip_qb and p["group"] == "QB")]
        if names:
            parts.append(f"{team}: {', '.join(names)}")
    note = "; ".join(parts) if parts and (abs(dm) >= 0.1 or abs(dt) >= 0.1) else None
    return dm, dt, note, {"home": lh, "away": la}


# ----------------------------------------------------------------------
# Measurement
# ----------------------------------------------------------------------
def _fit(X, y, n_free, penalty=2.0):
    """
    Least squares where the first n_free columns (the injury terms and the
    intercept) are unpenalised and the rest (team-season strength dummies)
    get a light ridge, which also removes the dummies' redundant direction.
    Returns coefficients and standard errors for the first n_free columns.
    """
    k = X.shape[1]
    P = np.zeros((k, k))
    P[n_free:, n_free:] = np.eye(k - n_free) * penalty
    A = X.T @ X + P
    Ainv = np.linalg.pinv(A)
    beta = Ainv @ (X.T @ y)
    resid = y - X @ beta
    dof = max(len(y) - k, 1)
    s2 = float(resid @ resid) / dof
    cov = s2 * Ainv @ (X.T @ X) @ Ainv
    se = np.sqrt(np.clip(np.diag(cov)[:n_free], 1e-12, None))
    return beta[:n_free], se, math.sqrt(s2)


def _ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.pinv(X.T @ X)
    return beta, np.sqrt(np.clip(np.diag(cov), 1e-12, None))


def _stat(coef, se):
    return {"coef": round(float(coef), 4), "se": round(float(se), 4),
            "t": round(float(coef / se), 2) if se > 0 else 0.0}


def measure(seasons=None, nfl=None, log=print):
    """
    Estimate what injury load is worth, on history. Returns the dict that is
    saved as injury_adjustment.json.

    Value: home margin (and total) regressed on the injury loads with a
    strength term for every team-season, so the estimate comes from weeks
    a team was more or less healthy than its own average — the same thing
    the power rating cannot see.

    Market check: the same loads against the closing-line error. Near zero
    means the market already prices injuries, and the adjustment is only
    stopping the model from inventing edges. That is still worth doing.
    """
    if nfl is None:
        import nfl_data_py as nfl
    if seasons is None:
        last = datetime.now().year - 1
        seasons = list(range(2013, last + 1))
    seasons = [int(s) for s in seasons]
    snap_seasons = sorted(set(seasons) | {min(seasons) - 1})

    log(f"Loading schedules {seasons[0]}-{seasons[-1]}...")
    sched = nfl.import_schedules(seasons)
    log("Loading injury reports...")
    inj = prep_injuries(nfl.import_injuries(seasons))
    log("Loading weekly rosters (injured reserve)...")
    try:
        inj = combine_reports(
            inj, prep_reserve(nfl.import_weekly_rosters(seasons)))
    except Exception as e:
        log(f"Weekly rosters unavailable ({e}); report only.")
    log("Loading snap counts...")
    snaps = prep_snaps(nfl.import_snap_counts(snap_seasons))

    # --- How often does each status actually sit? -------------------------
    # Only players who log snaps at some point that season, so a name that
    # simply fails to match is not mistaken for a player who sat.
    seen = snaps[["season", "key"]].drop_duplicates()
    played = snaps[snaps["share"] > 0][["season", "week", "team", "key"]] \
        .drop_duplicates().assign(played=1)
    s = inj.merge(seen, on=["season", "key"]) \
           .merge(played, on=["season", "week", "team", "key"], how="left")
    s["played"] = s["played"].fillna(0)
    miss_prob, status_n = {}, {}
    for st_, grp in s.groupby("status"):
        miss_prob[st_] = round(float(1.0 - grp["played"].mean()), 3)
        status_n[st_] = int(len(grp))
    log(f"Miss rates by status: {miss_prob}")

    # --- Loads for every team-week ----------------------------------------
    L = pd.concat([season_loads(inj, snaps, int(s_), miss_prob)
                   for s_ in sorted(inj["season"].unique())],
                  ignore_index=True)
    if L.empty:
        raise RuntimeError("No injury loads could be built.")

    g = sched.dropna(subset=["home_score", "away_score"]).copy()
    g["home_team"] = g["home_team"].map(canon)
    g["away_team"] = g["away_team"].map(canon)
    g["season"] = pd.to_numeric(g["season"], errors="coerce")
    g["week"] = pd.to_numeric(g["week"], errors="coerce")
    for side in ("home", "away"):
        ren = {c: f"{side}_{c}" for c in GROUPS}
        ren["team"] = f"{side}_team"
        g = g.merge(L.rename(columns=ren),
                    on=["season", "week", f"{side}_team"], how="left")
    for side in ("home", "away"):
        for c in GROUPS:
            g[f"{side}_{c}"] = g[f"{side}_{c}"].fillna(0.0)
    g["margin"] = g["home_score"] - g["away_score"]
    g["total"] = g["home_score"] + g["away_score"]
    n = len(g)
    log(f"{n:,} completed games with injury data.")

    # Team-season strength dummies.
    ts = sorted(set(zip(g["season"], g["home_team"]))
                | set(zip(g["season"], g["away_team"])))
    ix = {k: i for i, k in enumerate(ts)}
    D_m = np.zeros((n, len(ts)))
    D_t = np.zeros((n, len(ts)))
    for r, (se_, h, a) in enumerate(zip(g["season"], g["home_team"], g["away_team"])):
        D_m[r, ix[(se_, h)]] = 1.0
        D_m[r, ix[(se_, a)]] = -1.0
        D_t[r, ix[(se_, h)]] = 1.0
        D_t[r, ix[(se_, a)]] = 1.0

    grp = list(GROUPS)
    diff = np.column_stack([g[f"home_{c}"] - g[f"away_{c}"] for c in grp])
    off = sum(g[f"home_{c}"] + g[f"away_{c}"] for c in OFFENSE).values
    dfn = sum(g[f"home_{c}"] + g[f"away_{c}"] for c in DEFENSE).values
    ones = np.ones((n, 1))

    # Value to the rating.
    Xm = np.hstack([diff, ones, D_m])
    bm, sem, sdm = _fit(Xm, g["margin"].values.astype(float), len(grp) + 1)
    qb = (g["home_QB"] + g["away_QB"]).values
    Xt = np.hstack([off[:, None], dfn[:, None], qb[:, None], ones, D_t])
    bt, set_, sdt = _fit(Xt, g["total"].values.astype(float), 4)

    out = {
        "version": 4,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seasons": [seasons[0], seasons[-1]],
        "n_games": n,
        "miss_prob": miss_prob,
        "status_n": status_n,
        "margin": {c: _stat(bm[i], sem[i]) for i, c in enumerate(grp)},
        "total": {"OFF": _stat(bt[0], set_[0]), "DEF": _stat(bt[1], set_[1]),
                  "QB": _stat(bt[2], set_[2])},
        "resid_sd": {"margin": round(sdm, 2), "total": round(sdt, 2)},
    }

    # Market check: does injury load predict the closing line's error?
    mc = {}
    if "spread_line" in g.columns:
        d = g.dropna(subset=["spread_line"])
        if len(d) > 200:
            sign = 1.0 if np.corrcoef(d["spread_line"], d["margin"])[0, 1] > 0 else -1.0
            err = (d["margin"] - sign * d["spread_line"]).values.astype(float)
            dd = np.column_stack([d[f"home_{c}"] - d[f"away_{c}"] for c in grp]
                                 + [np.ones(len(d))])
            b, se = _ols(dd, err)
            mc["margin"] = {c: _stat(b[i], se[i]) for i, c in enumerate(grp)}
    if "total_line" in g.columns:
        d = g.dropna(subset=["total_line"])
        if len(d) > 200:
            err = (d["total"] - d["total_line"]).values.astype(float)
            o = sum(d[f"home_{c}"] + d[f"away_{c}"] for c in OFFENSE).values
            f = sum(d[f"home_{c}"] + d[f"away_{c}"] for c in DEFENSE).values
            q = (d["home_QB"] + d["away_QB"]).values
            b, se = _ols(np.column_stack([o, f, q, np.ones(len(d))]), err)
            mc["total"] = {"OFF": _stat(b[0], se[0]), "DEF": _stat(b[1], se[1]),
                           "QB": _stat(b[2], se[2])}
    out["market_check"] = mc

    # Individual quarterback ratings. Optional: if play-by-play cannot be
    # loaded the injury model still stands, with its flat QB group.
    try:
        out["qb"] = measure_qb(nfl, sched, [s_ for s_ in seasons if s_ >= 2016],
                               log=log)
    except Exception as e:
        log(f"QB ratings not measured: {type(e).__name__}: {e}")
        out["qb_error"] = f"{type(e).__name__}: {e}"
    log(json.dumps(out, indent=2))
    return out

# ======================================================================
# Quarterback ratings
#
# Every quarterback gets his own value: expected points added per dropback,
# built so a small sample cannot fool it, and measured against real games
# before it is allowed to move a line.
#
#   * Garbage time out: only plays with win probability 10-90%.
#   * Opponent-adjusted: each play is credited net of how many EPA that
#     defense has been allowing.
#   * Recency-weighted: a play loses half its weight every QB_HALF_LIFE
#     weeks, so this season counts most and three years ago barely.
#   * Blended: EPA/dropback is noisy; completion % over expected and sack
#     rate settle faster. How much each predicts future EPA is fit on past
#     QB seasons, not assumed.
#   * Shrunk to a prior set by draft slot, measured from how quarterbacks
#     drafted in that range actually played in their first 300 dropbacks.
#
# The points value (how many points one EPA/dropback of QB quality is worth)
# is measured on history with a strength term for every team-season, so it
# comes from games where a team's QB changed mid-season.
# ======================================================================
QB_HALF_LIFE = 20.0          # weeks
QB_LOOKBACK = 4              # seasons of plays used for a rating
QB_K = {"epa": 250.0, "cpoe": 150.0, "sack": 150.0}   # shrinkage, dropbacks
QB_K_DEF = 250.0             # shrinkage for the defense adjustment
QB_MIN_T = 2.0
QB_CAP_PTS = 12.0            # safety rail per game


def _draft_bucket(rnd):
    try:
        r = int(rnd)
    except Exception:
        return "late"
    return "r1" if r == 1 else ("r23" if r <= 3 else "late")


def prep_draft(dp):
    """{player key: bucket} for drafted quarterbacks. Keyed by gsis id AND
    name so either identifier finds him."""
    out = {}
    if dp is None or len(dp) == 0:
        return out
    d = dp.copy()
    pos = d.get("position", d.get("pos", pd.Series("", index=d.index)))
    d = d[pos.astype(str).str.upper() == "QB"]
    rcol = "round" if "round" in d.columns else None
    if rcol is None:
        return out
    namecol = ("pfr_player_name" if "pfr_player_name" in d.columns
               else "player_name" if "player_name" in d.columns else None)
    for _, r in d.iterrows():
        b = _draft_bucket(r[rcol])
        gid = r.get("gsis_id")
        if isinstance(gid, str) and gid:
            out[gid] = b
        if namecol and isinstance(r.get(namecol), str):
            out["name:" + name_key(r[namecol])] = b
    return out


def prep_pbp(pbp):
    """nflverse play-by-play -> quarterback dropbacks only."""
    cols = ["season", "week", "t", "team", "opp", "qb_id", "qb_name", "epa",
            "cpoe", "sack", "att", "ok"]
    if pbp is None or len(pbp) == 0:
        return pd.DataFrame(columns=cols)
    d = pbp
    if "season_type" in d.columns:
        d = d[d["season_type"].astype(str).str.upper().isin(["REG", "POST"])]
    db = pd.to_numeric(d.get("qb_dropback"), errors="coerce").fillna(0)
    d = d[db == 1]
    qid = d["passer_id"] if "passer_id" in d.columns else d.get("passer_player_id")
    qnm = d["passer"] if "passer" in d.columns else d.get("passer_player_name")
    if "rusher_player_id" in d.columns:            # scrambles, older data
        qid = qid.fillna(d["rusher_player_id"])
        if "rusher_player_name" in d.columns:
            qnm = qnm.fillna(d["rusher_player_name"])
    epa = pd.to_numeric(d.get("qb_epa", d.get("epa")), errors="coerce")
    wp = pd.to_numeric(d.get("wp"), errors="coerce")
    sack = pd.to_numeric(d.get("sack"), errors="coerce").fillna(0)
    att = pd.to_numeric(d.get("pass_attempt"), errors="coerce").fillna(0)
    out = pd.DataFrame({
        "season": pd.to_numeric(d["season"], errors="coerce"),
        "week": pd.to_numeric(d["week"], errors="coerce"),
        "game_id": d["game_id"].astype(str),
        "team": d["posteam"].map(canon),
        "opp": d["defteam"].map(canon),
        "qb_id": qid.astype(str),
        "qb_name": qnm.astype(str),
        "epa": epa,
        "cpoe": pd.to_numeric(d.get("cpoe"), errors="coerce"),
        "sack": sack,
        "att": ((att == 1) & (sack == 0)).astype(float),
        # Garbage time out; unknown win probability kept.
        "ok": (wp.isna() | ((wp >= 0.10) & (wp <= 0.90))),
    })
    out = out.dropna(subset=["season", "week", "epa"])
    out = out[out["qb_id"].str.len() > 3]
    out["t"] = out["season"] * 100 + out["week"]
    return out


def qb_components(db, season, week, draft, priors):
    """
    Each QB's shrunk EPA/dropback, CPOE and sack rate, using only plays
    before (season, week). Returns a DataFrame indexed by qb_id.
    """
    t0 = season * 100 + week
    d = db[(db["t"] < t0) & (db["season"] >= season - QB_LOOKBACK) & db["ok"]]
    if d.empty:
        return pd.DataFrame(columns=["name", "team", "n", "epa", "cpoe",
                                     "sack", "bucket"])
    age = (season - d["season"]) * 20.0 + (week - d["week"])
    w = np.power(0.5, age.clip(lower=0) / QB_HALF_LIFE)
    d = d.assign(w=w)
    lg = float(np.average(d["epa"], weights=d["w"]))
    # Defense adjustment from the same window, same weights.
    dd = d.assign(x=d["w"] * (d["epa"] - lg)).groupby("opp")
    def_eff = dd["x"].sum() / (dd["w"].sum() + QB_K_DEF)
    d = d.assign(ea=d["epa"] - d["opp"].map(def_eff).fillna(0.0))
    has_c = d["cpoe"].notna() & (d["att"] > 0)
    d = d.assign(we=d["w"] * d["ea"], ws=d["w"] * d["sack"],
                 wc=np.where(has_c, d["w"] * d["cpoe"].fillna(0), 0.0),
                 wcn=np.where(has_c, d["w"], 0.0))
    g = d.sort_values("t").groupby("qb_id").agg(
        W=("w", "sum"), E=("we", "sum"), S=("ws", "sum"), C=("wc", "sum"),
        CN=("wcn", "sum"), n=("w", "size"), name=("qb_name", "last"),
        team=("team", "last"))
    b = pd.Series([draft.get(i, draft.get("name:" + name_key(nm), "late"))
                   for i, nm in zip(g.index, g["name"])], index=g.index)
    pe = b.map(lambda x: priors.get(x, priors["late"])["epa"])
    pc = b.map(lambda x: priors.get(x, priors["late"])["cpoe"])
    ps = b.map(lambda x: priors.get(x, priors["late"])["sack"])
    return pd.DataFrame({
        "name": g["name"], "team": g["team"], "n": g["n"], "bucket": b,
        "epa": (g["E"] + QB_K["epa"] * pe) / (g["W"] + QB_K["epa"]),
        "cpoe": (g["C"] + QB_K["cpoe"] * pc) / (g["CN"] + QB_K["cpoe"]),
        "sack": (g["S"] + QB_K["sack"] * ps) / (g["W"] + QB_K["sack"]),
    })


def qb_rating_of(comp_row_or_bucket, qbp):
    """Composite rating (EPA/dropback scale) from components, or from the
    draft-bucket prior for a QB with no plays."""
    c = qbp["composite"]
    if isinstance(comp_row_or_bucket, str):
        p = qbp["priors"].get(comp_row_or_bucket, qbp["priors"]["late"])
        e, cp, sk = p["epa"], p["cpoe"], p["sack"]
    else:
        e, cp, sk = (comp_row_or_bucket["epa"], comp_row_or_bucket["cpoe"],
                     comp_row_or_bucket["sack"])
    return float(c["a"] + c["epa"] * e + c["cpoe"] * cp + c["sack"] * sk)


def qb_ratings(db, season, week, draft, qbp):
    comp = qb_components(db, season, week, draft, qbp["priors"])
    if comp.empty:
        return comp.assign(rating=pd.Series(dtype=float))
    c = qbp["composite"]
    return comp.assign(rating=c["a"] + c["epa"] * comp["epa"]
                       + c["cpoe"] * comp["cpoe"] + c["sack"] * comp["sack"])


def game_starters(db):
    """Who actually quarterbacked each past game: most dropbacks per team."""
    if db.empty:
        return pd.DataFrame(columns=["season", "week", "game_id", "team", "qb_id"])
    n = (db.groupby(["season", "week", "game_id", "team", "qb_id"]).size()
           .rename("n").reset_index()
           .sort_values("n", ascending=False)
           .drop_duplicates(["game_id", "team"]))
    return n[["season", "week", "game_id", "team", "qb_id"]]


def starter_ratings_table(db, seasons, draft, qbp):
    """Pregame rating of the QB who actually started, for every team-game in
    these seasons. Used for the past games the power ratings are fit on."""
    st_ = game_starters(db[db["season"].isin(list(seasons))])
    rows = []
    for (s_, w_), grp in st_.groupby(["season", "week"]):
        r = qb_ratings(db, int(s_), int(w_), draft, qbp)["rating"]
        for row in grp.itertuples():
            rows.append({"season": int(s_), "week": int(w_), "team": row.team,
                         "qb_id": row.qb_id,
                         "rating": float(r.get(row.qb_id, qb_rating_of("late", qbp)))})
    return pd.DataFrame(rows, columns=["season", "week", "team", "qb_id", "rating"])


def qb_usable(qbp, key="margin"):
    try:
        c = qbp[key]
        return c["coef"] > 0 and c["t"] >= QB_MIN_T
    except Exception:
        return False


def _fe(games, symmetric):
    ts = sorted(set(zip(games["season"], games["home_team"]))
                | set(zip(games["season"], games["away_team"])))
    ix = {k: i for i, k in enumerate(ts)}
    D = np.zeros((len(games), len(ts)))
    for r, (s_, h, a) in enumerate(zip(games["season"], games["home_team"],
                                       games["away_team"])):
        D[r, ix[(s_, h)]] = 1.0
        D[r, ix[(s_, a)]] = 1.0 if symmetric else -1.0
    return D


def measure_qb(nfl, sched, seasons, log=print):
    """
    Fit the QB model: draft priors, the component blend, and the points
    value of QB quality. Returns the dict stored under "qb".
    """
    seasons = [int(s) for s in seasons]
    pbp_seasons = list(range(min(seasons) - 3, max(seasons) + 1))
    log("Loading play-by-play for QB ratings...")
    db = prep_pbp(load_pbp(nfl, pbp_seasons))
    if db.empty:
        raise RuntimeError("no play-by-play")
    try:
        draft = prep_draft(nfl.import_draft_picks())
    except Exception:
        draft = {}

    # Priors: how QBs from each draft range played in their first 300
    # dropbacks. Garbage time out.
    ok = db[db["ok"]].sort_values("t")
    ok = ok.assign(i=ok.groupby("qb_id").cumcount())
    early = ok[ok["i"] < 300]
    bkt = [draft.get(i, draft.get("name:" + name_key(n), "late"))
           for i, n in zip(early["qb_id"], early["qb_name"])]
    early = early.assign(b=bkt)
    priors = {}
    for b_ in ("r1", "r23", "late"):
        e = early[early["b"] == b_]
        if len(e) < 500:
            e = early
        att = e[e["att"] > 0]
        priors[b_] = {"epa": float(e["epa"].mean()),
                      "cpoe": float(att["cpoe"].mean()),
                      "sack": float(e["sack"].mean())}
    log(f"QB priors by draft slot: {priors}")

    # Composite: which shrunk components predict a QB's EPA/dropback in the
    # season AHEAD. Weighted by that season's dropbacks.
    X, y, wts = [], [], []
    for s_ in seasons:
        comp = qb_components(db, s_, 1, draft, priors)
        cur = db[(db["season"] == s_) & db["ok"]].groupby("qb_id")["epa"] \
            .agg(["mean", "size"])
        cur = cur[cur["size"] >= 150]
        j = comp.join(cur, how="inner")
        for r in j.itertuples():
            X.append([1.0, r.epa, r.cpoe, r.sack])
            y.append(float(r.mean))
            wts.append(float(r.size))
    X, y, wts = np.array(X), np.array(y, dtype=float), np.array(wts)
    if len(y) < 40:
        raise RuntimeError("too few QB seasons for the blend")
    sw = np.sqrt(wts)
    # A component must point the way football says (EPA and CPOE up, sacks
    # down). One that fits backwards is noise; drop it and refit.
    names = ["a", "epa", "cpoe", "sack"]
    sign = {"epa": 1, "cpoe": 1, "sack": -1}
    keep = [0, 1, 2, 3]
    for _ in range(3):
        b_, *_r = np.linalg.lstsq(X[:, keep] * sw[:, None], y * sw, rcond=None)
        coef = dict(zip([names[i] for i in keep], b_))
        bad = [names.index(k_) for k_, v in coef.items()
               if k_ in sign and np.sign(v) != sign[k_]]
        if not bad:
            break
        keep = [i for i in keep if i not in bad]
    composite = {k_: float(coef.get(k_, 0.0)) for k_ in names}
    log(f"QB composite: {composite} on {len(y)} QB-seasons")
    qbp = {"priors": priors, "composite": composite}

    # Points per EPA/dropback of QB quality, from mid-season QB changes.
    tbl = starter_ratings_table(db, seasons, draft, qbp)
    g = sched.dropna(subset=["home_score", "away_score"]).copy()
    for c_ in ("home_team", "away_team"):
        g[c_] = g[c_].map(canon)
    g["season"] = pd.to_numeric(g["season"], errors="coerce")
    g["week"] = pd.to_numeric(g["week"], errors="coerce")
    g = g[g["season"].isin(seasons)]
    for side in ("home", "away"):
        g = g.merge(tbl.rename(columns={"team": f"{side}_team",
                                        "rating": f"{side}_qbr",
                                        "qb_id": f"{side}_qb"}),
                    on=["season", "week", f"{side}_team"], how="inner")
    g = g.drop_duplicates("game_id")
    ref = float(pd.concat([g["home_qbr"], g["away_qbr"]]).mean())
    margin = (g["home_score"] - g["away_score"]).values.astype(float)
    total = (g["home_score"] + g["away_score"]).values.astype(float)
    one = np.ones((len(g), 1))
    Xm = np.hstack([(g["home_qbr"] - g["away_qbr"]).values[:, None], one,
                    _fe(g, False)])
    bm, sem, _ = _fit(Xm, margin, 2)
    Xt = np.hstack([(g["home_qbr"] + g["away_qbr"] - 2 * ref).values[:, None],
                    one, _fe(g, True)])
    bt, set_, _ = _fit(Xt, total, 2)
    out = {"priors": priors, "composite": composite, "ref": ref,
           "n_games": int(len(g)), "n_qb_seasons": int(len(y)),
           "margin": _stat(bm[0], sem[0]), "total": _stat(bt[0], set_[0])}
    log(f"QB points value: {out['margin']} (margin), {out['total']} (total)")
    return out


def load_pbp(nfl, seasons):
    """Only the columns the QB model needs; play-by-play is the heaviest
    nflverse file and Streamlit's memory is limited."""
    want = ["season", "week", "game_id", "season_type", "posteam", "defteam",
            "qb_dropback", "sack", "pass_attempt", "qb_epa", "cpoe", "wp",
            "passer", "passer_id"]
    alt = ["season", "week", "game_id", "season_type", "posteam", "defteam",
           "qb_dropback", "sack", "pass_attempt", "qb_epa", "cpoe", "wp",
           "passer_player_id", "passer_player_name", "rusher_player_id",
           "rusher_player_name"]
    frames = []
    for s_ in seasons:
        df = None
        for cols in (want, alt):
            try:
                df = nfl.import_pbp_data([int(s_)], columns=cols,
                                         downcast=True, cache=False)
                break
            except TypeError:
                df = nfl.import_pbp_data([int(s_)])
                break
            except Exception:
                continue
        if df is not None and len(df):
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# Part of every injury cache key: when the position groups change, anything
# Streamlit cached under the old groups is ignored instead of reused.
INJ_SCHEMA = "v4:" + ",".join(GROUPS)

inj_mod = types.SimpleNamespace(
    GROUPS=GROUPS, GROUP_LABEL=GROUP_LABEL, usable=usable,
    prep_injuries=prep_injuries, prep_snaps=prep_snaps,
    prep_reserve=prep_reserve, combine_reports=combine_reports,
    team_injury_loads=team_injury_loads, season_loads=season_loads,
    history_offsets=history_offsets, game_deltas=game_deltas,
    measure=measure,
)

# Stadium coordinates, and whether the venue is exposed to weather.
# Retractable roofs are treated as outdoor: the roof is usually open in
# fair weather and closed in bad, which is conservative here.
STADIUM = {
    "ARI": (33.528, -112.263, False), "ATL": (33.755, -84.401, False),
    "BAL": (39.278, -76.623, True),   "BUF": (42.774, -78.787, True),
    "CAR": (35.226, -80.853, True),   "CHI": (41.863, -87.617, True),
    "CIN": (39.095, -84.516, True),   "CLE": (41.506, -81.699, True),
    "DAL": (32.748, -97.093, False),  "DEN": (39.744, -105.020, True),
    "DET": (42.340, -83.046, False),  "GB":  (44.501, -88.062, True),
    "HOU": (29.685, -95.411, False),  "IND": (39.760, -86.164, False),
    "JAX": (30.324, -81.637, True),   "KC":  (39.049, -94.484, True),
    "LA":  (33.953, -118.339, False), "LAR": (33.953, -118.339, False),
    "LAC": (33.953, -118.339, False), "LV":  (36.091, -115.183, False),
    "MIA": (25.958, -80.239, True),   "MIN": (44.974, -93.258, False),
    "NE":  (42.091, -71.264, True),   "NO":  (29.951, -90.081, False),
    "NYG": (40.814, -74.074, True),   "NYJ": (40.814, -74.074, True),
    "PHI": (39.901, -75.168, True),   "PIT": (40.447, -80.016, True),
    "SEA": (47.595, -122.332, True),  "SF":  (37.403, -121.970, True),
    "TB":  (27.976, -82.503, True),   "TEN": (36.166, -86.771, True),
    "WAS": (38.908, -76.864, True),
}

# Points off the total per mph of wind, measured on 3,551 outdoor games.
# The effect held out of sample (-0.185 before 2016, -0.214 after) and
# survived realistic forecast error, though at roughly 40% of its
# perfect-knowledge size — hence the discount below.
WIND_PTS_PER_MPH = -0.196

# Forecast error haircut. With actual wind the holdout ROI was +8.2%; with
# a realistic +/-3.5 mph forecast error it was +3.3%. Taking the full
# coefficient would price an accuracy you do not have.
WIND_FORECAST_DISCOUNT = 0.40

# Below this the effect is noise and the market has it priced anyway.
WIND_MIN_MPH = 8.0

# Injuries (non-QB). Coefficients come from injury_adjustment.json, produced
# by injuries.py on history; absent that file nothing is applied. A position
# group's coefficient is used only if it has the sign football predicts AND
# a t-stat of at least this, so the app never acts on a number that could be
# noise.
INJ_MIN_T = 2.0
# Safety rail, not a measurement: no pile of listed players (quarterback
# included) moves a margin or total by more than this.
INJ_MAX_PTS = 10.0
# The measurement re-runs itself this often (it is kept in the Google Sheet
# between runs, so the few-minute cost is paid about once a week).
INJ_REFRESH_DAYS = 7


# Odds API full names -> nflverse abbreviations.
ODDS_TEAM = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN",
    "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE",
    "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}
# nflverse has used both LA and LAR for the Rams depending on version.
ODDS_ALT = {"LA": "LAR", "LAR": "LA"}

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
RIDGE_ALPHA   = 8.0
# Year-over-year persistence of NFL team strength (~0.5-0.6 empirically).
# Applied ONLY where there is no in-season data. This scales last year's
# ratings toward average; too low and every game looks like a pick'em,
# which makes the model take underdogs indiscriminately in early weeks.
# No extra shrinkage on prior-season ratings. The rolling window already
# spans multiple seasons, so r_all is a multi-year average with regression to
# the mean baked in — scaling it again double-counted that. Worse, it
# compressed the rating gaps while home field stayed at full strength, so in
# Week 1 the model drifted toward the home side in every game and the card
# filled up with home underdogs. Ratings and home field now sit on the same
# scale.
CARRYOVER     = 1.00
WINDOW_GAMES  = 320

# Residual SDs measured on 4,254 games, 2007-2025. These convert a point
# edge into a cover probability, so they must come from data, not a guess.
SD_MARGIN     = 13.19
SD_TOTAL      = 13.35

# Weight on the model when blending with the market line. From the
# backtest regression: market 1.011, model 0.099. The model gets 0.099
# because that is what it earned, not because it feels too low.
MODEL_WEIGHT  = 0.099

# Backtest verdict, stated up front and shown in the UI.
BACKTEST_T    = 1.19
BACKTEST_N    = 4254

# Bet threshold in EXPECTED VALUE, not points of edge.
#
# Points only map to EV at a fixed price. Breakeven needs 0.79 points at -110,
# 0.55 at -105 and 0.30 at +100 — so a points bar silently means different
# things at different prices, and throws away the one thing you can actually
# control. Gating on EV means a bet qualifies when the price your book is
# offering makes it worth taking, and never otherwise.
MIN_EV = 0.0

# The price assumed when your book is not in the odds feed. 734 Games is not
# an Odds API bookmaker, so by default the card prices every market at the
# consensus number and this price, rather than at whichever book happened to
# have the best number -- a price you cannot actually get.
ASSUMED_PRICE = -110
CONSENSUS_LABEL = "Consensus line @ -110"

# How far the model must sit from the line for a market to make the card,
# in points of raw disagreement (before the 0.099 blend).
#
# Set from the real distribution: median disagreement is 2.2 pts, the 75th
# percentile 3.9, the 90th 5.5. At 4 points roughly 24% of markets qualify,
# which is about seven plays on a full Sunday, and the chance of a week
# producing nothing is under 2%. At 5 it drops to four plays and one Sunday
# in ten comes up empty; at 3 it is eleven plays, most of the board.
MIN_GAP_PTS = 4.0

# TIERS (v1.2):
#   OFFICIAL  expected value >= MIN_EV at the price you will actually get.
#             With the 0.099 blend a spread needs roughly 8 points of raw
#             disagreement to get there at -110; a strong-wind total can get
#             there on its own. Some weeks this is empty, and that is correct.
#   WATCH     the model is MIN_GAP_PTS+ off the line but the edge does not
#             beat the vig. Frozen to its own ledger so the question "do the
#             big disagreements land?" still gets answered -- without money
#             riding on bets the app's own math says lose ~2% each.


MODEL_VERSION_BASE = f"1.2.0-a{RIDGE_ALPHA}-w{MODEL_WEIGHT}"


def model_version():
    """Threshold is part of the version: change the bar and the record it
    produces is no longer comparable with what came before."""
    v = f"{MODEL_VERSION_BASE}-ev{MIN_EV:g}-g{MIN_GAP_PTS:g}"
    # The injury adjustment changes the model line, so bets frozen with it
    # on are tagged and can be scored separately from those without.
    if load_injury_adjustment():
        v += f"-inj{load_injury_adjustment().get('version', 1)}"
        if qb_model_on():
            v += "-qbr1"
    return v

TRACKER_COLS = [
    "record_key", "frozen_at", "season", "week", "game_id", "kickoff",
    "matchup", "home_team", "away_team", "market_type", "pick_side",
    "pick_label", "bet_line", "model_line", "edge_pts", "cover_prob",
    "expected_value", "odds", "bet_tier", "model_version",
    "status", "result", "units_result", "result_margin",
    "final_home_score", "final_away_score",
    "closing_line", "clv_points", "closing_captured_at", "graded_at",
    "kickoff_utc",
]

st.set_page_config(page_title="Sunday Edge", page_icon="🏈", layout="wide")


# ----------------------------------------------------------------------
# Storage — Google Sheets, with a session fallback
# ----------------------------------------------------------------------
SHEETS_SETUP_STEPS = """1. Google Cloud - create a service account, download its JSON key
2. Streamlit - Settings - Secrets - add a `gcp_service_account_json` entry with the whole JSON pasted in
3. Create a Google Sheet named exactly **sunday_edge_tracker**
4. Share that sheet with the service account's `client_email` as **Editor** (not Viewer)
5. Add `gspread` to requirements.txt, then reboot the app"""


def _sheet(return_error=False):
    """
    Connect to the tracker spreadsheet, and on failure say WHICH step failed.

    The old version returned the raw exception, so a missing spreadsheet, a
    sheet not shared with the service account, and a missing library all
    surfaced as unrelated-looking tracebacks. Each of those has a different
    fix, and the error is the only thing standing between a working tracker
    and bets that vanish on restart.
    """
    try:
        import gspread
    except Exception:
        err = ("gspread is not installed. Add `gspread` to requirements.txt "
               "in your repo and redeploy.")
        return (None, err) if return_error else None

    # Secrets are case-sensitive; accept the usual spellings rather than
    # letting the name be the thing that breaks it.
    raw = None
    for _n in ("gcp_service_account_json", "GCP_SERVICE_ACCOUNT_JSON",
               "gcp_service_account", "GCP_SERVICE_ACCOUNT"):
        try:
            if _n in st.secrets:
                raw = st.secrets[_n]
                break
        except Exception:
            break
    if raw is None:
        err = ("No Google credentials in Secrets. Add a "
               "`gcp_service_account_json` entry containing the service "
               "account JSON (Streamlit → Settings → Secrets).")
        return (None, err) if return_error else None

    try:
        creds = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except Exception as e:
        err = (f"The credentials in Secrets are not valid JSON ({e}). Paste "
               "the whole downloaded key file, braces included.")
        return (None, err) if return_error else None

    email = str(creds.get("client_email", "the service account"))
    try:
        gc = gspread.service_account_from_dict(creds)
    except Exception as e:
        err = f"Google rejected those credentials ({e})."
        return (None, err) if return_error else None

    name = "sunday_edge_tracker"
    for _n in ("tracker_sheet_name", "TRACKER_SHEET_NAME"):
        try:
            if _n in st.secrets:
                name = str(st.secrets[_n])
                break
        except Exception:
            break

    try:
        sh = gc.open(name)
    except Exception as e:
        err = (f"No spreadsheet named '{name}' is visible to this app. "
               f"Create one with exactly that name, then share it with "
               f"{email} as an Editor. "
               f"(If you use a different name, set a `tracker_sheet_name` "
               f"secret — but note both Edge apps read that same secret, so "
               f"give each app its own Streamlit project.) [{type(e).__name__}]")
        return (None, err) if return_error else None

    try:
        ws = sh.worksheet("tracker")
    except Exception:
        try:
            ws = sh.add_worksheet("tracker", rows=2000, cols=len(TRACKER_COLS))
        except Exception as e:
            err = (f"Opened '{name}' but could not create the 'tracker' tab. "
                   f"Check {email} has Editor access, not Viewer. ({e})")
            return (None, err) if return_error else None
    return (ws, None) if return_error else ws


def empty_tracker():
    return pd.DataFrame(columns=TRACKER_COLS)


def load_tracker():
    ws = _sheet()
    if ws is not None:
        try:
            recs = ws.get_all_records()
            df = pd.DataFrame(recs) if recs else empty_tracker()
            for c in TRACKER_COLS:
                if c not in df.columns:
                    df[c] = None
            return df[TRACKER_COLS]
        except Exception:
            pass
    return st.session_state.get("tracker", empty_tracker())


def is_owner():
    """
    Only the owner writes to the shared ledger.

    Without this, anyone with the link can freeze bets into the record the
    app exists to measure. Set `owner_code` in Streamlit Secrets to enable;
    unset, the app behaves as single-user and says so.
    """
    try:
        code = st.secrets.get("owner_code", "")
    except Exception:
        code = ""
    if not code:
        return True
    return st.session_state.get("owner_ok") is True


def save_tracker(df):
    # One choke point. Per-call-site checks get forgotten; this cannot be.
    if not is_owner():
        return df
    x = df.copy()
    for c in TRACKER_COLS:
        if c not in x.columns:
            x[c] = None
    x = x[TRACKER_COLS]
    # Dedup on the FULL key, which includes tier. Official and watch are
    # independent ledgers on purpose.
    if not x.empty:
        x = x[~x["record_key"].astype(str).duplicated(keep="first")]
    st.session_state["tracker"] = x
    ws = _sheet()
    if ws is not None:
        try:
            # DO NOT clear() then update(). If the clear lands and the update
            # fails — rate limit, network blip, oversized payload — the whole
            # ledger is gone and the session copy dies on the next restart.
            # Overwrite in place, then trim any surplus rows: there is no
            # moment at which the sheet is empty.
            body = [TRACKER_COLS] + x.fillna("").astype(str).values.tolist()
            try:
                old_rows = len(ws.get_all_values())
            except Exception:
                old_rows = 0
            ws.update(body)
            if old_rows > len(body):
                try:
                    ws.delete_rows(len(body) + 1, old_rows)
                except Exception:
                    # Stale trailing rows are visible and recoverable; an
                    # empty sheet is neither. Leave them.
                    pass
            st.session_state["last_save_error"] = ""
        except Exception as e:
            st.session_state["last_save_error"] = str(e)[:300]
            st.warning(f"Sheet write failed, kept in session only: {e}")
    return x


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=60 * 60)
def fetch_wind(home_team, kickoff_iso):
    """
    Forecast wind at the stadium for the hour of kickoff. Open-Meteo is free
    and needs no key. Returns mph, or None for domes and failures.
    """
    st_info = STADIUM.get(str(home_team).upper())
    if not st_info:
        return None
    lat, lon, outdoor = st_info
    if not outdoor:
        return 0.0
    try:
        kt = pd.to_datetime(kickoff_iso)
        if pd.isna(kt):
            return None
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon,
                    "hourly": "wind_speed_10m",
                    "wind_speed_unit": "mph",
                    "start_date": kt.strftime("%Y-%m-%d"),
                    "end_date": kt.strftime("%Y-%m-%d"),
                    "timezone": "America/New_York"},
            timeout=12,
        )
        if r.status_code != 200:
            return None
        h = r.json().get("hourly", {})
        times = pd.to_datetime(pd.Series(h.get("time", [])))
        speeds = h.get("wind_speed_10m", [])
        if not len(times) or not speeds:
            return None
        i = int((times - kt.tz_localize(None)).abs().idxmin())
        return float(speeds[i])
    except Exception:
        return None


def wind_adjustment(mph):
    """Points to subtract from the market total, after the forecast haircut."""
    if mph is None or mph < WIND_MIN_MPH:
        return 0.0
    return WIND_PTS_PER_MPH * float(mph) * WIND_FORECAST_DISCOUNT


@st.cache_data(show_spinner=False, ttl=60 * 10)
def fetch_live_odds(_bust=0):
    """
    Live spreads and totals from every US book, kept as raw offers so the
    card can price each one. Returns (offers, credits_left, error).
    """
    # Streamlit secrets are case-sensitive, and ODDS_API_KEY is the more
    # natural thing to type. Accept any common spelling rather than making
    # the name the thing that breaks it.
    key = None
    try:
        _sec = st.secrets
        for _n in ("odds_api_key", "ODDS_API_KEY", "oddsApiKey",
                   "odds_api", "ODDS_API"):
            if _n in _sec:
                key = _sec[_n]
                break
        if key is None:
            raise KeyError("odds_api_key")
    except Exception as e:
        # Distinguish the three ways this fails, because they have different
        # fixes: no secrets at all, a malformed secrets file (which makes
        # EVERY lookup throw, not just this one), or the wrong key name.
        try:
            _names = list(st.secrets.keys())
        except Exception:
            return {}, None, ("secrets unreadable \u2014 the file is not valid "
                              f"TOML ({type(e).__name__}). Every value must be "
                              "quoted: odds_api_key = \"abc123\"")
        if not _names:
            return {}, None, "no secrets set for this app"
        return {}, None, (f"no odds_api_key found. Keys present: "
                          f"{', '.join(map(str, _names))}")
    key = str(key).strip()
    if not key:
        return {}, None, "odds_api_key is empty"
    try:
        r = requests.get(
            "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds",
            params={"apiKey": key, "regions": "us",
                    "markets": "spreads,totals,h2h",
                    "oddsFormat": "american"},
            timeout=20,
        )
        if r.status_code == 401:
            return {}, None, (
                f"the key was sent but rejected (401). It is {len(key)} "
                f"characters and starts {key[:4]}\u2026 \u2014 check it "
                f"against the-odds-api.com, and that the trial has not run out."
            )
        if r.status_code != 200:
            return {}, None, f"HTTP {r.status_code}: {r.text[:160]}"
        left = r.headers.get("x-requests-remaining")
        offers = {}
        for ev in r.json():
            h = ODDS_TEAM.get(ev.get("home_team"))
            a = ODDS_TEAM.get(ev.get("away_team"))
            if not h or not a:
                continue
            spreads, totals, mls = [], [], []
            for bk in ev.get("bookmakers", []):
                book = bk.get("title") or bk.get("key")
                for mk in bk.get("markets", []):
                    for o in mk.get("outcomes", []):
                        pt, pr = o.get("point"), o.get("price")
                        if mk.get("key") == "h2h" and pr is not None:
                            _t = ODDS_TEAM.get(o.get("name"))
                            if _t in (h, a):
                                mls.append((_t, float(pr), book))
                            continue
                        if pt is None or pr is None:
                            continue
                        if mk.get("key") == "spreads":
                            side = ODDS_TEAM.get(o.get("name"))
                            if side in (h, a):
                                spreads.append((side, float(pt), float(pr), book))
                        elif mk.get("key") == "totals":
                            nm = str(o.get("name", "")).upper()
                            if nm in ("OVER", "UNDER"):
                                totals.append((nm, float(pt), float(pr), book))
            offers[(a, h)] = {"spreads": spreads, "totals": totals,
                              "moneylines": mls,
                              "commence": ev.get("commence_time")}
        st.session_state["odds_pulled_at"] = datetime.now(timezone.utc)
        return offers, left, None
    except Exception as e:
        return {}, None, str(e)


def lookup_offers(offers, away, home):
    for a, h in ((away, home), (ODDS_ALT.get(away, away), home),
                 (away, ODDS_ALT.get(home, home))):
        if (a, h) in offers:
            return offers[(a, h)]
    return None


def _american_to_prob(o):
    o = float(o)
    return 100.0 / (o + 100.0) if o > 0 else -o / (-o + 100.0)


def _prob_to_american(p):
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return -100.0 * p / (1 - p) if p >= 0.5 else 100.0 * (1 - p) / p


def american_payout(odds, default=ASSUMED_PRICE):
    """Profit per unit staked on a win. Falls back to the assumed price when
    the stored odds are missing, which is what old -110 rows were."""
    try:
        o = float(odds)
        if not math.isfinite(o) or o == 0:
            raise ValueError
    except (TypeError, ValueError):
        o = float(default)
    return (100.0 / abs(o)) if o < 0 else (o / 100.0)


def consensus_offers(offers, price=ASSUMED_PRICE, label="consensus"):
    """
    One synthetic book: the median number across every book, both sides at
    `price`. This is the honest default when your own book is not in the
    feed -- your book posts roughly the consensus number, and pricing each
    pick at the single best number anywhere overstates every edge you log.
    """
    out = {}
    for k, v in (offers or {}).items():
        away, home = k
        sp = v.get("spreads", []) or []
        hp = [p for t, p, _, _ in sp if t == home]
        ap = [-p for t, p, _, _ in sp if t != home]
        pts = hp + ap                      # all stated from the home side
        spreads = []
        if pts:
            m = float(np.median(pts))
            spreads = [(home, m, float(price), label),
                       (away, -m, float(price), label)]
        tt = [p for _, p, _, _ in (v.get("totals", []) or [])]
        totals = []
        if tt:
            m = float(np.median(tt))
            totals = [("OVER", m, float(price), label),
                      ("UNDER", m, float(price), label)]
        mls = []
        for team in (home, away):
            ps = [_american_to_prob(pr) for t, pr, *_ in
                  (v.get("moneylines", []) or []) if t == team]
            if ps:
                mls.append((team, round(_prob_to_american(np.median(ps))),
                            label))
        out[k] = {"spreads": spreads, "totals": totals, "moneylines": mls,
                  "commence": v.get("commence")}
    return out


def filter_offers(offers, book):
    """Keep one book's offers, moneylines included."""
    return {
        k: {"spreads": [r for r in v.get("spreads", []) if r[3] == book],
            "totals": [r for r in v.get("totals", []) if r[3] == book],
            "moneylines": [r for r in v.get("moneylines", [])
                           if len(r) > 2 and r[2] == book],
            "commence": v.get("commence")}
        for k, v in (offers or {}).items()
    }


def kickoff_utc(kickoff, commence=None):
    """
    Kickoff as a UTC timestamp. nflverse gameday/gametime are EASTERN local
    times with no zone attached; parsing them with utc=True (as the old code
    did) put every kickoff 4-5 hours early. The odds feed's commence_time is
    true UTC, so it wins when we have it.
    """
    if commence:
        t = pd.to_datetime(commence, errors="coerce", utc=True)
        if pd.notna(t):
            return t
    t = pd.to_datetime(kickoff, errors="coerce")
    if pd.isna(t):
        return pd.NaT
    if t.tzinfo is None:
        try:
            t = t.tz_localize("America/New_York")
        except Exception:
            return pd.NaT
    return t.tz_convert("UTC")


def best_offer(cands, fair, sd):
    """
    cands: (tag, threshold, price, book). The TAG is returned with the
    winner — re-deriving which side won from the threshold is ambiguous,
    because KC -2.5 and DEN +2.5 both reduce to a threshold of +2.5, so the
    lookup always matched the home offer and mislabelled away picks.

    Line shopping done properly: price every book's actual point AND price,
    then take the highest EV. Best number and best price are often at
    different books, so picking on either one alone leaves money behind.
    """
    best = None
    for tag, side, thresh, price, book in cands:
        edge = (fair - thresh) if side == "OVER_LIKE" else (thresh - fair)
        p = norm_cdf(edge / sd)
        e = ev_from_prob(p, price)
        if e is None:
            continue
        if best is None or e > best["ev"]:
            best = {"ev": e, "cover": p, "edge": edge, "point": thresh,
                    "price": price, "book": book, "tag": tag}
    return best


@st.cache_data(show_spinner=False, ttl=60 * 30)
def load_schedules(seasons, _bust=0):
    """_bust is unused, but changing it forces a fresh pull past the cache."""
    st.session_state["lines_pulled_at"] = datetime.now(timezone.utc)
    df = nfl.import_schedules(list(seasons))
    keep = ["game_id", "season", "week", "gameday", "gametime",
            "home_team", "away_team", "home_score", "away_score",
            "spread_line", "total_line"]
    return df[[c for c in keep if c in df.columns]].copy()


def line_sign(g):
    """nflverse states spread_line from the home side. Verify, never assume:
    a flipped sign would invert every pick silently."""
    d = g.dropna(subset=["spread_line", "home_score", "away_score"])
    if len(d) < 50:
        return 1.0
    m = d["home_score"] - d["away_score"]
    return 1.0 if np.corrcoef(d["spread_line"], m)[0, 1] > 0 else -1.0


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_qb_adjustment():
    """
    The measured cost of a team missing its usual quarterback, in points.

    Produced by nfl_qb_adjustment.py against historical games and committed
    as qb_adjustment.json. Absent the file there is NO adjustment — the app
    will not invent a number for something worth 5-7 points.
    """
    try:
        with open("qb_adjustment.json") as fh:
            d = json.load(fh)
        pts = float(d.get("penalty_points", 0.0))
        if not (math.isfinite(pts) and 0.0 < pts < 15.0):
            return None
        return d
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def qb_status(season, week):
    """
    Which teams are starting someone other than their usual quarterback.

    Depth charts give the listed starter; play-by-play gives who has actually
    been taking the snaps this season. A mismatch is the flag.

    Returns {team: {"expected": id, "usual": id, "changed": bool}}.
    """
    try:
        import nfl_data_py as nfl
        dc = nfl.import_depth_charts([int(season)])
    except Exception:
        return {}
    try:
        dc = dc[(dc["position"].astype(str).str.upper() == "QB")]
        if "depth_team" in dc.columns:
            dc = dc[pd.to_numeric(dc["depth_team"], errors="coerce") == 1]
        wk = pd.to_numeric(dc.get("week"), errors="coerce")
        cur = dc[wk == int(week)] if wk.notna().any() else dc
        if cur.empty:
            cur = dc[wk == wk.max()] if wk.notna().any() else dc
        tcol = "club_code" if "club_code" in cur.columns else "team"
        idcol = ("gsis_id" if "gsis_id" in cur.columns
                 else ("player_id" if "player_id" in cur.columns else None))
        if idcol is None:
            return {}
        listed = (cur.dropna(subset=[idcol])
                     .drop_duplicates(tcol, keep="first")
                     .set_index(tcol)[idcol].astype(str).to_dict())
    except Exception:
        return {}

    # Who has actually been starting, from earlier weeks this season.
    try:
        pbp = nfl.import_pbp_data([int(season)], downcast=True, cache=False)
        pbp = pbp[pd.to_numeric(pbp["week"], errors="coerce") < int(week)]
        pbp = pbp.dropna(subset=["passer_player_id", "posteam"])
        usual = (pbp.groupby(["posteam", "passer_player_id"]).size()
                    .rename("n").reset_index()
                    .sort_values("n", ascending=False)
                    .drop_duplicates("posteam", keep="first")
                    .set_index("posteam")["passer_player_id"].astype(str).to_dict())
    except Exception:
        usual = {}

    # How many snaps has each passer actually taken? Needed to tell an
    # upgrade from a downgrade.
    try:
        counts = (pbp.groupby("passer_player_id").size()
                     .rename("n").to_dict())
    except Exception:
        counts = {}

    out = {}
    for team, exp_id in listed.items():
        u = usual.get(str(team))
        changed = bool(u is not None and str(exp_id) != str(u))
        # DIRECTION MATTERS, and the first version of this got it backwards.
        #
        # A flat "starter changed" penalty treats a returning franchise
        # quarterback exactly like a third-stringer: both differ from whoever
        # has been taking the snaps, so both were penalised. Atlanta getting
        # Penix back would have had four points subtracted for it.
        #
        # Without per-quarterback values we cannot price an upgrade, so we do
        # not try. Penalise only a clear downgrade — the listed starter has
        # taken materially fewer snaps than the man he is replacing — and for
        # anything else flag it for display and adjust nothing.
        direction = "none"
        if changed:
            n_exp = float(counts.get(str(exp_id), 0) or 0)
            n_usu = float(counts.get(str(u), 0) or 0)
            if n_exp < 0.5 * n_usu:
                direction = "downgrade"
            elif n_exp > n_usu:
                direction = "upgrade"
            else:
                direction = "unclear"
        out[str(team)] = {
            "expected": str(exp_id),
            "usual": u,
            "changed": changed,
            "direction": direction,
            # Only a downgrade moves the number.
            "penalise": direction == "downgrade",
        }
    return out


def fit_ratings(hist, teams, target, symmetric=False, min_games=40, alpha=None):
    """
    Home field is estimated OUTSIDE the ridge, and team ratings are rescaled
    so predicted margins have the same spread as real ones.

    Ridge penalises every coefficient, but a team appears in only 10-20 games
    of the window while the home-field column appears in all of them. So team
    ratings were shrunk hard and home field barely at all, which pulled every
    prediction toward "home by 2.66". Where the market already had the home
    side favoured that barely showed; where it had them as a dog the model
    disagreed loudly — and the card filled with home underdogs.
    """
    if len(hist) < min_games:
        return None, None
    # Shrink harder on smaller samples. RIDGE_ALPHA was chosen for a full
    # 320-game window; applied unchanged to 32 games it would let two results
    # per team move a rating several points. Scaling the penalty by how much
    # less data there is keeps a small in-season fit honest instead of
    # excluding it entirely.
    _alpha = float(alpha) if alpha is not None else \
        RIDGE_ALPHA * max(1.0, WINDOW_GAMES / max(len(hist), 1))
    y = np.asarray(hist[target].values, dtype=float)

    # Constant term first, unpenalised: the league-average home margin (or
    # average total). What is left is what the teams have to explain.
    base = float(np.mean(y))
    y0 = y - base

    idx = {t: i for i, t in enumerate(teams)}
    X = np.zeros((len(hist), len(teams)))
    h, a = hist["home_team"].values, hist["away_team"].values
    for r in range(len(hist)):
        if h[r] in idx:
            X[r, idx[h[r]]] = 1.0
        if a[r] in idx:
            X[r, idx[a[r]]] = 1.0 if symmetric else -1.0

    m = Ridge(alpha=_alpha, fit_intercept=False).fit(X, y0)
    fitted = X @ m.coef_

    # Undo the compression: scale ratings so the spread of predicted margins
    # matches the spread actually explainable, rather than sitting flat near
    # the constant. Capped so a thin window cannot blow the ratings up.
    sd_fit = float(np.std(fitted))
    scale = 1.0
    if sd_fit > 1e-6:
        target_sd = float(np.std(y0)) * float(np.corrcoef(fitted, y0)[0, 1])
        if np.isfinite(target_sd) and target_sd > 0:
            scale = min(max(target_sd / sd_fit, 1.0), 2.5)

    return {t: float(m.coef_[idx[t]] * scale) for t in teams}, base


def build_ratings(sched, season, week):
    """Ratings from games played strictly before the target week."""
    g = sched.dropna(subset=["home_score", "away_score"]).copy()
    g["home_margin"] = g["home_score"] - g["away_score"]
    g["total_points"] = g["home_score"] + g["away_score"]
    prior = g[(g["season"] < season) | ((g["season"] == season) & (g["week"] < week))]
    prior = prior.sort_values(["season", "week"])
    if len(prior) < 40:
        return None
    # Take out the part of each past result that was down to injuries, so the
    # ratings measure healthy strength. This week's report is then applied in
    # full, which makes the adjustment incremental: only what is missing
    # relative to what the ratings already assume moves the line.
    fit = prior.tail(WINDOW_GAMES)
    _om, _ot = injury_history_offsets(fit)
    prior = prior.copy()
    # Scores arrive as integers; the adjusted results are not.
    prior["home_margin"] = prior["home_margin"].astype(float)
    prior["total_points"] = prior["total_points"].astype(float)
    prior.loc[fit.index, "home_margin"] = fit["home_margin"] - _om
    prior.loc[fit.index, "total_points"] = fit["total_points"] - _ot
    n_inj_adj = int((_om.abs() >= 0.1).sum())
    recent = prior.tail(WINDOW_GAMES)
    in_season = prior[prior["season"] == season]
    teams = sorted(set(g["home_team"]) | set(g["away_team"]))

    r_all, hfa = fit_ratings(recent, teams, "home_margin")
    if r_all is None:
        return None
    t_all, tbase = fit_ratings(recent, teams, "total_points", symmetric=True)

    # The in-season fit used the same 40-game floor as the full window, so
    # through week 3 (32 games) it returned None and the weight collapsed to
    # zero — the model was running entirely on last season while showing
    # "32 of them this season". Lower the floor and let the WEIGHT, which
    # already scales with sample size, do the work.
    r_cur, _ = fit_ratings(in_season, teams, "home_margin", min_games=16)
    t_cur, _ = fit_ratings(in_season, teams, "total_points", symmetric=True,
                           min_games=16)
    w = min(1.0, len(in_season) / 160.0) if r_cur is not None else 0.0

    def _blend(cur, allr):
        """Same treatment for both markets. Team ratings are deviations
        from league average, so carryover scales them toward average;
        the constant (home field, base total) is not scaled."""
        if allr is None:
            return None
        if cur is None:
            return {t: CARRYOVER * allr.get(t, 0.0) for t in teams}
        return {t: w * cur.get(t, 0.0) + (1 - w) * CARRYOVER * allr.get(t, 0.0)
                for t in teams}

    return {"margin": _blend(r_cur, r_all), "hfa": hfa,
            "total": _blend(t_cur, t_all), "tbase": tbase,
            "n_prior": len(prior), "in_season_weight": w,
            "n_in_season": len(in_season), "prior_ratings": r_all,
            "n_injury_adjusted": n_inj_adj}


def ev_from_prob(p, odds=-110):
    try:
        p = float(p); o = float(odds)
        if not (math.isfinite(p) and math.isfinite(o) and o != 0):
            return None
    except Exception:
        return None
    payout = (100.0 / abs(o)) if o < 0 else (o / 100.0)
    return p * payout - (1.0 - p)


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def qb_delta(season, week, h, a):
    """Points added to the home margin for a quarterback downgrade on either
    side. Zero when the QB measurement file is absent. Returns (delta, note)."""
    adj = load_qb_adjustment()
    if not adj or qb_model_on():
        return 0.0, None
    pen = float(adj["penalty_points"])
    stat = qb_status(season, week)
    h_out = bool(stat.get(str(h), {}).get("penalise"))
    a_out = bool(stat.get(str(a), {}).get("penalise"))
    d = (-pen if h_out else 0.0) + (pen if a_out else 0.0)
    who = [t for t, o in ((h, h_out), (a, a_out)) if o]
    return d, (f"backup QB: {', '.join(who)}" if who else None)


def _valid_injury_adj(d):
    try:
        # v2 added quarterbacks, v3 injured reserve, v4 locked snap shares
        # and individual QB ratings; older copies re-measure.
        return (isinstance(d, dict) and isinstance(d.get("margin"), dict)
                and "QB" in d["margin"] and int(d.get("version", 1)) >= 4
                and int(d.get("n_games", 0)) >= 500)
    except Exception:
        return False


def _injury_ws(create=False):
    """The 'injury_model' tab in the tracker spreadsheet, where the automatic
    measurement is kept so it survives the app going to sleep."""
    ws = _sheet()
    if ws is None:
        return None
    try:
        return ws.spreadsheet.worksheet("injury_model")
    except Exception:
        if not create:
            return None
        try:
            return ws.spreadsheet.add_worksheet("injury_model", rows=5, cols=2)
        except Exception:
            return None


def _read_saved_injury_adj():
    ws = _injury_ws()
    if ws is None:
        return None
    try:
        d = json.loads(ws.acell("A1").value or "")
    except Exception:
        return None
    return d if _valid_injury_adj(d) else None


def _save_injury_adj(d):
    ws = _injury_ws(create=True)
    if ws is None:
        return False
    try:
        ws.update_acell("A1", json.dumps(d))
        return True
    except Exception:
        return False


@st.cache_data(ttl=86400, show_spinner=(
    "Measuring what injuries are worth from 13 seasons of history "
    "\u2014 this happens about once a week and takes a few minutes..."))
def _auto_injury_adjustment(_stamp=0):
    """
    Keeps the injury measurement current without anyone running anything.

    Uses the saved copy in the Google Sheet if it is less than
    INJ_REFRESH_DAYS old; otherwise re-measures from nflverse and saves it.
    Raises on failure so a failure is never cached as "no adjustment".
    """
    saved = _read_saved_injury_adj()
    if saved:
        try:
            age = (datetime.now(timezone.utc)
                   - datetime.fromisoformat(saved["created_at"])).days
        except Exception:
            age = INJ_REFRESH_DAYS
        # A copy whose QB ratings failed is retried the next day, not held
        # for a week: that failure is usually a download hiccup.
        limit = INJ_REFRESH_DAYS if saved.get("qb") else 1
        if age < limit:
            return saved
    res = inj_mod.measure(nfl=nfl, log=lambda m: None)
    if not _valid_injury_adj(res):
        raise RuntimeError("measurement came back incomplete")
    res["saved_to_sheet"] = True
    if not _save_injury_adj(res):
        res["saved_to_sheet"] = False
    return res


def load_injury_adjustment():
    """
    What non-QB injuries are worth. In order of preference:
      1. injury_adjustment.json in the repo, if you ever want to pin one;
      2. the automatic measurement (refreshed weekly, kept in the Sheet).
    If measuring fails, there is NO adjustment and the app says so; it does
    not retry on every click, because each attempt takes minutes.
    """
    try:
        with open("injury_adjustment.json") as fh:
            d = json.load(fh)
        if _valid_injury_adj(d):
            d.setdefault("source", "repo file")
            return d
    except Exception:
        pass
    if st.session_state.get("inj_auto_failed"):
        return None
    try:
        d = _auto_injury_adjustment(st.session_state.get("inj_stamp", 0))
        d = dict(d)
        d.setdefault("source", "automatic")
        return d
    except Exception as e:
        st.session_state["inj_auto_failed"] = f"{type(e).__name__}: {e}"
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def _inj_report(season, schema=None):
    """Injury report plus players on reserve, for one season."""
    rep = inj_mod.prep_injuries(nfl.import_injuries([int(season)]))
    try:
        res = inj_mod.prep_reserve(nfl.import_weekly_rosters([int(season)]))
    except Exception:
        res = inj_mod.prep_reserve(None)
    # The weekly roster for the newest week can lag the news (a Thursday IR
    # move). If this week's report exists but its roster week does not,
    # take reserve status from the season roster, which is current.
    try:
        wk_rep = int(rep["week"].max()) if len(rep) else None
        wk_res = int(res["week"].max()) if len(res) else None
        if wk_rep is not None and (wk_res is None or wk_res < wk_rep):
            cur = nfl.import_seasonal_rosters([int(season)]).copy()
            cur["week"] = wk_rep
            res = pd.concat([res, inj_mod.prep_reserve(cur)],
                            ignore_index=True)
    except Exception:
        pass
    return inj_mod.combine_reports(rep, res)


@st.cache_data(ttl=1800, show_spinner=False)
def _inj_snaps(season):
    """Snap counts for this season and last; last alone if this season's
    file does not exist yet."""
    try:
        return inj_mod.prep_snaps(
            nfl.import_snap_counts([int(season) - 1, int(season)]))
    except Exception:
        return inj_mod.prep_snaps(nfl.import_snap_counts([int(season) - 1]))


@st.cache_data(ttl=1800, show_spinner=False)
def injury_loads(season, week, schema=None):
    """
    This week's injury load per team and position group. Returns
    (loads, status) where status explains an empty result, because "no
    injuries" and "no report published yet" must not look the same.
    """
    adj = load_injury_adjustment()
    if not adj:
        return {}, "off"
    try:
        rep = _inj_report(season, INJ_SCHEMA)
    except Exception as e:
        return {}, f"injury report unavailable ({type(e).__name__})"
    if rep.empty or not ((rep["week"] == int(week)).any()):
        return {}, f"no injury report published for week {week} yet"
    try:
        sn = _inj_snaps(season)
    except Exception as e:
        return {}, f"snap counts unavailable ({type(e).__name__})"
    loads = inj_mod.team_injury_loads(rep, sn, int(season), int(week),
                                      adj.get("miss_prob"))
    return loads, "ok"


@st.cache_data(ttl=1800, show_spinner=False)
def season_injury_table(season, miss_prob_items, schema=None):
    """Injury load for every team-week of a season, for the games the ratings
    are fit on. Empty (no offset) if the data is not available."""
    try:
        return inj_mod.season_loads(_inj_report(season, INJ_SCHEMA), _inj_snaps(season),
                                    int(season), dict(miss_prob_items))
    except Exception:
        return pd.DataFrame(columns=["season", "week", "team",
                                     *inj_mod.GROUPS])


# ----------------------------------------------------------------------
# Quarterback ratings, live
# ----------------------------------------------------------------------
def qb_params():
    """The measured QB model, or None if it is not measured or did not clear
    the bar (then the injury model's flat QB penalty is used instead)."""
    adj = load_injury_adjustment()
    q = (adj or {}).get("qb")
    return q if (q and qb_usable(q, "margin")) else None


def qb_model_on():
    return qb_params() is not None


def _skip_inj_qb():
    """The injury model's flat QB group steps aside when something better
    prices quarterbacks, so no absence is charged twice."""
    return bool(load_qb_adjustment()) or qb_model_on()


@st.cache_data(ttl=7 * 86400, show_spinner=False)
def _qb_pbp_past(season):
    return prep_pbp(load_pbp(nfl, [int(season)]))


@st.cache_data(ttl=1800, show_spinner=False)
def _qb_pbp_current(season):
    return prep_pbp(load_pbp(nfl, [int(season)]))


def _qb_db(season):
    this_year = datetime.now().year
    frames = []
    for s_ in range(int(season) - QB_LOOKBACK, int(season) + 1):
        try:
            frames.append(_qb_pbp_current(s_) if s_ >= this_year - 1
                          else _qb_pbp_past(s_))
        except Exception:
            continue
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else prep_pbp(None)


@st.cache_data(ttl=7 * 86400, show_spinner=False)
def _qb_draft():
    try:
        return prep_draft(nfl.import_draft_picks())
    except Exception:
        return {}


@st.cache_data(ttl=1800, show_spinner=False)
def qb_ratings_now(season, week, schema=None):
    """Every QB's pregame rating for this week, from plays before it."""
    q = qb_params()
    if not q:
        return pd.DataFrame()
    return qb_ratings(_qb_db(season), int(season), int(week), _qb_draft(), q)


@st.cache_data(ttl=1800, show_spinner=False)
def _sched_one(season):
    try:
        return nfl.import_schedules([int(season)])
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner=False)
def depth_qbs(season, week, schema=None):
    """
    {team: [(gsis_id, full name), ...]} quarterbacks in depth-chart order for
    this week. Handles both nflverse layouts: the weekly one (club_code,
    depth_team, week) and the daily snapshots used since 2025 (team,
    pos_rank, dt).
    """
    try:
        dc = nfl.import_depth_charts([int(season)])
    except Exception:
        return {}
    if dc is None or len(dc) == 0:
        return {}
    c = dc.columns
    pcol = "position" if "position" in c else ("pos_abb" if "pos_abb" in c else None)
    rcol = "depth_team" if "depth_team" in c else ("pos_rank" if "pos_rank" in c else None)
    tcol = "club_code" if "club_code" in c else ("team" if "team" in c else None)
    ncol = "full_name" if "full_name" in c else ("player_name" if "player_name" in c else None)
    if not all([pcol, rcol, tcol, ncol]):
        return {}
    d = dc[dc[pcol].astype(str).str.upper() == "QB"].copy()
    d["_rank"] = pd.to_numeric(d[rcol], errors="coerce")
    if "week" in c and pd.to_numeric(d["week"], errors="coerce").notna().any():
        wk = pd.to_numeric(d["week"], errors="coerce")
        cur = d[wk == int(week)]
        if cur.empty:
            prior = wk[wk <= int(week)]
            cur = d[wk == prior.max()] if len(prior) else d.iloc[0:0]
    elif "dt" in c:
        # Daily snapshots: the latest one before this week's games.
        sch = _sched_one(season)
        cutoff = None
        try:
            gd = pd.to_datetime(sch.loc[pd.to_numeric(sch["week"]) == int(week),
                                        "gameday"], errors="coerce")
            cutoff = gd.max() + pd.Timedelta(days=1)
        except Exception:
            pass
        dt = pd.to_datetime(d["dt"], errors="coerce", utc=True).dt.tz_localize(None)
        d = d.assign(_dt=dt)
        if cutoff is not None and pd.notna(cutoff):
            d = d[d["_dt"] <= cutoff]
        if d.empty:
            return {}
        last = d.groupby(tcol)["_dt"].transform("max")
        cur = d[d["_dt"] == last]
    else:
        cur = d
    out = {}
    for team, g in cur.sort_values("_rank").groupby(tcol):
        seen, lst = set(), []
        for r in g.itertuples():
            nm = str(getattr(r, ncol))
            gid = str(getattr(r, "gsis_id", "") or "")
            k_ = name_key(nm)
            if k_ in seen:
                continue
            seen.add(k_)
            lst.append((gid, nm))
        out[canon(team)] = lst
    return out


def _short_key(full):
    """'Jaxson Dart' -> 'jdart', to match play-by-play's 'J.Dart'."""
    toks = [t for t in re.split(r"[\s.]+", str(full))
            if t and t.lower().strip(".") not in {"jr", "sr", "ii", "iii", "iv", "v"}]
    if len(toks) < 2:
        return name_key(full)
    return name_key(toks[0][0] + toks[-1])


def _unavailable(season, week, team):
    """Name keys of this team's players ruled out, doubtful or on reserve."""
    try:
        rep = _inj_report(season, INJ_SCHEMA)
    except Exception:
        return set()
    r = rep[(rep["season"] == int(season)) & (rep["week"] == int(week))
            & (rep["team"] == canon(team))
            & rep["status"].isin(["out", "doubtful", "reserve"])]
    return set(r["key"])


def expected_qbs(season, week, team):
    """
    Who is expected to start this week, and who the team's usual starter is.

    Starter: the depth chart's top QB who is not out, doubtful or on reserve.
    Without a depth chart: the QB with the most dropbacks this season who is
    available. If nobody known is available, a replacement-level backup.
    """
    q = qb_params()
    if not q:
        return None
    R = qb_ratings_now(season, week, INJ_SCHEMA)
    db = _qb_db(season)
    team = canon(team)
    out_keys = _unavailable(season, week, team)

    def _find(gid, nm):
        if gid and gid in R.index:
            return gid
        sk = _short_key(nm)
        hit = [i for i, n in zip(R.index, R["name"]) if name_key(n) == sk] \
            if len(R) else []
        return hit[0] if hit else None

    def _rate(qid):
        if qid is not None and qid in R.index:
            return float(R.loc[qid, "rating"])
        return qb_rating_of("late", q)

    # Usual starter: most dropbacks for this team, this season before this
    # week, else last season.
    usual_id, usual_name = None, None
    for s_ in (int(season), int(season) - 1):
        d = db[(db["team"] == team) & (db["season"] == s_)
               & ((db["season"] < int(season)) | (db["week"] < int(week)))]
        if len(d):
            top = d.groupby("qb_id").size().idxmax()
            usual_id = top
            usual_name = str(d.loc[d["qb_id"] == top, "qb_name"].iloc[-1])
            break

    starter_id, starter_name, source = None, None, None
    for gid, nm in depth_qbs(season, week, INJ_SCHEMA).get(team, []):
        if name_key(nm) in out_keys:
            continue
        starter_id, starter_name, source = _find(gid, nm), nm, "depth chart"
        break
    if starter_name is None:
        # No usable depth chart: most-used QB this season who is available.
        d = db[(db["team"] == team) & (db["season"] == int(season))
               & (db["week"] < int(week))]
        order = (d.groupby(["qb_id", "qb_name"]).size()
                   .sort_values(ascending=False).reset_index())
        for r in order.itertuples():
            if not any(k_ in out_keys and _short_key_match(k_, r.qb_name)
                       for k_ in out_keys):
                starter_id, starter_name, source = r.qb_id, r.qb_name, "usage"
                break
    if starter_name is None:
        starter_name, source = "unknown backup", "replacement level"

    return {"team": team, "starter_id": starter_id, "starter": starter_name,
            "rating": _rate(starter_id), "source": source,
            "usual_id": usual_id, "usual": usual_name,
            "usual_rating": _rate(usual_id) if usual_id else None,
            "changed": bool(usual_id and starter_id != usual_id)}


def _short_key_match(full_key, pbp_name):
    """Does a full-name key ('jaxsondart') belong to 'J.Dart'?"""
    pk = name_key(pbp_name)
    return len(pk) > 1 and full_key.endswith(pk[1:]) and full_key[0] == pk[0]


def _qb_last(nm):
    """'J.Dart' / 'Jaxson Dart' / 'Kenneth Walker III' -> surname."""
    t = [x for x in re.split(r"[\s.]+", str(nm))
         if x and x.lower() not in {"jr", "sr", "ii", "iii", "iv", "v"}]
    return t[-1] if t else str(nm)


def qb_game_delta(season, week, h, a):
    """(margin_delta, total_delta, note, info) from the two starters.
    The power ratings are fit with every past QB taken out, so this adds the
    actual starters back in: an unchanged QB nets to roughly nothing, a
    change moves the line by the gap between the two men."""
    q = qb_params()
    if not q:
        return 0.0, 0.0, None, {}
    eh, ea = expected_qbs(season, week, h), expected_qbs(season, week, a)
    if not eh or not ea:
        return 0.0, 0.0, None, {}
    k = float(q["margin"]["coef"])
    kt = float(q["total"]["coef"]) if qb_usable(q, "total") else 0.0
    ref = float(q["ref"])
    dm = float(np.clip(k * (eh["rating"] - ea["rating"]), -QB_CAP_PTS, QB_CAP_PTS))
    dt = float(np.clip(kt * (eh["rating"] + ea["rating"] - 2 * ref),
                       -QB_CAP_PTS, QB_CAP_PTS))
    notes = []
    for e_ in (ea, eh):
        if e_["changed"] and e_.get("usual_rating") is not None:
            e_["change_pts"] = k * (e_["rating"] - e_["usual_rating"])
            notes.append(f"{e_['team']}: {_qb_last(e_['starter'])} for "
                         f"{_qb_last(e_['usual'])} ({e_['change_pts']:+.1f})")
    return dm, dt, ("; ".join(notes) or None), {"home": eh, "away": ea}


@st.cache_data(ttl=1800, show_spinner=False)
def qb_history_table(season, schema=None):
    """Pregame rating of each past starter this season, for the fit."""
    q = qb_params()
    if not q:
        return pd.DataFrame(columns=["season", "week", "team", "qb_id", "rating"])
    try:
        return starter_ratings_table(_qb_db(season), [int(season)],
                                     _qb_draft(), q)
    except Exception:
        return pd.DataFrame(columns=["season", "week", "team", "qb_id", "rating"])


def qb_history_offsets(games):
    """Points of each past result that came from who played quarterback."""
    z = pd.Series(0.0, index=games.index)
    q = qb_params()
    if not q or games.empty:
        return z, z.copy()
    ref = float(q["ref"])
    k = float(q["margin"]["coef"])
    kt = float(q["total"]["coef"]) if qb_usable(q, "total") else 0.0
    T = pd.concat([qb_history_table(int(s_), INJ_SCHEMA) for s_ in
                   sorted(pd.to_numeric(games["season"]).unique())],
                  ignore_index=True)
    if T.empty:
        return z, z.copy()
    T = T.drop_duplicates(["season", "week", "team"])
    key = T.set_index(["season", "week", "team"])["rating"]
    def _r(side):
        idx = list(zip(pd.to_numeric(games["season"]).astype(int),
                       pd.to_numeric(games["week"]).astype(int),
                       games[f"{side}_team"].map(canon)))
        return pd.Series([key.get(i, ref) for i in idx], index=games.index)
    rh, ra = _r("home"), _r("away")
    return ((k * (rh - ra)).clip(-QB_CAP_PTS, QB_CAP_PTS),
            (kt * (rh + ra - 2 * ref)).clip(-QB_CAP_PTS, QB_CAP_PTS))


def injury_history_offsets(games):
    """(margin_offset, total_offset) for past games: injuries plus who played
    quarterback. Zeros when the adjustment is off."""
    adj = load_injury_adjustment()
    z = pd.Series(0.0, index=games.index)
    if not adj or games.empty:
        return z, z.copy()
    mp = tuple(sorted((adj.get("miss_prob") or {}).items()))
    tables = [season_injury_table(int(s_), mp, INJ_SCHEMA)
              for s_ in sorted(pd.to_numeric(games["season"]).unique())]
    table = pd.concat(tables, ignore_index=True) if tables else None
    om, ot = inj_mod.history_offsets(adj, table, games, min_t=INJ_MIN_T,
                                     cap=INJ_MAX_PTS, skip_qb=_skip_inj_qb())
    try:
        qm, qt = qb_history_offsets(games)
    except Exception:
        qm, qt = z, z.copy()
    return om + qm, ot + qt


def injury_delta(season, week, h, a):
    """(margin_delta, total_delta, note, detail) for one game: this week's
    injury report plus the two starting quarterbacks. All zero when the
    measurement is absent or the report is not out yet."""
    adj = load_injury_adjustment()
    if not adj:
        return 0.0, 0.0, None, {}
    loads, _ = injury_loads(season, week, INJ_SCHEMA)
    dm, dt, note, det = inj_mod.game_deltas(adj, loads, h, a, min_t=INJ_MIN_T,
                                            cap=INJ_MAX_PTS,
                                            skip_qb=_skip_inj_qb())
    try:
        qm, qt, qnote, qinfo = qb_game_delta(season, week, h, a)
    except Exception:
        qm, qt, qnote, qinfo = 0.0, 0.0, None, {}
    det = dict(det or {})
    if qinfo:
        det["qb"] = qinfo
    note = "; ".join(n for n in (qnote, note) if n) or None
    return dm + qm, dt + qt, note, det


def assign_tiers(card):
    """OFFICIAL = positive value at your price. WATCH = big disagreement that
    does not beat the vig. Everything else is on the board but not tracked."""
    if card.empty:
        card["bet_tier"] = pd.Series(dtype=object)
        return card
    gap = (pd.to_numeric(card["model_line"], errors="coerce")
           - pd.to_numeric(card["bet_line"], errors="coerce")).abs()
    ev = pd.to_numeric(card["expected_value"], errors="coerce")
    tier = np.where(ev >= MIN_EV, "OFFICIAL",
                    np.where(gap >= MIN_GAP_PTS, "WATCH", None))
    return card.assign(gap_pts=gap, bet_tier=tier)


def build_card(sched, season, week, sign, offers=None):
    rt = build_ratings(sched, season, week)
    if rt is None:
        return pd.DataFrame(), None

    # Quarterback adjustment. The rating is a ridge on final margins and has
    # no idea who is playing, so when a starter is out it keeps the value he
    # earned — and disagrees with a line that moved on the news. That is not
    # edge, it is the model not knowing something everyone else knows, and it
    # is large: the penalty is measured, not guessed, and if it has not been
    # measured yet nothing is applied.
    _qb_adj = load_qb_adjustment()
    _qb_pen = (float(_qb_adj["penalty_points"])
               if (_qb_adj and not qb_model_on()) else 0.0)
    _qb_stat = qb_status(season, week) if _qb_pen else {}

    games = sched[(sched["season"] == season) & (sched["week"] == week)].copy()
    rows = []
    _inj = {}
    for _, g in games.iterrows():
        h, a = g["home_team"], g["away_team"]
        if h not in rt["margin"] or a not in rt["margin"]:
            continue

        h_out = bool(_qb_stat.get(str(h), {}).get("penalise"))
        a_out = bool(_qb_stat.get(str(a), {}).get("penalise"))
        _qbd = (-_qb_pen if h_out else 0.0) + (_qb_pen if a_out else 0.0)
        # Injuries go into the RAW line, like the QB adjustment, so they pass
        # through the 0.099 blend. Their job is mostly to stop the model
        # disagreeing with a line that moved on news it cannot see.
        _im, _it, _inote, _ = injury_delta(season, week, h, a)
        _inj[g["game_id"]] = (_im, _it, _inote)
        raw_model = (rt["margin"][h] - rt["margin"][a] + rt["hfa"]) + _qbd + _im
        mkt = sign * g["spread_line"] if pd.notna(g.get("spread_line")) else None
        live = lookup_offers(offers, a, h) if offers else None

        # SPREAD — live books first, nflverse as fallback
        if live and live["spreads"]:
            pts = [-p for t, p, _, _ in live["spreads"] if t == h]
            if pts:
                mkt = float(np.median(pts))
            fair = mkt + MODEL_WEIGHT * (raw_model - mkt)
            # Home side covers above -point; away side covers below +point.
            cands = [(("HOME" if t == h else "AWAY"),
                      ("OVER_LIKE" if t == h else "UNDER_LIKE"),
                      (-p if t == h else p), pr, bk)
                     for t, p, pr, bk in live["spreads"]]
            b = best_offer(cands, fair, SD_MARGIN)
            if b:
                side = b["tag"]
                team = h if side == "HOME" else a
                shown = -b["point"] if side == "HOME" else b["point"]
                rows.append({
                    "game_id": g["game_id"], "season": season, "week": week,
                    "kickoff": f"{g.get('gameday','')} {g.get('gametime','')}".strip(),
                    "matchup": f"{a} @ {h}", "home_team": h, "away_team": a,
                    "market_type": "SPREAD", "pick_side": side,
                    "pick_label": f"{team} {shown:+g} ({b['price']:+.0f}) "
                                  f"@ {b['book']}",
                    "bet_line": float(b["point"]), "model_line": float(raw_model),
                    "edge_pts": float(b["edge"]), "cover_prob": b["cover"],
                    "expected_value": b["ev"], "odds": b["price"],
                })
                mkt = None  # handled

        if mkt is not None:
            # Blend toward the market at the weight the backtest earned.
            # Betting the raw model line means betting a number the data
            # says is worse than the one already on the board.
            fair = mkt + MODEL_WEIGHT * (raw_model - mkt)
            edge = fair - mkt
            side = "HOME" if edge > 0 else "AWAY"
            p = norm_cdf(abs(edge) / SD_MARGIN)
            line_for_side = -mkt if side == "HOME" else mkt
            rows.append({
                "game_id": g["game_id"], "season": season, "week": week,
                "kickoff": f"{g.get('gameday','')} {g.get('gametime','')}".strip(),
                "matchup": f"{a} @ {h}", "home_team": h, "away_team": a,
                "market_type": "SPREAD", "pick_side": side,
                "pick_label": f"{h if side=='HOME' else a} "
                              f"{line_for_side:+.1f}",
                "bet_line": float(mkt), "model_line": float(raw_model),
                "edge_pts": float(edge), "cover_prob": p,
                "expected_value": ev_from_prob(p, ASSUMED_PRICE),
                "odds": ASSUMED_PRICE,
            })

        # TOTAL — live books first
        if live and live["totals"] and rt["total"]:
            raw_total = (rt["total"].get(h, 0.0) + rt["total"].get(a, 0.0)
                         + rt["tbase"] + _it)
            pts = [p for _, p, _, _ in live["totals"]]
            mt = float(np.median(pts))
            # Wind adjusts the FAIR line directly rather than going through
            # the 0.099 blend. That blend is the discount for a power rating
            # that showed no edge; wind was measured against the closing line
            # and survived out of sample, so it is a different kind of claim
            # and takes its own (forecast-error) haircut instead.
            _mph = fetch_wind(h, f"{g.get('gameday','')} {g.get('gametime','')}")
            _wadj = wind_adjustment(_mph)
            fair_t = mt + MODEL_WEIGHT * (raw_total - mt) + _wadj
            cands = [(nm, ("OVER_LIKE" if nm == "OVER" else "UNDER_LIKE"),
                      p, pr, bk)
                     for nm, p, pr, bk in live["totals"]]
            b = best_offer(cands, fair_t, SD_TOTAL)
            if b:
                side = b["tag"]
                rows.append({
                    "game_id": g["game_id"], "season": season, "week": week,
                    "kickoff": f"{g.get('gameday','')} {g.get('gametime','')}".strip(),
                    "matchup": f"{a} @ {h}", "home_team": h, "away_team": a,
                    "market_type": "TOTAL", "pick_side": side,
                    "wind_mph": _mph, "wind_adj": _wadj,
                    "pick_label": f"{side.title()} {b['point']:g} "
                                  f"({b['price']:+.0f}) @ {b['book']}"
                                  + (f" \u00b7 {_mph:.0f}mph wind"
                                     if _wadj else ""),
                    "bet_line": float(b["point"]), "model_line": float(raw_total),
                    "edge_pts": float(b["edge"]), "cover_prob": b["cover"],
                    "expected_value": b["ev"], "odds": b["price"],
                })
                continue

        # TOTAL fallback
        if rt["total"] and pd.notna(g.get("total_line")):
            raw_total = (rt["total"].get(h, 0.0) + rt["total"].get(a, 0.0)
                         + rt["tbase"] + _it)
            mt = float(g["total_line"])
            _mph = fetch_wind(h, f"{g.get('gameday','')} {g.get('gametime','')}")
            _wadj = wind_adjustment(_mph)
            fair_t = mt + MODEL_WEIGHT * (raw_total - mt) + _wadj
            edge_t = fair_t - mt
            side = "OVER" if edge_t > 0 else "UNDER"
            p = norm_cdf(abs(edge_t) / SD_TOTAL)
            rows.append({
                "game_id": g["game_id"], "season": season, "week": week,
                "kickoff": f"{g.get('gameday','')} {g.get('gametime','')}".strip(),
                "matchup": f"{a} @ {h}", "home_team": h, "away_team": a,
                "market_type": "TOTAL", "pick_side": side,
                "wind_mph": _mph, "wind_adj": _wadj,
                "pick_label": f"{side.title()} {mt:g}"
                              + (f" \u00b7 {_mph:.0f}mph wind"
                                 if _wadj else ""),
                "bet_line": mt, "model_line": float(raw_total),
                "edge_pts": float(edge_t), "cover_prob": p,
                "expected_value": ev_from_prob(p, ASSUMED_PRICE),
                "odds": ASSUMED_PRICE,
            })

    card = pd.DataFrame(rows)
    if card.empty:
        return card, rt
    # What the injury adjustment did to each row's model line, for display.
    card["inj_adj"] = [(_inj.get(gid, (0.0, 0.0, None))[0] if m == "SPREAD"
                        else _inj.get(gid, (0.0, 0.0, None))[1])
                       for gid, m in zip(card["game_id"], card["market_type"])]
    card["inj_note"] = [_inj.get(gid, (0.0, 0.0, None))[2]
                        for gid in card["game_id"]]
    card["abs_edge"] = card["edge_pts"].abs()
    card = card.sort_values("abs_edge", ascending=False).reset_index(drop=True)
    # Tiers are FILTERS, not quotas. If nothing clears the floor the
    # section is empty, which is a legitimate answer for a week.
    # The card ranks by how far the model is from the line. EV is still
    # computed and shown on every row, but it is no longer a gate: a card
    # that says "no bets" most weeks does not answer the question this app
    # exists to answer, which is whether these picks land on the right side
    # more than 52.4% of the time. That gets settled by the record, not by
    # a threshold.
    # v1.2: the old code tagged EVERY row OFFICIAL here, so "Freeze" logged
    # the whole slate (~30 markets) while the card showed ~7. The tracker was
    # measuring a different set of bets than the one on screen.
    card = assign_tiers(card)
    # Qualifying rows first, then by disagreement.
    card["_tier_rank"] = card["bet_tier"].map(
        {"OFFICIAL": 0, "WATCH": 1}).fillna(2)
    card = card.sort_values(["_tier_rank", "abs_edge"],
                            ascending=[True, False]).drop(columns="_tier_rank")
    return card.reset_index(drop=True), rt


# ----------------------------------------------------------------------
# Freeze + grade
# ----------------------------------------------------------------------
def freeze(card, tracker):
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    existing = set(tracker["record_key"].astype(str)) if not tracker.empty else set()
    # Base keys already logged in EITHER ledger. Without this, a market frozen
    # as WATCH on Wednesday and OFFICIAL on Sunday is counted twice.
    existing_base = {k[:-2] if k.endswith("|W") else k for k in existing}
    card = card[card["bet_tier"].isin(["OFFICIAL", "WATCH"])]
    new = []
    for _, r in card.iterrows():
        base = f"{r['game_id']}|{r['market_type']}"
        key = base + ("" if r["bet_tier"] == "OFFICIAL" else "|W")
        if base in existing_base:
            continue
        row = {c: None for c in TRACKER_COLS}
        row.update({
            "record_key": key, "frozen_at": now,
            "season": r["season"], "week": r["week"], "game_id": r["game_id"],
            "kickoff": r["kickoff"], "matchup": r["matchup"],
            "home_team": r["home_team"], "away_team": r["away_team"],
            "market_type": r["market_type"], "pick_side": r["pick_side"],
            "pick_label": r["pick_label"], "bet_line": r["bet_line"],
            "model_line": r["model_line"], "edge_pts": r["edge_pts"],
            "cover_prob": r["cover_prob"], "expected_value": r["expected_value"],
            "odds": r["odds"], "bet_tier": r["bet_tier"],
            "model_version": model_version(), "status": "FROZEN",
        })
        new.append(row)
    if not new:
        return tracker, 0
    out = pd.concat([tracker, pd.DataFrame(new)], ignore_index=True)
    return save_tracker(out), len(new)


def capture_closing(tracker, offers, sched=None):
    """
    Record the market number just BEFORE kickoff, so CLV means something.

    v1.2 rewrite. The previous version only wrote rows whose kickoff had
    PASSED, and it parsed the nflverse kickoff (Eastern, no zone) as UTC --
    4-5 hours early. So a "close" was whatever the feed showed some time
    after 9am ET on a 1pm game: sometimes a pregame number, often a LIVE
    in-game line, because the Odds API keeps returning in-play odds for
    games in progress. The "clean, within 3 hours" bucket was mostly the
    in-game ones.

    Now:
      * Every pull before kickoff OVERWRITES the snapshot, so the stored
        close is the last pregame number the app saw.
      * Once kickoff passes the snapshot is frozen and never rewritten.
      * Games the feed already shows as started are ignored entirely.
      * The close is the consensus median across all books -- the market's
        number, not one book's shading.

    Limitation: the app only pulls when it is open. For a real close, open
    the Tracker in the hour before each kickoff window. The CLV panel only
    counts snapshots taken within 3 hours before kickoff as clean.
    """
    if tracker is None or tracker.empty or not offers:
        return tracker, 0
    df = tracker.copy()
    now = pd.Timestamp.now(tz="UTC")

    n = 0
    for idx in df.index:
        if str(df.at[idx, "status"]).upper() == "GRADED":
            continue
        off = lookup_offers(offers, str(df.at[idx, "away_team"]),
                            str(df.at[idx, "home_team"]))
        if not off:
            continue
        kick = kickoff_utc(df.at[idx, "kickoff"], off.get("commence"))
        if pd.isna(kick) or kick <= now:
            continue          # started: whatever is stored is final
        mt = str(df.at[idx, "market_type"]).upper()
        side = str(df.at[idx, "pick_side"]).upper()
        home = str(df.at[idx, "home_team"])
        pts = []
        if mt == "TOTAL":
            for _nm, pt, _pr, _bk in off.get("totals", []) or []:
                if math.isfinite(float(pt)):
                    pts.append(float(pt))
        else:
            # Book states a home favourite as -4.5; bet_line is nflverse
            # convention (+4.5). Negate so both are on the same scale. The
            # away side's point is already the home-favoured number.
            for team, pt, _pr, _bk in off.get("spreads", []) or []:
                if not math.isfinite(float(pt)):
                    continue
                pts.append(-float(pt) if str(team) == home else float(pt))
        if not pts:
            continue
        close = round(float(np.median(pts)), 2)

        try:
            bl = float(df.at[idx, "bet_line"])
        except (TypeError, ValueError):
            continue
        # Positive CLV = the number moved in the bet's favour.
        # HOME lays the number: took -3, closed -4.25 -> +1.25.
        # AWAY receives it: took +3, closed +4.25 -> -1.25.
        if mt == "TOTAL":
            clv = (close - bl) if side == "OVER" else (bl - close)
        elif side == "HOME":
            clv = close - bl
        else:
            clv = bl - close

        prev = pd.to_numeric(pd.Series([df.at[idx, "closing_line"]]),
                             errors="coerce").iloc[0]
        df.at[idx, "closing_line"] = close
        df.at[idx, "clv_points"] = round(float(clv), 2)
        df.at[idx, "closing_captured_at"] = now.isoformat(timespec="seconds")
        # Store kickoff in UTC too, so the lag can be measured correctly
        # later without re-deriving the zone.
        df.at[idx, "kickoff_utc"] = kick.isoformat()
        if pd.isna(prev) or float(prev) != close:
            n += 1
    # Only write to the sheet when a number actually moved; the timestamp
    # refresh alone rides along with the next real save.
    return (save_tracker(df), n) if n else (df, 0)


def clv_summary(df):
    """
    CLV with the dispersion that makes a mean readable, split by how long
    after kickoff the close was captured.

    A mean on its own says nothing: +0.36 across 66 bets was "significant"
    in the college app right up until the captures turned out to be hours
    stale. Only the near-kickoff bucket is a real measurement.
    """
    out = {"n": 0, "mean": float("nan"), "sd": float("nan"),
           "t": float("nan"), "beat": float("nan"), "zero": float("nan"),
           "n_clean": 0, "clean_mean": float("nan")}
    if df is None or df.empty or "clv_points" not in df.columns:
        return out
    c = pd.to_numeric(df["clv_points"], errors="coerce")
    ok = c.notna()
    if not ok.any():
        return out
    cc = c[ok]
    out["n"] = int(len(cc))
    out["mean"] = float(cc.mean())
    out["beat"] = float((cc > 0).mean())
    out["zero"] = float((cc == 0).mean())
    if len(cc) > 1 and cc.std(ddof=1) > 0:
        out["sd"] = float(cc.std(ddof=1))
        out["t"] = out["mean"] / (out["sd"] / math.sqrt(len(cc)))

    # Clean = snapshot taken in the 3 hours BEFORE kickoff. Kickoff comes
    # from the stored UTC time when present; otherwise the nflverse string
    # is read as Eastern (it used to be read as UTC, 4-5 hours off). Rows
    # captured after kickoff -- every pre-v1.2 capture -- are in-game lines
    # and never count.
    sub = df.loc[ok]
    ku = (sub["kickoff_utc"] if "kickoff_utc" in sub.columns
          else pd.Series(None, index=sub.index))
    kick = pd.Series([kickoff_utc(k, c if isinstance(c, str) and c else None)
                      for k, c in zip(sub["kickoff"], ku)], index=sub.index)
    kick = pd.to_datetime(kick, errors="coerce", utc=True)
    cap = pd.to_datetime(sub["closing_captured_at"], errors="coerce", utc=True)
    lead_h = (kick - cap).dt.total_seconds() / 3600.0
    clean = lead_h.notna() & (lead_h >= 0) & (lead_h <= 3.0)
    out["n_clean"] = int(clean.sum())
    # The signal-strength number is computed on clean captures only; mixing
    # in stale or in-game closes is exactly what fooled the college app.
    out["t"] = float("nan")
    if out["n_clean"]:
        c2 = cc[clean.values]
        out["clean_mean"] = float(c2.mean())
        if len(c2) > 1 and c2.std(ddof=1) > 0:
            out["t"] = out["clean_mean"] / (c2.std(ddof=1) / math.sqrt(len(c2)))
    return out


def grade(tracker, sched):
    if tracker.empty:
        return tracker, 0
    df = tracker.copy()
    pend = df["status"].astype(str) != "GRADED"
    if not pend.any():
        return df, 0

    fin = sched.dropna(subset=["home_score", "away_score"]).set_index("game_id")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    n = 0
    for idx in df[pend].index:
        gid = df.at[idx, "game_id"]
        if gid not in fin.index:
            continue
        g = fin.loc[gid]
        hs, as_ = float(g["home_score"]), float(g["away_score"])
        try:
            bl = float(df.at[idx, "bet_line"])
        except Exception:
            continue
        mt = str(df.at[idx, "market_type"]).upper()
        sd = str(df.at[idx, "pick_side"]).upper()

        if mt == "TOTAL":
            m = (hs + as_) - bl if sd == "OVER" else bl - (hs + as_)
        elif sd == "HOME":
            # bet_line is stored in nflverse convention: POSITIVE when the
            # home team is favoured (build_card does `[-p for t == home]` on
            # the book's American-style point, flipping the sign). So a home
            # favourite must BEAT the number, not be spotted it.
            #
            # This was `(hs - as_) + bl`, which spotted the home side its own
            # handicap: a home favourite laying 4.5 and winning by 2 graded
            # WIN, and a home underdog getting 3 who lost by 1 graded LOSS.
            # Every home spread bet in the ledger was graded against the
            # wrong sign. The away branch was always correct, which is why
            # roughly half the record looked plausible.
            m = (hs - as_) - bl
        else:
            m = (as_ - hs) + bl

        res = "PUSH" if abs(m) < 1e-9 else ("WIN" if m > 0 else "LOSS")
        # Pay at the price recorded when the bet was frozen. This was a flat
        # 100/110, so a +100 or -120 pick was graded as if it were -110.
        units = (0.0 if res == "PUSH" else
                 (american_payout(df.at[idx, "odds"]) if res == "WIN"
                  else -1.0))
        df.at[idx, "final_home_score"] = hs
        df.at[idx, "final_away_score"] = as_
        df.at[idx, "result_margin"] = round(m, 2)
        df.at[idx, "result"] = res
        df.at[idx, "units_result"] = round(units, 4)
        df.at[idx, "status"] = "GRADED"
        df.at[idx, "graded_at"] = now
        n += 1
    return (save_tracker(df), n) if n else (df, 0)


def summarize(df):
    if df.empty:
        return dict(w=0, l=0, p=0, units=0.0, roi=0.0, n=0)
    g = df[df["result"].isin(["WIN", "LOSS", "PUSH"])]
    if g.empty:
        return dict(w=0, l=0, p=0, units=0.0, roi=0.0, n=0)
    w = int((g["result"] == "WIN").sum())
    l = int((g["result"] == "LOSS").sum())
    p = int((g["result"] == "PUSH").sum())
    u = float(pd.to_numeric(g["units_result"], errors="coerce").fillna(0).sum())
    return dict(w=w, l=l, p=p, units=u, roi=u / len(g), n=len(g))


def ml_flags(sched, season, week, rt, sign, offers):
    """
    Moneylines worth taking, priced on the same scale as the board: shrunk
    toward the market's implied probability, not toward a coin flip.
    """
    if not rt or not offers:
        return []
    out = []
    games = sched[(sched["season"] == season) & (sched["week"] == week)]
    for _, g in games.iterrows():
        h, a = g["home_team"], g["away_team"]
        if h not in rt["margin"] or a not in rt["margin"]:
            continue
        live = lookup_offers(offers, a, h)
        if not live:
            continue
        raw = (rt["margin"][h] - rt["margin"][a] + rt["hfa"]
               + qb_delta(season, week, h, a)[0]
               + injury_delta(season, week, h, a)[0])
        p_home = norm_cdf(raw / SD_MARGIN)
        for team, prob in ((h, p_home), (a, 1.0 - p_home)):
            price = None
            for nm, pr, *_ in (live.get("moneylines") or []):
                if nm == team and (price is None or pr > price):
                    price = pr
            if price is None:
                continue
            imp = 1.0 / (1.0 + (price / 100.0)) if price > 0 else \
                abs(price) / (abs(price) + 100.0)
            ps = imp + MODEL_WEIGHT * (prob - imp)
            e = ev_from_prob(ps, price)
            if e is None:
                continue
            out.append({"matchup": f"{a} @ {h}", "pick": f"{team} ML",
                        "odds": int(price), "prob": ps, "ev": e})
    return out


# ----------------------------------------------------------------------
# Card rendering
# ----------------------------------------------------------------------
CARD_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700;800;900&display=swap');

/* Dark navy, blue accent, heavy display type — the same language as the
   college app, so the two read as one product. */
:root{
  --bg:#0A1020; --panel:#111A2E; --panel2:#16213A;
  --line:rgba(116,151,183,.16); --line2:rgba(116,151,183,.28);
  --ink:#E8F0FA; --muted:#8CA3BE; --faint:#61748C;
  --accent:#3B82F6; --accent2:#60A5FA;
  --go:#34D399; --warn:#F2C14E; --loss:#F87171;
}
html,body,[class*="css"],.stMarkdown,.stButton button,input,select{
  font-family:'Archivo',system-ui,-apple-system,sans-serif!important}
.stApp{background:
  radial-gradient(1100px 620px at 50% -12%,#15233F 0%,transparent 62%),
  linear-gradient(180deg,#0A1020 0%,#080D1A 100%)}
#MainMenu,footer,header[data-testid="stHeader"]{visibility:hidden;height:0}
.block-container{padding-top:1rem;padding-bottom:3.5rem;max-width:44rem}

/* Brand */
.se-brand{display:flex;align-items:center;gap:13px;margin:2px 0 16px}
.se-mark{width:52px;height:52px;border-radius:14px;flex:0 0 auto;
  border:1px solid var(--line2);display:flex;align-items:center;
  justify-content:center;
  background:linear-gradient(160deg,rgba(59,130,246,.22),rgba(59,130,246,.05))}
.se-mark b{font-size:1.45rem;font-weight:900;color:var(--accent2);
  line-height:1}
.se-mark svg{display:block}
.se-brand h1{margin:0;font-size:1.5rem;font-weight:900;letter-spacing:-.03em;
  line-height:1;color:var(--ink);font-style:italic}
.se-brand h1 em{color:var(--accent2);font-style:italic}
.se-brand .tag{font-size:.6rem;letter-spacing:.2em;color:var(--faint);
  font-weight:700;margin-top:5px}

/* Section labels */
.se-sec{font-size:.62rem;font-weight:800;letter-spacing:.17em;
  text-transform:uppercase;color:var(--accent2);margin:26px 0 8px}
.cap{color:var(--muted);font-size:.8rem;margin:-4px 0 10px}

/* Display heading */
.se-head{margin:0 0 6px}
.se-head h1{font-size:2rem;font-weight:900;letter-spacing:-.035em;
  line-height:1.02;margin:0;color:var(--ink)}
.se-head h1 span{color:var(--accent2)}
.se-head .sub{color:var(--muted);font-size:.86rem;margin-top:6px}

/* The card */
.sc-wrap{border:1px solid var(--line);border-radius:18px;overflow:hidden;
  margin-bottom:12px;background:var(--panel);
  box-shadow:0 18px 40px -26px rgba(0,0,0,.9)}
.sc-top{padding:16px 18px 14px;
  background:linear-gradient(135deg,#16233D 0%,#101A2E 100%);
  border-bottom:1px solid var(--line)}
.sc-top h2{margin:0;font-size:1.2rem;font-weight:900;letter-spacing:-.025em;
  color:var(--ink)}
.sc-top span{display:block;font-size:.72rem;color:var(--muted);margin-top:4px;
  font-weight:600}
.sc-grp{font-size:.6rem;font-weight:800;letter-spacing:.17em;
  text-transform:uppercase;color:var(--faint);padding:12px 18px 5px;
  background:rgba(255,255,255,.015)}
.sc-row{display:flex;align-items:center;gap:12px;padding:12px 18px;
  border-top:1px solid var(--line)}
.sc-rank{flex:0 0 22px;height:22px;border-radius:7px;display:flex;
  align-items:center;justify-content:center;font-size:.68rem;font-weight:800;
  color:var(--accent2);background:rgba(59,130,246,.12);
  border:1px solid rgba(59,130,246,.22)}
.sc-main{flex:1 1 auto;min-width:0}
.sc-main b{display:block;font-size:1.04rem;font-weight:800;color:var(--ink);
  letter-spacing:-.025em;white-space:nowrap;overflow:hidden;
  text-overflow:ellipsis}
.sc-main small{display:block;font-size:.71rem;color:var(--muted);margin-top:2px;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.sc-num{flex:0 0 auto;text-align:right;font-variant-numeric:tabular-nums}
.sc-num b{display:block;font-size:.92rem;font-weight:800;color:var(--ink)}
.sc-num span{display:block;font-size:.66rem;color:var(--muted);font-weight:600}
.sc-foot{padding:11px 18px;font-size:.68rem;color:var(--faint);
  background:rgba(255,255,255,.015);border-top:1px solid var(--line)}

/* Rows in the detail view */
.se-row{border-top:1px solid var(--line);padding:16px 0}
.se-row:last-of-type{border-bottom:1px solid var(--line)}
.se-tag{display:inline-block;font-size:.62rem;font-weight:800;
  padding:4px 10px;border-radius:6px;letter-spacing:.06em;
  text-transform:uppercase}
.se-tag.go{background:var(--go);color:#06281C}
.se-tag.hold{background:rgba(242,193,78,.15);color:var(--warn)}
.se-tag.off{background:rgba(140,163,190,.12);color:var(--muted)}
.se-pick{font-size:1.4rem;font-weight:900;letter-spacing:-.03em;
  line-height:1.1;margin:9px 0 3px;color:var(--ink)}
.se-meta{color:var(--muted);font-size:.8rem}
.se-meta .sep{color:var(--faint);padding:0 7px}
.se-stats{display:grid;grid-template-columns:repeat(4,1fr);gap:0 8px;
  margin-top:14px}
.se-stats > div{display:flex;flex-direction:column}
.se-stats .v{font-size:1rem;font-weight:800;line-height:1.3;color:var(--ink);
  font-variant-numeric:tabular-nums}
.se-stats .k{font-size:.63rem;color:var(--faint);font-weight:700;
  letter-spacing:.06em;text-transform:uppercase}
.v.pos{color:var(--go)} .v.neg{color:var(--loss)}
.se-note{margin-top:12px;font-size:.76rem;color:var(--muted);
  background:rgba(255,255,255,.03);border:1px solid var(--line);
  border-radius:9px;padding:9px 12px;line-height:1.5}
.se-note b{color:var(--ink);font-weight:700}

/* Moneyline flags */
.se-mlf{display:flex;align-items:center;gap:12px;padding:12px 15px;
  border:1px solid rgba(242,193,78,.24);border-radius:13px;margin-bottom:8px;
  background:linear-gradient(180deg,rgba(242,193,78,.08),rgba(242,193,78,.02))}
.se-mlf-main{flex:1 1 auto;min-width:0}
.se-mlf-main b{display:block;font-size:1rem;font-weight:800;color:var(--ink);
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.se-mlf-main small{display:block;font-size:.72rem;color:var(--muted);
  margin-top:2px}
.se-mlf-stats{flex:0 0 auto;text-align:right;
  font-variant-numeric:tabular-nums}
.se-mlf-stats b{display:block;font-size:.9rem;font-weight:800;
  color:var(--warn)}
.se-mlf-stats span{display:block;font-size:.68rem;color:var(--muted)}

/* Empty state */
.se-empty{border:1px solid var(--line);border-radius:14px;padding:26px 20px;
  background:var(--panel)}
.se-empty h2{font-size:1.25rem;font-weight:800;margin:0 0 7px;
  letter-spacing:-.025em;color:var(--ink)}
.se-empty p{color:var(--muted);font-size:.86rem;margin:0;line-height:1.55}

/* Streamlit widgets.
   These are forced here rather than left to config.toml: if that file is
   missing from the repo the components render light on a dark background,
   which is unreadable. Belt and braces. */
h1,h2,h3,h4,h5,p,span,label,li{color:var(--ink)}
[data-testid="stWidgetLabel"] p,[data-testid="stWidgetLabel"] label{
  color:var(--muted)!important;font-size:.72rem!important;font-weight:700!important;
  letter-spacing:.04em!important}
[data-baseweb="select"] > div{background:var(--panel2)!important;
  border:1px solid var(--line2)!important;color:var(--ink)!important;
  border-radius:12px!important}
[data-baseweb="select"] svg{fill:var(--muted)!important}
[data-baseweb="popover"] li{background:var(--panel2)!important;
  color:var(--ink)!important}
[data-baseweb="menu"]{background:var(--panel2)!important}
input,textarea{background:var(--panel2)!important;color:var(--ink)!important}
[data-testid="stTabs"] button{color:var(--muted)!important;
  font-weight:700!important}
[data-testid="stTabs"] button[aria-selected="true"]{color:var(--ink)!important}
[data-testid="stTabs"] [data-baseweb="tab-highlight"]{
  background:var(--accent)!important}
[data-testid="stCaptionContainer"] p{color:var(--muted)!important}
[data-testid="stAlert"]{border-radius:12px!important;
  border:1px solid var(--line2)!important;background:var(--panel)!important}
[data-testid="stAlert"] p{color:var(--ink)!important}
[data-testid="stExpander"] summary p{color:var(--ink)!important;
  font-weight:700!important}
[data-testid="stDataFrame"]{background:var(--panel)!important;
  border:1px solid var(--line)!important;border-radius:12px!important}
.stSlider [data-baseweb="slider"] div{background:var(--accent)!important}

/* A table I control, instead of st.dataframe */
.se-kv{width:100%;border-collapse:collapse;border:1px solid var(--line);
  border-radius:12px;overflow:hidden;background:var(--panel);margin:6px 0 4px}
.se-kv tr{border-bottom:1px solid var(--line)}
.se-kv tr:last-child{border-bottom:0}
.se-kv td{padding:11px 14px;font-size:.87rem}
.se-kv td:first-child{color:var(--muted);font-weight:600}
.se-kv td:last-child{text-align:right;color:var(--ink);font-weight:700;
  font-variant-numeric:tabular-nums}

/* Streamlit widgets */
.stButton button{border-radius:12px!important;font-weight:800!important;
  letter-spacing:-.01em!important;border:1px solid var(--line2)!important}
.stButton button[kind="primary"]{
  background:linear-gradient(180deg,#3B82F6,#2563EB)!important;
  border:0!important;box-shadow:0 12px 26px -14px rgba(59,130,246,.9)!important}
[data-testid="stMetricValue"]{font-weight:900!important;
  letter-spacing:-.03em!important}
[data-testid="stMetricLabel"]{font-size:.62rem!important;
  letter-spacing:.14em!important;text-transform:uppercase!important;
  color:var(--faint)!important}
div[data-testid="stExpander"]{border:1px solid var(--line)!important;
  border-radius:13px!important;background:var(--panel)!important}

/* ---- Saturday Edge parity ------------------------------------------- */

/* Segmented pill nav. Streamlit's default tabs are an underlined text row;
   the college app uses filled pills in a rounded tray, and the two products
   should not read as different apps. Styled rather than rebuilt, so the
   `with tab_x:` blocks below stay exactly as they are. */
/* Streamlit renames these internals between versions, so target the stable
   testid and the ARIA roles rather than data-baseweb, which did not match
   on the deployed build and left the default red-underlined text tabs. */
[data-testid="stTabs"] div[role="tablist"]{
  display:grid!important;grid-template-columns:repeat(3,minmax(0,1fr))!important;
  gap:4px!important;padding:4px!important;margin:2px 0 14px!important;
  border-radius:14px!important;background:#0A1628!important;
  border:1px solid var(--line)!important;
}
/* Kill the underline/highlight bar in every form it ships as. */
[data-testid="stTabs"] div[role="tablist"] > div[data-baseweb="tab-highlight"],
[data-testid="stTabs"] div[role="tablist"] > div[data-baseweb="tab-border"],
[data-testid="stTabs"] div[role="tablist"]::after,
[data-testid="stTabs"] div[role="tablist"] > div:not([role="tab"]):empty{
  display:none!important;height:0!important;background:transparent!important;
}
[data-testid="stTabs"] button[role="tab"]{
  width:100%!important;min-height:38px!important;padding:9px 2px!important;
  margin:0!important;border:0!important;border-radius:10px!important;
  background:transparent!important;
  display:flex!important;align-items:center!important;
  justify-content:center!important;
  transition:background .15s ease,color .15s ease;
}
[data-testid="stTabs"] button[role="tab"] p,
[data-testid="stTabs"] button[role="tab"] div{
  margin:0!important;font-size:.72rem!important;font-weight:850!important;
  letter-spacing:-.01em!important;color:var(--muted)!important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"]{
  background:linear-gradient(145deg,#174676,#10345a)!important;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.035)!important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p,
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] div{
  color:#F7FBFF!important;
}

/* Alerts. Without a config.toml the theme falls back to Streamlit red for
   primaryColor and a green-ish panel for st.warning, neither of which is in
   this palette. The owner warning in particular must read as a warning. */
div[data-testid="stAlertContainer"],div[data-testid="stAlert"]{
  border-radius:13px!important;
}
div[data-testid="stAlertContainer"]:has(svg),
div[data-testid="stAlert"]{
  background:rgba(242,193,78,.09)!important;
  border:1px solid rgba(242,193,78,.34)!important;
}
div[data-testid="stAlertContainer"] p,div[data-testid="stAlert"] p{
  color:#F6DFA4!important;font-size:.76rem!important;
}
div[data-testid="stAlertContainer"] code,div[data-testid="stAlert"] code{
  background:rgba(242,193,78,.16)!important;color:#F8E9BF!important;
}

/* Four-up stat strip, in place of st.metric rows. */
.se-stat-strip{display:flex;gap:0;margin-bottom:14px;padding:14px 8px;
  border-radius:14px;background:rgba(12,26,44,.55);border:1px solid var(--line)}
.se-stat-strip > div{flex:1;text-align:center;
  border-right:1px solid rgba(116,151,183,.10)}
.se-stat-strip > div:last-child{border-right:none}
.se-stat-strip b{display:block;font-size:1.02rem;color:var(--ink);font-weight:800;
  font-variant-numeric:tabular-nums}
.se-stat-strip b.pos{color:var(--go)} .se-stat-strip b.neg{color:var(--loss)}
.se-stat-strip span{display:block;margin-top:3px;font-size:.58rem;
  letter-spacing:.09em;text-transform:uppercase;color:var(--faint)}

/* Chart frame + status chip. */
.se-curve{padding:10px 8px 10px;border-radius:14px;margin-bottom:10px;
  background:rgba(12,26,44,.5);border:1px solid var(--line)}
.se-curve svg{display:block;width:100%;height:auto}
.se-verdict{display:flex;align-items:baseline;gap:8px;margin:2px 0 10px;
  padding:10px 13px;border-radius:12px;
  background:rgba(12,26,44,.55);border:1px solid var(--line)}
.se-verdict b{font-size:.80rem;font-weight:800;color:var(--ink);
  letter-spacing:-.01em}
.se-verdict em{font-style:normal;font-size:.62rem;font-weight:700;
  color:var(--muted);margin-left:auto}
.se-verdict-dot{width:7px;height:7px;border-radius:50%;flex:0 0 7px;
  align-self:center}
.se-verdict.pos .se-verdict-dot{background:var(--go);
  box-shadow:0 0 8px rgba(52,211,153,.55)}
.se-verdict.neg .se-verdict-dot{background:var(--loss);
  box-shadow:0 0 8px rgba(248,113,113,.45)}
.se-verdict.wait .se-verdict-dot{background:var(--warn);
  box-shadow:0 0 8px rgba(242,193,78,.45)}

/* Compact list rows. */
.se-extra{display:flex;align-items:center;gap:10px;padding:9px 4px;
  border-bottom:1px solid rgba(116,151,183,.08)}
.se-extra:last-child{border-bottom:none}
.se-extra-rank{width:20px;flex:0 0 20px;font-size:.62rem;color:var(--muted);
  font-weight:800;text-align:center}
.se-extra-main{flex:1;min-width:0}
.se-extra-main b{display:block;font-size:.82rem;color:var(--ink);font-weight:800}
.se-extra-main small{display:block;font-size:.63rem;color:var(--muted);
  margin-top:1px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.se-extra-ev{font-size:.8rem;font-weight:800;white-space:nowrap;color:var(--go)}
.se-extra-ev.neg{color:var(--loss)}
.sc-note{padding:8px 12px;margin:0 0 2px;font-size:.66rem;line-height:1.5;
  color:var(--muted);border-bottom:1px solid rgba(116,151,183,.08)}
.sc-note b{color:var(--ink)}
.sc-note.warn{color:#F6DFA4;background:rgba(242,193,78,.07)}
</style>
"""


def se_equity_svg(units, w=320, h=104):
    """Cumulative units, with its own scale. Ported from the college app so
    the two products show a record the same way."""
    pts = [float(v) for v in units
           if v is not None and isinstance(v, (int, float)) and math.isfinite(float(v))]
    if len(pts) < 2:
        return ""
    cum, run = [], 0.0
    for v in pts:
        run += v
        cum.append(run)
    peak, trough = max(max(cum), 0.0), min(min(cum), 0.0)
    lo, hi = trough, peak
    if hi - lo < 1e-9:
        hi, lo = hi + 1, lo - 1
    pad = (hi - lo) * 0.18
    lo, hi = lo - pad, hi + pad
    ml, mr, mt, mb = 34, 10, 12, 14

    def X(i):
        return ml + (w - ml - mr) * (i / max(len(cum) - 1, 1))

    def Y(v):
        return mt + (h - mt - mb) * (1 - (v - lo) / (hi - lo))

    zero = Y(0.0)
    line = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(cum))
    area = f"{X(0):.1f},{zero:.1f} " + line + f" {X(len(cum)-1):.1f},{zero:.1f}"
    end = cum[-1]
    col = "#34D399" if end >= 0 else "#F87171"
    return (
        f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" '
        f'preserveAspectRatio="xMidYMid meet" style="display:block" '
        f'xmlns="http://www.w3.org/2000/svg" role="img" '
        f'aria-label="Cumulative units over {len(cum)} bets">'
        f'<defs><linearGradient id="seEq" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%" stop-color="{col}" stop-opacity=".18"/>'
        f'<stop offset="100%" stop-color="{col}" stop-opacity="0"/>'
        f'</linearGradient></defs>'
        f'<polygon points="{area}" fill="url(#seEq)"/>'
        f'<line x1="{ml}" y1="{zero:.1f}" x2="{w-mr}" y2="{zero:.1f}" '
        f'stroke="#8CA3BE" stroke-opacity=".40" stroke-width="1" '
        f'stroke-dasharray="3 3"/>'
        f'<text x="{ml-5}" y="{Y(peak)+3:.1f}" fill="#8CA3BE" font-size="8" '
        f'font-weight="700" text-anchor="end">{peak:+.1f}u</text>'
        f'<text x="{ml-5}" y="{zero+3:.1f}" fill="#A8BCD4" font-size="8" '
        f'font-weight="800" text-anchor="end">0</text>'
        f'<text x="{ml-5}" y="{Y(trough)+3:.1f}" fill="#8CA3BE" font-size="8" '
        f'font-weight="700" text-anchor="end">{trough:+.1f}u</text>'
        f'<polyline points="{line}" fill="none" stroke="{col}" stroke-width="1.8" '
        f'stroke-linejoin="round" stroke-linecap="round"/>'
        f'<circle cx="{X(len(cum)-1):.1f}" cy="{Y(end):.1f}" r="3.2" fill="{col}"/>'
        f'<text x="{ml}" y="{h-3}" fill="#61748C" font-size="7.5" '
        f'font-weight="700" text-anchor="start">BET 1</text>'
        f'<text x="{w-mr}" y="{h-3}" fill="#61748C" font-size="7.5" '
        f'font-weight="700" text-anchor="end">BET {len(cum)}</text>'
        f'</svg>'
    )


def se_winrate_ci_svg(wins, losses, w=320, h=74, breakeven=0.5238):
    """Observed hit rate with its 95% band, against the -110 breakeven."""
    n = int(wins) + int(losses)
    if n < 5:
        return ""
    p = wins / n
    se = math.sqrt(breakeven * (1 - breakeven) / n)
    lo_ci, hi_ci = max(p - 1.96 * se, 0.0), min(p + 1.96 * se, 1.0)
    lo, hi = min(lo_ci, breakeven) - 0.05, max(hi_ci, breakeven) + 0.05
    ml, mr, bar_y, bar_h = 8, 8, 30, 13

    def X(v):
        return ml + (w - ml - mr) * ((v - lo) / (hi - lo))

    inside = lo_ci <= breakeven <= hi_ci
    col = "#60A5FA" if inside else ("#34D399" if p > breakeven else "#F87171")
    return (
        f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" '
        f'preserveAspectRatio="xMidYMid meet" style="display:block" '
        f'xmlns="http://www.w3.org/2000/svg" role="img" '
        f'aria-label="Hit rate {p:.1%}, 95% interval {lo_ci:.1%} to {hi_ci:.1%}">'
        f'<rect x="{ml}" y="{bar_y}" width="{w-ml-mr}" height="{bar_h}" rx="6" '
        f'fill="#8CA3BE" fill-opacity=".10"/>'
        f'<rect x="{X(lo_ci):.1f}" y="{bar_y}" '
        f'width="{max(X(hi_ci)-X(lo_ci),2):.1f}" height="{bar_h}" rx="6" '
        f'fill="{col}" fill-opacity=".30"/>'
        f'<line x1="{X(breakeven):.1f}" y1="{bar_y-7}" x2="{X(breakeven):.1f}" '
        f'y2="{bar_y+bar_h+7}" stroke="#F2C14E" stroke-width="1.6"/>'
        f'<text x="{X(breakeven):.1f}" y="{bar_y-11}" fill="#F2C14E" '
        f'font-size="8.5" font-weight="800" text-anchor="middle">'
        f'BREAK EVEN {breakeven*100:.1f}%</text>'
        f'<circle cx="{X(p):.1f}" cy="{bar_y+bar_h/2:.1f}" r="4.2" fill="{col}"/>'
        f'<text x="{X(lo_ci):.1f}" y="{bar_y+bar_h+16}" fill="#8CA3BE" '
        f'font-size="8" font-weight="700" text-anchor="start">{lo_ci*100:.0f}%</text>'
        f'<text x="{X(hi_ci):.1f}" y="{bar_y+bar_h+16}" fill="#8CA3BE" '
        f'font-size="8" font-weight="700" text-anchor="end">{hi_ci*100:.0f}%</text>'
        f'<text x="{X(p):.1f}" y="{bar_y+bar_h+16}" fill="#E8F0FA" font-size="8.5" '
        f'font-weight="800" text-anchor="middle">{p*100:.1f}%</text>'
        f'</svg>'
    )


def stat_strip(items):
    """items: list of (value, label, tone) where tone is "", "pos" or "neg"."""
    cells = "".join(
        f'<div><b{f" class=\"{t}\"" if t else ""}>{_html.escape(str(v))}</b>'
        f'<span>{_html.escape(str(k))}</span></div>'
        for v, k, t in items
    )
    return f'<div class="se-stat-strip">{cells}</div>'


def brand_header():
    st.markdown(
        '<div class="se-brand">'
        '<div class="se-mark">'
        '<svg viewBox="0 0 24 24" width="26" height="26" fill="none" '
        'stroke="#60A5FA" stroke-width="2.4" stroke-linecap="round">'
        '<path d="M6 4v6"/><path d="M18 4v6"/><path d="M6 10h12"/>'
        '<path d="M12 10v10"/></svg>'
        '</div>'
        '<div><h1>SUNDAY <em>EDGE</em></h1>'
        '<div class="tag">MEASURED. NOT ASSUMED.</div></div>'
        '</div>', unsafe_allow_html=True)


def render_row(r, badge):
    """
    Verdict first, then the pick, then the numbers on one aligned grid.
    The note spells out the discount: readers kept seeing a six-point
    disagreement and a "no bet" and could not connect the two, because
    the weighting step happened invisibly between them.
    """
    cls = {"Bet": "go", "Best bet": "go", "Watch": "hold"}.get(badge, "off")
    ev = float(r["expected_value"])
    e = _html.escape
    side = str(r.get("pick_side", "")).upper()
    line, model, edge = (float(r["bet_line"]), float(r["model_line"]),
                         float(r["edge_pts"]))
    if str(r["market_type"]).upper() == "SPREAD":
        shown_line = -line if side == "HOME" else line
        shown_model = -model if side == "HOME" else model
        lab = "Line"
        fl, fm = f"{shown_line:+g}", f"{shown_model:+.1f}"
    else:
        shown_line, shown_model, lab = line, model, "Total"
        fl, fm = f"{shown_line:g}", f"{shown_model:.1f}"
    gap = abs(shown_model - shown_line)
    kick = str(r.get("kickoff", "")).strip()
    try:
        _ia = float(r.get("inj_adj") or 0.0)
    except Exception:
        _ia = 0.0
    _in = r.get("inj_note")
    _inj_html = ""
    if abs(_ia) >= 0.05 and isinstance(_in, str) and _in:
        # Same sign convention as the Model number on this row.
        _shown_ia = -_ia if (lab == "Line" and side == "HOME") else _ia
        _inj_html = (f"<br>Injuries moved the model line <b>{_shown_ia:+.1f}</b>"
                     f" ({e(_in)}).")
    st.markdown(f"""
<div class="se-row">
  <span class="se-tag {cls}">{e(badge)}</span>
  <div class="se-pick">{e(str(r['pick_label']))}</div>
  <div class="se-meta">{e(str(r['matchup']))}{
     f'<span class="sep">/</span>{e(kick)}' if kick else ''}</div>
  <div class="se-stats">
    <div><span class="v">{fl}</span><span class="k">{lab}</span></div>
    <div><span class="v">{fm}</span><span class="k">Model</span></div>
    <div><span class="v">{edge:+.2f}</span><span class="k">Edge</span></div>
    <div><span class="v {'pos' if ev>0 else 'neg'}">{ev:+.2%}</span>
         <span class="k">Value</span></div>
  </div>
  <div class="se-note">Model is <b>{gap:.1f}</b> points off the market.
    Weighted at {MODEL_WEIGHT}, that becomes <b>{abs(edge):.2f}</b> points of
    edge &mdash; the weight this model earned against
    {BACKTEST_N:,} past games.{_inj_html}</div>
</div>""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.markdown(CARD_CSS, unsafe_allow_html=True)
brand_header()
st.caption(f"NFL spreads and totals · model {model_version()}")

try:
    _owner_code = st.secrets.get("owner_code", "")
except Exception:
    _owner_code = ""
if not _owner_code:
    st.warning(
        "**No owner code set.** Anyone with this link can freeze bets into "
        "the shared record. Add an `owner_code` to Streamlit Secrets before "
        "sharing the URL."
    )
elif not is_owner():
    st.caption("Viewing the shared record \u2014 picks and grading are "
               "managed by the owner.")
    with st.expander("Owner sign-in", expanded=False):
        _try = st.text_input("Owner code", type="password", key="owner_try")
        if st.button("Unlock", key="owner_btn"):
            if _try == _owner_code:
                st.session_state["owner_ok"] = True
                st.rerun()
            else:
                st.error("Incorrect code.")

st.warning(
    f"**This model did not beat the closing line in backtest.** Across "
    f"{BACKTEST_N:,} games (2007-2025) it added no measurable information "
    f"beyond the market (t = +{BACKTEST_T:.2f}). Picks below are the "
    f"model's lean, not a demonstrated edge. The tracker is built to give "
    f"you a real answer as the record accumulates."
)

c_ref, c_stamp = st.columns([1, 3])
if c_ref.button("Refresh lines", use_container_width=True):
    st.session_state["bust"] = st.session_state.get("bust", 0) + 1

sched_all = None
try:
    this_season = datetime.now().year
    sched_all = load_schedules(range(this_season - 3, this_season + 1),
                               _bust=st.session_state.get("bust", 0))
except Exception as e:
    st.error(f"Could not load NFL schedules: {e}")
    st.stop()

live_offers, credits_left, odds_err = fetch_live_odds(
    _bust=st.session_state.get("bust", 0))

# Price at the book you actually bet. The default used to be "Best of all
# books", which logged every pick at the single best number and price anywhere
# in the feed -- numbers you cannot get at 734 Games, so every edge and every
# graded unit was overstated. The default is now the consensus number at
# ASSUMED_PRICE. The unfiltered feed is kept for closing lines, which should
# be the market's number, not one book's.
all_offers = live_offers or {}
_books = sorted({bk for o in all_offers.values()
                 for _, _, _, bk in (o.get("spreads", []) + o.get("totals", []))})
if _books:
    _book = st.selectbox(
        "Your book", [CONSENSUS_LABEL] + _books + ["Best of all books"],
        key="se_book",
        help="734 Games is not in the odds feed, so the consensus line is "
             "the closest stand-in. Check your book's number before betting.")
    if _book == CONSENSUS_LABEL:
        live_offers = consensus_offers(all_offers, ASSUMED_PRICE)
    elif _book != "Best of all books":
        live_offers = filter_offers(all_offers, _book)
if odds_err == "no key":
    st.info("Add `odds_api_key` to Streamlit secrets for live multi-book "
            "lines and line shopping. Using nflverse lines for now.")
elif odds_err:
    st.warning(f"Live odds unavailable ({odds_err}). Using nflverse lines.")
elif live_offers:
    st.success(
        f"Live odds for {len(live_offers)} games across US books"
        + (f" · {credits_left} API credits left" if credits_left else "")
    )

_pulled = st.session_state.get("lines_pulled_at")
if _pulled:
    _age = (datetime.now(timezone.utc) - _pulled).total_seconds() / 60
    c_stamp.caption(
        f"Lines pulled {int(_age)} min ago from nflverse. These update "
        f"periodically, not tick-by-tick — check your book before betting."
    )

sign = line_sign(sched_all)

tab_slate, tab_game, tab_tracker = st.tabs(["Slate", "Game", "Tracker"])

with tab_slate:
    seasons = sorted(sched_all["season"].unique())
    c1, c2 = st.columns(2)
    season = c1.selectbox("Season", seasons, index=len(seasons) - 1)
    weeks = sorted(sched_all[sched_all["season"] == season]["week"].unique())
    week = c2.selectbox("Week", weeks, index=min(len(weeks) - 1, 0))

    # st.stop() here would halt the WHOLE script, not just this tab, so the
    # Game and Tracker tabs rendered blank until the card was run. Use a flag.
    if st.button("Build card", type="primary", use_container_width=True):
        st.session_state["se_ran"] = True
    _ran = bool(st.session_state.get("se_ran"))

    if not _ran:
        st.caption(
            "Pick the week, then build. Thursday, Sunday and Monday games "
            "each get their own card \u2014 choose the day below."
        )
        card, rt = pd.DataFrame(), None
    else:
        card, rt = build_card(sched_all, season, week, sign,
                              offers=live_offers)
    st.markdown(CARD_CSS, unsafe_allow_html=True)

    if not _ran:
        pass
    elif rt is None:
        st.info("Not enough completed games yet to build ratings.")
    else:
        # An NFL week runs Thursday to Monday. The card is for ONE day, and
        # never includes a game that has already kicked off.
        _now = pd.Timestamp.now(tz="America/New_York").tz_localize(None)
        if not card.empty:
            _k = pd.to_datetime(card["kickoff"], errors="coerce")
            card = card.assign(_kick=_k)
            card = card[card["_kick"].notna() & (card["_kick"] > _now)]

        _ok = not card.empty
        if not _ok:
            st.info(
                "Every game in this week has kicked off \u2014 Thursday "
                "through Monday. Pick a later week above."
            )

        _days = sorted(card["_kick"].dt.date.unique()) if _ok else []
        _day = None
        if _ok:
            _day = st.selectbox(
                "Day", _days, index=0,
                format_func=lambda d: pd.Timestamp(d).strftime("%A, %b %-d"),
                help="Every day in this week that still has games to play.",
            )
            card = card[card["_kick"].dt.date == _day]
            _ok = not card.empty

        _ml = ([f for f in (ml_flags(sched_all, season, week, rt, sign,
                                     live_offers) or [])
                if str(f.get("matchup")) in set(card["matchup"])]
               if _ok else [])
        _kick = (pd.Timestamp(_day).strftime("%A, %b %-d")
                 if _day is not None else f"Week {week}")

        def _rows(df, n=None):
            """
            Bets (positive value at your price) first, then watch-list
            markets (MIN_GAP_PTS+ off the line but not beating the vig).
            Same rule build_card tiers on and freeze logs, so what is on
            screen is exactly what the tracker records.
            """
            if df.empty:
                return [], 0
            d = df[df["bet_tier"].isin(["OFFICIAL", "WATCH"])]
            return list(d.iterrows()), len(d)

        _sp, _nsp = _rows(card[card["market_type"] == "SPREAD"])
        _to, _nto = _rows(card[card["market_type"] == "TOTAL"])
        # Moneylines keep the EV test. Spreads and totals are ranked by
        # disagreement because both sides of those are playable at the same
        # number; a moneyline has no number, so the only thing separating
        # the two sides is price. Without this the card showed DEN +110 and
        # KC -135 on the same game, which cannot both be bets.
        _mo = sorted([f for f in _ml if f["ev"] >= MIN_EV],
                     key=lambda r: -r["ev"])[:2]
        _seen = set()
        _mo = [f for f in _mo
               if not (f["matchup"] in _seen or _seen.add(f["matchup"]))]
        _shown = len(_sp) + len(_to)
        _nq = _nsp + _nto + len(_mo)
        _n = len(_sp) + len(_to) + len(_mo)
        # Moneylines count in the numerator, so they must count in the
        # denominator too — otherwise "3 of 2 markets qualify".
        _total_markets = len(card) + len(_ml)

        _h = [f'<div class="sc-wrap">'
              f'<div class="sc-top"><h2>Top picks</h2>'
              f'<span>{_html.escape(_kick)} \u00b7 {_n} plays \u00b7 '
              f'bets beat the vig \u00b7 watch = {MIN_GAP_PTS:g}+ pts off'
              f'</span></div>']

        # Say what the quarterback adjustment did. If it is not applied, say
        # that too — a reader should never have to guess whether a pick
        # exists because the model spotted something or because it did not
        # know a starter was out.
        _iq = load_injury_adjustment() or {}
        if _iq and not qb_model_on():
            _why_q = (_iq.get("qb_error") or
                      ("QB value did not clear the significance bar"
                       if _iq.get("qb") else "not measured yet"))
            _h.append(
                f'<div class="sc-note warn">Individual QB ratings are OFF '
                f'({_html.escape(str(_why_q)[:140])}). Quarterback changes use '
                f'a flat penalty that ignores who the replacement is. Retries '
                f'within a day, or tap Re-measure in Tracker \u2192 Injury '
                f'model.</div>')
        if qb_model_on():
            # Individual QB ratings: name every change and its price.
            _qbc = []
            for _, _gr in card.drop_duplicates("matchup").iterrows() if not card.empty else []:
                for _t in (_gr["away_team"], _gr["home_team"]):
                    try:
                        _e = expected_qbs(season, week, _t)
                    except Exception:
                        _e = None
                    if _e and _e["changed"] and _e.get("usual_rating") is not None:
                        _pts = float(qb_params()["margin"]["coef"]) * (
                            _e["rating"] - _e["usual_rating"])
                        _qbc.append(f'{_e["team"]}: {_qb_last(_e["starter"])} '
                                    f'for {_qb_last(_e["usual"])} ({_pts:+.1f})')
            if _qbc:
                _h.append(
                    '<div class="sc-note">Quarterback changes priced: <b>'
                    + _html.escape("; ".join(_qbc)) + '</b></div>')
            else:
                _h.append('<div class="sc-note">Quarterbacks rated individually '
                          '\u2014 no starter changes on this card.</div>')
        else:
            _adj = load_qb_adjustment()
            if _adj:
                _stat = qb_status(season, week)
                _flagged = sorted(t for t, v in _stat.items() if v.get("changed"))
                _teams_on_card = set(card.get("home_team", [])) | \
                    set(card.get("away_team", []))
                _down = [t for t in _flagged if t in _teams_on_card
                         and _stat.get(t, {}).get("direction") == "downgrade"]
                _other = [t for t in _flagged if t in _teams_on_card
                          and t not in _down]
                if _down:
                    _h.append(
                        f'<div class="sc-note">Backup quarterback adjusted for: '
                        f'<b>{_html.escape(", ".join(_down))}</b> '
                        f'({_adj["penalty_points"]:.1f} pts, measured over '
                        f'{_adj.get("n_games_total", 0):,} games)</div>')
                if _other:
                    # A returning starter is an upgrade, and the model has no way
                    # to price one. Say so rather than silently ignoring it.
                    _h.append(
                        f'<div class="sc-note warn">Quarterback change NOT '
                        f'adjusted for: <b>{_html.escape(", ".join(_other))}</b> '
                        f'\u2014 the listed starter is not a downgrade, and the '
                        f'model cannot price an upgrade. Treat these picks with '
                        f'caution.</div>')
                if not _down and not _other:
                    _h.append(
                        '<div class="sc-note">No quarterback changes detected '
                        'this week.</div>')
            else:
                if load_injury_adjustment():
                    _h.append(
                        '<div class="sc-note">Quarterbacks on the injury report '
                        'are adjusted through the injury model below. A benched '
                        '(healthy) starter is not on that report and is NOT '
                        'adjusted for \u2014 check QB news before betting.</div>')
                else:
                    _h.append(
                        '<div class="sc-note warn">No quarterback adjustment '
                        'applied \u2014 the model does not know who is starting. '
                        'A pick can exist purely because a starter is out and the '
                        'line moved without it.</div>')

        # Same for everyone else on the injury report.
        _iadj = load_injury_adjustment()
        if not _iadj:
            _why = st.session_state.get("inj_auto_failed") or "not measured"
            _h.append(
                f'<div class="sc-note warn">No injury adjustment for non-QB '
                f'players \u2014 the automatic measurement did not complete '
                f'({_html.escape(str(_why)[:120])}). It will retry next '
                f'session.</div>')
        else:
            _il, _ist = injury_loads(season, week, INJ_SCHEMA)
            if _ist != "ok":
                _h.append(
                    f'<div class="sc-note warn">Injury adjustment is on, but '
                    f'{_html.escape(_ist)} \u2014 these picks do not reflect '
                    f'this week\'s injuries. Rebuild once Friday\'s report is '
                    f'out.</div>')
            else:
                _ib = card[card["inj_adj"].abs() >= 0.3] if (
                    not card.empty and "inj_adj" in card.columns) else card
                _ng = _ib["matchup"].nunique() if len(_ib) else 0
                _used = [inj_mod.GROUP_LABEL[g] for g in inj_mod.GROUPS
                         if inj_mod.usable(_iadj, "margin", g, -1, INJ_MIN_T)
                         and not (g == "QB" and load_qb_adjustment())]
                _h.append(
                    f'<div class="sc-note">Injury report applied '
                    f'({_html.escape(", ".join(_used)) or "no position group cleared the bar"}'
                    f'). Moved the model line 0.3+ pts in {_ng} game(s). '
                    f'Sunday inactives can still change things.</div>')

        def _blk(title, items):
            if not items:
                return
            _h.append(f'<div class="sc-grp">{title}</div>')
            for i, (_, r) in enumerate(items, start=1):
                side = str(r.get("pick_side", "")).upper()
                line = float(r["bet_line"])
                if str(r["market_type"]).upper() == "SPREAD":
                    shown = -line if side == "HOME" else line
                    num = f"{shown:+g}"
                else:
                    num = f"{line:g}"
                odds = int(r.get("odds") or -110)
                kick = str(r.get("kickoff", "")).strip()
                try:
                    kick = pd.to_datetime(kick).strftime("%-I:%M %p")
                except Exception:
                    pass
                _tag = ("" if r.get("bet_tier") == "OFFICIAL"
                        else "Watch \u00b7 ")
                _h.append(
                    f'<div class="sc-row"><div class="sc-rank">{i}</div>'
                    f'<div class="sc-main"><b>{_html.escape(_tag + str(r["pick_label"]))}</b>'
                    f'<small>{_html.escape(str(r["matchup"]))}'
                    f'{" \u00b7 " + _html.escape(kick) if kick else ""}</small></div>'
                    f'<div class="sc-num"><b>{odds:+d}</b>'
                    f'<span>{float(r["cover_prob"]):.0%}</span></div></div>')

        _blk("Spreads", _sp)
        _blk("Totals", _to)
        if _mo:
            _h.append('<div class="sc-grp">Moneyline</div>')
            for i, f in enumerate(_mo[:4], start=1):
                _h.append(
                    f'<div class="sc-row"><div class="sc-rank">{i}</div>'
                    f'<div class="sc-main"><b>{_html.escape(f["pick"])}</b>'
                    f'<small>{_html.escape(f["matchup"])}</small></div>'
                    f'<div class="sc-num"><b>{f["odds"]:+d}</b>'
                    f'<span>{f["prob"]:.0%}</span></div></div>')

        _h.append(
            f'<div class="sc-foot">Sunday Edge \u00b7 every market where the '
            f'model sits {MIN_GAP_PTS:g}+ points off the line, ranked.'
            f'</div></div>')
        st.markdown("".join(_h), unsafe_allow_html=True)

        _bets = (card[card["bet_tier"].isin(["OFFICIAL", "WATCH"])]
                 if not card.empty else card)
        with st.expander("Detail"):
            for _, r in (_sp + _to):
                render_row(r, "Bet" if r.get("bet_tier") == "OFFICIAL"
                           else "Watch")

        _n_off = int((_bets["bet_tier"] == "OFFICIAL").sum()) if len(_bets) else 0
        if not _bets.empty and st.button(
                f"Freeze card ({_n_off} bets, {len(_bets) - _n_off} watch)",
                use_container_width=True):
            tr, n = freeze(_bets, load_tracker())
            st.success(f"Froze {n} new bets." if n else "Nothing new to freeze.")

with tab_game:
    gs = sorted(sched_all["season"].unique())
    d1, d2 = st.columns(2)
    g_season = d1.selectbox("Season", gs, index=len(gs) - 1, key="g_season")
    g_weeks = sorted(sched_all[sched_all["season"] == g_season]["week"].unique())
    g_week = d2.selectbox("Week", g_weeks, index=0, key="g_week")

    wk = sched_all[(sched_all["season"] == g_season)
                   & (sched_all["week"] == g_week)].copy()
    wk["label"] = wk["away_team"] + " @ " + wk["home_team"]
    pick = st.selectbox("Game", wk["label"].tolist())
    row = wk[wk["label"] == pick].iloc[0]

    st.markdown(CARD_CSS, unsafe_allow_html=True)
    rt_g = build_ratings(sched_all, g_season, g_week)
    if rt_g is None:
        st.info("Not enough completed games to build ratings yet.")
    else:
        h, a = row["home_team"], row["away_team"]
        rh, ra = rt_g["margin"].get(h, 0.0), rt_g["margin"].get(a, 0.0)
        hfa = rt_g["hfa"]
        _qbd_g, _qb_note_g = qb_delta(g_season, g_week, h, a)
        _im_g, _it_g, _in_g, _idet_g = injury_delta(g_season, g_week, h, a)
        raw = rh - ra + hfa + _qbd_g + _im_g

        def _say(v, unit):
            """A margin as plain English, so there is no sign to misread."""
            if unit == "total":
                return f"{v:.1f} points"
            fav, dog = h, a
            if v < 0:
                fav, dog = a, h
            if abs(v) < 0.05:
                return "dead even"
            return f"{fav} by {abs(v):.1f}"

        def verdict_block(label, edge, p, e, lean, mkt, model, sd, unit):
            """Answer first. The arithmetic is available but folded away —
            on a phone the derivation was burying the actual call."""
            st.markdown(f'<div class="se-sec">{label}</div>',
                        unsafe_allow_html=True)
            # Same rule and same words as the card. This used to judge on
            # expected value while the card judged on points of disagreement,
            # so a game could be "no bet" here and on the card at the same
            # time.
            _gap = abs(model - mkt)
            if e is not None and e >= MIN_EV:
                st.success(f"**Bet {lean}.**")
            elif _gap >= MIN_GAP_PTS:
                st.warning(f"**Watch {lean}** \u2014 big disagreement, but "
                           f"the edge does not beat the vig.")
            else:
                st.info("**Don't bet.**")
            # Say it in words. The model works in margins (positive = home
            # team ahead) while a book quotes handicaps (KC -2.5 = KC gives
            # 2.5). Same fact, opposite sign, and nothing on screen said
            # which convention you were reading.
            # Three rows. Edge-after-blend and cover probability moved into
            # the expander: they explain the pricing, they do not answer
            # "is this on the card".
            _kv = [("Market says", _say(mkt, unit)),
                   ("Model says", _say(model, unit)),
                   ("Apart", f"{abs(model - mkt):.1f} pts")]
            st.markdown(
                '<table class="se-kv">'
                + "".join(f"<tr><td>{_html.escape(k)}</td>"
                          f"<td>{_html.escape(str(v))}</td></tr>"
                          for k, v in _kv)
                + "</table>", unsafe_allow_html=True)
            with st.expander("Pricing detail"):
                st.markdown(
                    '<table class="se-kv">'
                    f'<tr><td>Edge after blend</td><td>{edge:+.2f} pts</td></tr>'
                    f'<tr><td>Cover probability</td><td>{p:.1%}</td></tr>'
                    f'<tr><td>Value at this price</td><td>{e:+.2%}</td></tr>'
                    '</table>', unsafe_allow_html=True)
                st.write(
                    f"The model line comes from the two power ratings plus "
                    f"home field, then blended toward the market at "
                    f"{MODEL_WEIGHT} \u2014 the weight it earned in backtest:"
                )
                st.code(
                    f"blended fair = {mkt:.2f} + {MODEL_WEIGHT} x "
                    f"({model - mkt:+.2f}) = {mkt + MODEL_WEIGHT*(model-mkt):.2f}\n"
                    f"edge         = {edge:+.2f} pts\n"
                    f"cover prob   = normal({abs(edge):.2f} / {sd}) = {p:.1%}",
                    language=None)

        st.caption(
            f"{h} {rh:+.2f} · {a} {ra:+.2f} · home field {hfa:+.2f} — "
            f"fit on {rt_g['n_prior']:,} games, {rt_g['n_in_season']} of them "
            f"this season ({rt_g['in_season_weight']:.0%} weight)."
            + (f" Adjusted for {_qb_note_g} ({_qbd_g:+.1f})." if _qb_note_g
               else "")
            + (f" Injuries this week ({_in_g}): {_im_g:+.1f} to the home "
               f"margin, {_it_g:+.1f} to the total." if _in_g else "")
            + ((" QBs: " + " \u00b7 ".join(
                   f"{_e['team']} {_qb_last(_e['starter'])} "
                   f"({_e['rating']:+.3f} EPA/db)"
                   + (f", replacing {_qb_last(_e['usual'])} "
                      f"({_e['usual_rating']:+.3f})" if _e['changed']
                      and _e.get('usual_rating') is not None else "")
                   for _e in (_idet_g["qb"]["away"], _idet_g["qb"]["home"])) + ".")
               if _idet_g.get("qb") else "")
            + (f" Ratings fit net of injuries in "
               f"{rt_g.get('n_injury_adjusted', 0)} past games."
               if rt_g.get("n_injury_adjusted") else "")
        )
        if _idet_g and (_idet_g.get("home") or _idet_g.get("away")):
            with st.expander("Injury report used"):
                for _t, _side in ((a, "away"), (h, "home")):
                    _pl = _idet_g.get(_side, {}).get("players", [])
                    if not _pl:
                        st.caption(f"{_t}: nobody of note listed.")
                        continue
                    st.caption(f"{_t}: " + "; ".join(
                        f"{p['player']} ({inj_mod.GROUP_LABEL[p['group']]}, "
                        f"{p['status']}, {p['share']:.0%} of snaps)"
                        for p in _pl))

        if pd.notna(row.get("spread_line")):
            mkt = sign * float(row["spread_line"])
            fair = mkt + MODEL_WEIGHT * (raw - mkt)
            edge = fair - mkt
            # State the actual bet, not just the team: the side plus the
            # number, from that side's perspective.
            lean = (f"{h} {-mkt:+g}" if edge > 0 else f"{a} {mkt:+g}")
            verdict_block("Spread", edge, norm_cdf(abs(edge) / SD_MARGIN),
                          ev_from_prob(norm_cdf(abs(edge) / SD_MARGIN)),
                          lean, mkt, raw, SD_MARGIN, "margin")
        else:
            st.info("No spread posted for this game.")

        if rt_g["total"] and pd.notna(row.get("total_line")):
            th = rt_g["total"].get(h, 0.0); ta = rt_g["total"].get(a, 0.0)
            raw_t = th + ta + rt_g["tbase"] + _it_g
            mt = float(row["total_line"])
            _mph_g = fetch_wind(h, f"{row.get('gameday','')} {row.get('gametime','')}")
            _wadj_g = wind_adjustment(_mph_g)
            edge_t = MODEL_WEIGHT * (raw_t - mt) + _wadj_g
            if _wadj_g:
                st.caption(f"Wind {_mph_g:.0f} mph: {_wadj_g:+.2f} pts on the "
                           f"total after the forecast haircut.")
            lean = f"Over {mt:g}" if edge_t > 0 else f"Under {mt:g}"
            verdict_block("Total", edge_t, norm_cdf(abs(edge_t) / SD_TOTAL),
                          ev_from_prob(norm_cdf(abs(edge_t) / SD_TOTAL)),
                          lean, mt, raw_t, SD_TOTAL, "total")
        else:
            st.info("No total posted for this game.")

        if pd.notna(row.get("home_score")):
            st.caption(f"Final: {a} {row['away_score']:.0f} — "
                       f"{h} {row['home_score']:.0f}")

with tab_tracker:
    tr = load_tracker()
    # Snapshot pregame numbers on every load; after kickoff the last
    # snapshot is the close and is never touched again.
    tr, _ncap = capture_closing(tr, all_offers)
    tr, n = grade(tr, sched_all)
    if n:
        st.success(f"Graded {n} completed bets.")
    if _ncap:
        st.caption(f"Updated the pregame closing snapshot on {_ncap} bet(s).")

    ws, err = _sheet(return_error=True)
    if ws is None:
        st.error(
            "**Not connected to Google Sheets.** Every bet below lives in "
            "session memory only and disappears when this app restarts or "
            "sleeps \u2014 which Streamlit does after a few hours idle."
        )
        if err:
            st.markdown(f"**What to fix:** {err}")
        with st.expander("Set it up", expanded=False):
            st.markdown(SHEETS_SETUP_STEPS)
    else:
        st.caption("Storage: Google Sheets \u2014 history is saved permanently.")

    # THE question: are these picks on the right side more than 52.4% of
    # the time? Reported with its own error bar, because a hit rate from a
    # small number of bets is not an answer.
    _g = tr[tr["result"].isin(["WIN", "LOSS"])] if not tr.empty else tr
    if len(_g):
        _w = int((_g["result"] == "WIN").sum())
        _n = len(_g)
        _rate = _w / _n
        _se = (0.25 / _n) ** 0.5
        st.markdown('<div class="se-sec">RIGHT SIDE, HOW OFTEN?</div>',
                    unsafe_allow_html=True)
        _ci_svg = se_winrate_ci_svg(_w, _n - _w)
        if _ci_svg:
            st.markdown(f'<div class="se-curve">{_ci_svg}</div>',
                        unsafe_allow_html=True)
        _lo, _hi = _rate - 1.96 * _se, _rate + 1.96 * _se
        # Status chip, matching the college app: a one-line verdict rather
        # than three metric cards the reader has to compare themselves.
        if _lo > 0.524:
            _tone, _lab = "pos", "Beating the number"
        elif _hi < 0.524:
            _tone, _lab = "neg", "Below break-even"
        else:
            _tone, _lab = "wait", "Too early to call"
        st.markdown(
            f'<div class="se-verdict {_tone}"><span class="se-verdict-dot"></span>'
            f'<b>{_lab}</b><em>{_n} graded</em></div>',
            unsafe_allow_html=True,
        )
        st.caption(
            f"95% range given {_n} bets: {_lo:.1%} to {_hi:.1%}. "
            + ("Breakeven sits inside that range, so this does not yet "
               "distinguish a real edge from chance."
               if _lo <= 0.524 <= _hi else
               ("Breakeven is below the range \u2014 a real edge at this "
                "sample size." if _lo > 0.524 else
                "Breakeven is above the range \u2014 these picks are losing "
                "by more than variance explains."))
        )
        _need = int(0.25 * (1.96 / max(abs(_rate - 0.524), 0.005)) ** 2)
        st.caption(
            f"To separate a {_rate:.1%} hit rate from breakeven with "
            f"confidence you would need roughly {_need:,} graded bets."
        )

    # Closing line value — the only measurement that can say anything at
    # NFL volume, where a season is ~100 bets.
    _clv = clv_summary(tr)
    if _clv["n"]:
        st.markdown('<div class="se-sec">CLOSING LINE VALUE</div>',
                    unsafe_allow_html=True)
        st.markdown(
            stat_strip([
                (f"{_clv['mean']:+.2f}", "Avg pts vs close",
                 "pos" if _clv["mean"] >= 0 else "neg"),
                (f"{_clv['beat']:.0%}", "Beat the close", ""),
                (f"{_clv['zero']:.0%}", "No movement", ""),
                (f"{_clv['n']}", "Measured", ""),
            ]),
            unsafe_allow_html=True,
        )
        if _clv["n_clean"] < 30:
            st.caption(
                f"{_clv['n_clean']} of {_clv['n']} closes were snapshotted in "
                "the 3 hours before kickoff. Only those are a real measurement — a "
                "number pulled whenever the app happened to run is not a "
                "closing line. Nothing here counts as evidence until roughly "
                "100 clean captures, whatever the sign."
            )
        elif math.isfinite(_clv["t"]):
            st.caption(
                f"Clean closes average {_clv['clean_mean']:+.2f} pts. Signal "
                f"strength {_clv['t']:+.2f} on {_clv['n_clean']} pregame "
                "captures — above +2.00 would be meaningful."
            )

    if tr.empty:
        st.info("No bets frozen yet.")
    else:
        for tier in ["OFFICIAL", "WATCH"]:
            sub = tr[tr["bet_tier"] == tier]
            s = summarize(sub)
            st.markdown(f'<div class="se-sec">{tier} LEDGER</div>',
                         unsafe_allow_html=True)
            st.markdown(
                stat_strip([
                    (f"{s['w']}-{s['l']}-{s['p']}", "W \u00b7 L \u00b7 P", ""),
                    (f"{s['units']:+.2f}u", "Units",
                     "pos" if s["units"] >= 0 else "neg"),
                    (f"{s['roi']:+.1%}" if s["n"] else "\u2014", "ROI",
                     ("pos" if s["roi"] >= 0 else "neg") if s["n"] else ""),
                    (f"{s['n']}", "Graded", ""),
                ]),
                unsafe_allow_html=True,
            )
            _eq = ""
            if s["n"]:
                _gsub = sub[sub["result"].isin(["WIN", "LOSS", "PUSH"])].copy()
                _sort_cols = [c for c in ("season", "week", "kickoff")
                              if c in _gsub.columns]
                if _sort_cols:
                    _gsub = _gsub.sort_values(_sort_cols)
                _eq = se_equity_svg(
                    pd.to_numeric(_gsub.get("units_result"), errors="coerce")
                    .fillna(0.0).tolist()
                )
            if _eq:
                st.markdown(f'<div class="se-curve">{_eq}</div>',
                            unsafe_allow_html=True)
                st.caption("Every graded bet in order, 1 unit flat. "
                           "Nothing reset, nothing hidden.")

            m = pd.to_numeric(sub.get("result_margin"), errors="coerce").dropna()
            if len(m):
                st.caption(f"Average result margin {m.mean():+.2f} pts across "
                           f"{len(m)} graded bets — a continuous read that "
                           f"converges faster than win rate.")

        st.dataframe(tr, hide_index=True, use_container_width=True)
        st.download_button("Download tracker CSV",
                           tr.to_csv(index=False).encode(),
                           "sunday_edge_tracker.csv", "text/csv")

    # Injury model: what is in effect, and (owner only) a way to measure it.
    with st.expander("Injury model", expanded=False):
        _ia = load_injury_adjustment()
        if _ia:
            st.caption(
                f"Measured on {_ia.get('n_games', 0):,} games "
                f"({_ia['seasons'][0]}\u2013{_ia['seasons'][1]}), "
                f"created {str(_ia.get('created_at', ''))[:10]}. A group is "
                f"used only with the expected sign and |t| \u2265 {INJ_MIN_T:g}.")
            _mc = _ia.get("market_check", {}).get("margin", {})
            _tbl = []
            for _gname in inj_mod.GROUPS:
                _c = _ia["margin"].get(_gname, {})
                _tbl.append({
                    "Group": inj_mod.GROUP_LABEL[_gname],
                    "Pts per starter out": _c.get("coef"),
                    "t": _c.get("t"),
                    "Used": bool(inj_mod.usable(_ia, "margin", _gname, -1,
                                                INJ_MIN_T)),
                    "vs closing line (t)": _mc.get(_gname, {}).get("t"),
                })
            st.dataframe(pd.DataFrame(_tbl), hide_index=True,
                         use_container_width=True)
            _mp = _ia.get("miss_prob", {})
            st.caption("How often each status actually sat: " + ", ".join(
                f"{k} {v:.0%}" for k, v in _mp.items()))
            _q = _ia.get("qb")
            if _q:
                _qm = _q["margin"]
                st.caption(
                    f"Quarterbacks: each 0.10 EPA/dropback of QB quality is "
                    f"worth {_qm['coef'] * 0.10:.1f} pts on the spread "
                    f"(t = {_qm['t']:.1f}, from {_q['n_games']:,} games, blend "
                    f"fit on {_q['n_qb_seasons']} QB-seasons) \u2014 "
                    + ("in use." if qb_model_on() else
                       "did not clear the bar, so the flat QB penalty is used."))
            elif _ia.get("qb_error"):
                st.caption(f"Quarterback ratings not available "
                           f"({_ia['qb_error'][:120]}); the flat QB penalty is "
                           f"used instead.")
            st.caption("'vs closing line' near zero means the market already "
                       "prices these injuries: the adjustment then removes "
                       "false edges rather than creating real ones.")
        else:
            st.caption("Not measured \u2014 no injury adjustment is being "
                       "applied. "
                       + str(st.session_state.get("inj_auto_failed") or ""))
        if _ia:
            if _ia.get("source") == "repo file":
                st.caption("Source: injury_adjustment.json in the repo "
                           "(pinned). Delete that file to go automatic.")
            else:
                st.caption(
                    f"Automatic \u2014 refreshes itself every "
                    f"{INJ_REFRESH_DAYS} days"
                    + (", saved in your Google Sheet." if _ia.get("saved_to_sheet")
                       else " (not saved to the Sheet, so it re-measures "
                            "whenever the app wakes up)."))
        if is_owner() and st.button("Re-measure now", key="inj_remeasure"):
            # Stale the saved copy so the next load really re-measures.
            _ws_i = _injury_ws()
            if _ws_i is not None:
                try:
                    _ws_i.update_acell("A1", "")
                except Exception:
                    pass
            st.session_state["inj_stamp"] = \
                st.session_state.get("inj_stamp", 0) + 1
            st.session_state.pop("inj_auto_failed", None)
            st.rerun()

st.divider()
st.markdown(
    '<div style="text-align:center;padding:8px 0 4px">'
    '<div style="font-size:.62rem;letter-spacing:.2em;font-weight:900;'
    'color:#61748C">SUNDAY <span style="color:#60A5FA">EDGE</span></div>'
    '<div style="font-size:.66rem;color:#61748C;margin-top:7px;'
    'line-height:1.6;max-width:34rem;margin-left:auto;margin-right:auto">'
    'For entertainment and research. Backtested on 4,254 games this model '
    'did NOT beat the closing line (coefficient +0.099, t = 1.19) \u2014 no '
    'edge is claimed. 21+ where legal. If gambling stops being fun, call '
    '1-800-GAMBLER or text 800GAM.'
    '</div></div>',
    unsafe_allow_html=True,
)
