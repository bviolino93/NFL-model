"""
Sunday Edge — injury impact (non-quarterback).

The power rating is a ridge on final margins. It has no idea who is playing,
so when a team's left tackle and top two corners are out it keeps the value
they earned, and it "disagrees" with a line that already moved on the news.
That is not edge; it is the model missing something everyone else knows.

This module does two jobs, and follows the same rule as the QB and wind
adjustments: nothing is applied until it has been MEASURED.

1. team_injury_loads(): for each team this week, how much of its usual
   lineup is on the injury report, by position group. A player's weight is
   his share of the team's snaps so far (a 95%-snap tackle counts ~0.95, a
   rotational end ~0.3) times the chance his listed status means he sits.

2. measure(): what that load is worth in points, estimated on every game
   since 2013, and whether the market already prices it. Writes
   injury_adjustment.json. Absent that file the app applies nothing.

Quarterbacks are excluded here on purpose. The QB module already handles
them, and counting them twice would double the adjustment.

Known blind spots, stated rather than hidden:
  * Players on injured reserve are not on the weekly report, so a season-
    long absence is invisible here. Most of it is already in the in-season
    ratings, which is why long absences are also down-weighted below.
  * Rookies and players with no snaps yet have no share, so they count 0.
  * The report is Friday's; Sunday inactives (90 min before kickoff) can
    still change things. Questionable players are priced at the rate they
    actually sat historically, not assumed in or out.
"""

import json
import math
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# Position groups. Specialists and quarterbacks are deliberately absent.
GROUPS = {
    "OL":    {"T", "OT", "G", "OG", "C", "OL", "LT", "RT", "LG", "RG"},
    "SKILL": {"WR", "TE", "RB", "FB", "HB"},
    "FRONT": {"DE", "DT", "NT", "DL", "LB", "ILB", "OLB", "MLB", "EDGE"},
    "DB":    {"CB", "S", "FS", "SS", "DB", "SAF"},
}
OFFENSE = ("OL", "SKILL")
DEFENSE = ("FRONT", "DB")
GROUP_LABEL = {"OL": "offensive line", "SKILL": "WR/TE/RB",
               "FRONT": "front seven", "DB": "secondary"}

# Chance each status means the player sits. Used ONLY until measure() has run;
# the measured rates in injury_adjustment.json replace these.
DEFAULT_MISS_PROB = {"out": 1.0, "doubtful": 0.9, "questionable": 0.25}

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
    share = np.where(grp.isin(OFFENSE), off, dfn)
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
    Each player's share of his team's snaps in the games BEFORE this week,
    blended with last season's share early on.

    Dividing by the team's games (not the player's) is deliberate: a starter
    who has missed the last month has already been out of the games the
    in-season rating was fit on, so he should move this week's number less.
    """
    cur = snaps[(snaps["season"] == season) & (snaps["week"] < week)]
    prev = snaps[snaps["season"] == season - 1]

    def _shares(df):
        if df.empty:
            return pd.Series(dtype=float), pd.Series(dtype=float)
        n_team = df.groupby("team")["game_id"].nunique()
        last_team = (df.sort_values("week").groupby("key")["team"].last())
        tot = df.groupby("key")["share"].sum()
        n = last_team.map(n_team).reindex(tot.index).fillna(1.0).clip(lower=1)
        return tot, n

    s_cur, n_cur = _shares(cur)
    s_prev, n_prev = _shares(prev)
    keys = s_cur.index.union(s_prev.index)
    s_cur, n_cur = s_cur.reindex(keys).fillna(0.0), n_cur.reindex(keys).fillna(0.0)
    prev_rate = (s_prev / n_prev).reindex(keys).fillna(0.0).clip(0, 1)
    # Team games this season, for players who have not appeared yet.
    if not cur.empty:
        team_n = cur.groupby("team")["game_id"].nunique()
        n_cur = n_cur.where(n_cur > 0, team_n.median())
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


def history_offsets(adj, table, games, min_t=2.0, cap=6.0):
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
    dm = sum(usable(adj, "margin", c, -1, min_t)
             * (g[f"home_{c}"] - g[f"away_{c}"]) for c in GROUPS)
    off = sum(g[f"home_{c}"] + g[f"away_{c}"] for c in OFFENSE)
    dfn = sum(g[f"home_{c}"] + g[f"away_{c}"] for c in DEFENSE)
    dt = (usable(adj, "total", "OFF", -1, min_t) * off
          + usable(adj, "total", "DEF", +1, min_t) * dfn)
    return (pd.Series(np.clip(dm, -cap, cap), index=games.index),
            pd.Series(np.clip(dt, -cap, cap), index=games.index))


def game_deltas(adj, loads, home, away, min_t=2.0, cap=6.0):
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
        coef = usable(adj, "margin", g, -1, min_t)
        dm += coef * (lh.get(g, 0.0) - la.get(g, 0.0))

    off = sum(lh.get(g, 0.0) + la.get(g, 0.0) for g in OFFENSE)
    dfn = sum(lh.get(g, 0.0) + la.get(g, 0.0) for g in DEFENSE)
    dt = (usable(adj, "total", "OFF", -1, min_t) * off
          + usable(adj, "total", "DEF", +1, min_t) * dfn)

    # Safety rail, not a measurement: a pile of listed players on one side
    # should never move the number more than a starting quarterback does.
    dm = float(np.clip(dm, -cap, cap))
    dt = float(np.clip(dt, -cap, cap))

    parts = []
    for team, l in ((away, la), (home, lh)):
        names = [_last(p["player"]) for p in l.get("players", [])[:3]]
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
    Xt = np.hstack([off[:, None], dfn[:, None], ones, D_t])
    bt, set_, sdt = _fit(Xt, g["total"].values.astype(float), 3)

    out = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seasons": [seasons[0], seasons[-1]],
        "n_games": n,
        "miss_prob": miss_prob,
        "status_n": status_n,
        "margin": {c: _stat(bm[i], sem[i]) for i, c in enumerate(grp)},
        "total": {"OFF": _stat(bt[0], set_[0]), "DEF": _stat(bt[1], set_[1])},
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
            b, se = _ols(np.column_stack([o, f, np.ones(len(d))]), err)
            mc["total"] = {"OFF": _stat(b[0], se[0]), "DEF": _stat(b[1], se[1])}
    out["market_check"] = mc
    log(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    res = measure()
    with open("injury_adjustment.json", "w") as fh:
        json.dump(res, fh, indent=2)
    print("Wrote injury_adjustment.json")
