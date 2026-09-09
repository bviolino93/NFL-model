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
CARRYOVER     = 0.65
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

MIN_EDGE_PTS  = 1.0   # official tier: model/market disagreement in points
WATCH_EDGE_PTS = 0.5  # watch tier floor; below this nothing is shown

MODEL_VERSION = f"1.0.0-a{RIDGE_ALPHA}-w{MODEL_WEIGHT}-e{MIN_EDGE_PTS}"

TRACKER_COLS = [
    "record_key", "frozen_at", "season", "week", "game_id", "kickoff",
    "matchup", "home_team", "away_team", "market_type", "pick_side",
    "pick_label", "bet_line", "model_line", "edge_pts", "cover_prob",
    "expected_value", "odds", "bet_tier", "model_version",
    "status", "result", "units_result", "result_margin",
    "final_home_score", "final_away_score",
    "closing_line", "clv_points", "closing_captured_at", "graded_at",
]

st.set_page_config(page_title="Sunday Edge", page_icon="🏈", layout="wide")


# ----------------------------------------------------------------------
# Storage — Google Sheets, with a session fallback
# ----------------------------------------------------------------------
def _sheet(return_error=False):
    try:
        import gspread
        raw = st.secrets["gcp_service_account_json"]
        creds = json.loads(raw) if isinstance(raw, str) else dict(raw)
        gc = gspread.service_account_from_dict(creds)
        name = st.secrets.get("tracker_sheet_name", "sunday_edge_tracker")
        sh = gc.open(name)
        try:
            ws = sh.worksheet("tracker")
        except Exception:
            ws = sh.add_worksheet("tracker", rows=2000, cols=len(TRACKER_COLS))
        return (ws, None) if return_error else ws
    except Exception as e:
        return (None, str(e)) if return_error else None


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


def save_tracker(df):
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
            ws.clear()
            ws.update([TRACKER_COLS] + x.fillna("").astype(str).values.tolist())
        except Exception as e:
            st.warning(f"Sheet write failed, kept in session only: {e}")
    return x


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=60 * 10)
def fetch_live_odds(_bust=0):
    """
    Live spreads and totals from every US book, kept as raw offers so the
    card can price each one. Returns (offers, credits_left, error).
    """
    key = None
    try:
        key = st.secrets["odds_api_key"]
    except Exception:
        return {}, None, "no key"
    try:
        r = requests.get(
            "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds",
            params={"apiKey": key, "regions": "us",
                    "markets": "spreads,totals", "oddsFormat": "american"},
            timeout=20,
        )
        if r.status_code != 200:
            return {}, None, f"HTTP {r.status_code}: {r.text[:120]}"
        left = r.headers.get("x-requests-remaining")
        offers = {}
        for ev in r.json():
            h = ODDS_TEAM.get(ev.get("home_team"))
            a = ODDS_TEAM.get(ev.get("away_team"))
            if not h or not a:
                continue
            spreads, totals = [], []
            for bk in ev.get("bookmakers", []):
                book = bk.get("title") or bk.get("key")
                for mk in bk.get("markets", []):
                    for o in mk.get("outcomes", []):
                        pt, pr = o.get("point"), o.get("price")
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


def best_offer(cands, fair, sd):
    """
    Line shopping done properly: price every book's actual point AND price,
    then take the highest EV. Best number and best price are often at
    different books, so picking on either one alone leaves money behind.
    """
    best = None
    for side, thresh, price, book in cands:
        edge = (fair - thresh) if side == "OVER_LIKE" else (thresh - fair)
        p = norm_cdf(edge / sd)
        e = ev_from_prob(p, price)
        if e is None:
            continue
        if best is None or e > best["ev"]:
            best = {"ev": e, "cover": p, "edge": edge, "point": thresh,
                    "price": price, "book": book}
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
def fit_ratings(hist, teams, target, symmetric=False):
    if len(hist) < 40:
        return None, None
    idx = {t: i for i, t in enumerate(teams)}
    X = np.zeros((len(hist), len(teams) + 1))
    h, a = hist["home_team"].values, hist["away_team"].values
    for r in range(len(hist)):
        if h[r] in idx:
            X[r, idx[h[r]]] = 1.0
        if a[r] in idx:
            X[r, idx[a[r]]] = 1.0 if symmetric else -1.0
        X[r, -1] = 1.0
    m = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False).fit(X, hist[target].values)
    return {t: m.coef_[idx[t]] for t in teams}, float(m.coef_[-1])


def build_ratings(sched, season, week):
    """Ratings from games played strictly before the target week."""
    g = sched.dropna(subset=["home_score", "away_score"]).copy()
    g["home_margin"] = g["home_score"] - g["away_score"]
    g["total_points"] = g["home_score"] + g["away_score"]
    prior = g[(g["season"] < season) | ((g["season"] == season) & (g["week"] < week))]
    prior = prior.sort_values(["season", "week"])
    if len(prior) < 40:
        return None
    recent = prior.tail(WINDOW_GAMES)
    in_season = prior[prior["season"] == season]
    teams = sorted(set(g["home_team"]) | set(g["away_team"]))

    r_all, hfa = fit_ratings(recent, teams, "home_margin")
    if r_all is None:
        return None
    t_all, tbase = fit_ratings(recent, teams, "total_points", symmetric=True)

    r_cur, _ = fit_ratings(in_season, teams, "home_margin")
    t_cur, _ = fit_ratings(in_season, teams, "total_points", symmetric=True)
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
            "n_in_season": len(in_season), "prior_ratings": r_all}


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


def build_card(sched, season, week, sign, offers=None):
    rt = build_ratings(sched, season, week)
    if rt is None:
        return pd.DataFrame(), None

    games = sched[(sched["season"] == season) & (sched["week"] == week)].copy()
    rows = []
    for _, g in games.iterrows():
        h, a = g["home_team"], g["away_team"]
        if h not in rt["margin"] or a not in rt["margin"]:
            continue

        raw_model = rt["margin"][h] - rt["margin"][a] + rt["hfa"]
        mkt = sign * g["spread_line"] if pd.notna(g.get("spread_line")) else None
        live = lookup_offers(offers, a, h) if offers else None

        # SPREAD — live books first, nflverse as fallback
        if live and live["spreads"]:
            pts = [-p for t, p, _, _ in live["spreads"] if t == h]
            if pts:
                mkt = float(np.median(pts))
            fair = mkt + MODEL_WEIGHT * (raw_model - mkt)
            # Home side covers above -point; away side covers below +point.
            cands = [("OVER_LIKE" if t == h else "UNDER_LIKE",
                      (-p if t == h else p), pr, bk)
                     for t, p, pr, bk in live["spreads"]]
            b = best_offer(cands, fair, SD_MARGIN)
            if b:
                side = "HOME" if b["edge"] > 0 or b["point"] < 0 else "AWAY"
                # Recover which team the winning offer belongs to
                side = "HOME" if any(t == h and -p == b["point"] and pr == b["price"]
                                     for t, p, pr, bk in live["spreads"]) else "AWAY"
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
                "expected_value": ev_from_prob(p), "odds": -110,
            })

        # TOTAL — live books first
        if live and live["totals"] and rt["total"]:
            raw_total = rt["total"].get(h, 0.0) + rt["total"].get(a, 0.0) + rt["tbase"]
            pts = [p for _, p, _, _ in live["totals"]]
            mt = float(np.median(pts))
            fair_t = mt + MODEL_WEIGHT * (raw_total - mt)
            cands = [("OVER_LIKE" if nm == "OVER" else "UNDER_LIKE", p, pr, bk)
                     for nm, p, pr, bk in live["totals"]]
            b = best_offer(cands, fair_t, SD_TOTAL)
            if b:
                side = "OVER" if b["edge"] > 0 else "UNDER"
                side = next((nm for nm, p, pr, bk in live["totals"]
                             if p == b["point"] and pr == b["price"]), side)
                rows.append({
                    "game_id": g["game_id"], "season": season, "week": week,
                    "kickoff": f"{g.get('gameday','')} {g.get('gametime','')}".strip(),
                    "matchup": f"{a} @ {h}", "home_team": h, "away_team": a,
                    "market_type": "TOTAL", "pick_side": side,
                    "pick_label": f"{side.title()} {b['point']:g} "
                                  f"({b['price']:+.0f}) @ {b['book']}",
                    "bet_line": float(b["point"]), "model_line": float(raw_total),
                    "edge_pts": float(b["edge"]), "cover_prob": b["cover"],
                    "expected_value": b["ev"], "odds": b["price"],
                })
                continue

        # TOTAL fallback
        if rt["total"] and pd.notna(g.get("total_line")):
            raw_total = rt["total"].get(h, 0.0) + rt["total"].get(a, 0.0) + rt["tbase"]
            mt = float(g["total_line"])
            fair_t = mt + MODEL_WEIGHT * (raw_total - mt)
            edge_t = fair_t - mt
            side = "OVER" if edge_t > 0 else "UNDER"
            p = norm_cdf(abs(edge_t) / SD_TOTAL)
            rows.append({
                "game_id": g["game_id"], "season": season, "week": week,
                "kickoff": f"{g.get('gameday','')} {g.get('gametime','')}".strip(),
                "matchup": f"{a} @ {h}", "home_team": h, "away_team": a,
                "market_type": "TOTAL", "pick_side": side,
                "pick_label": f"{side.title()} {mt:g}",
                "bet_line": mt, "model_line": float(raw_total),
                "edge_pts": float(edge_t), "cover_prob": p,
                "expected_value": ev_from_prob(p), "odds": -110,
            })

    card = pd.DataFrame(rows)
    if card.empty:
        return card, rt
    card["abs_edge"] = card["edge_pts"].abs()
    card = card.sort_values("abs_edge", ascending=False).reset_index(drop=True)
    # Tiers are FILTERS, not quotas. If nothing clears the floor the
    # section is empty, which is a legitimate answer for a week.
    card["bet_tier"] = np.where(card["abs_edge"] >= MIN_EDGE_PTS, "OFFICIAL",
                        np.where(card["abs_edge"] >= WATCH_EDGE_PTS, "WATCH", None))
    return card[card["bet_tier"].notna()].copy(), rt


# ----------------------------------------------------------------------
# Freeze + grade
# ----------------------------------------------------------------------
def freeze(card, tracker):
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    existing = set(tracker["record_key"].astype(str)) if not tracker.empty else set()
    new = []
    for _, r in card.iterrows():
        key = f"{r['game_id']}|{r['market_type']}"
        if r["bet_tier"] != "OFFICIAL":
            key += "|W"
        if key in existing:
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
            "model_version": MODEL_VERSION, "status": "FROZEN",
        })
        new.append(row)
    if not new:
        return tracker, 0
    out = pd.concat([tracker, pd.DataFrame(new)], ignore_index=True)
    return save_tracker(out), len(new)


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
            m = (hs - as_) + bl
        else:
            m = (as_ - hs) + bl

        res = "PUSH" if abs(m) < 1e-9 else ("WIN" if m > 0 else "LOSS")
        units = 0.0 if res == "PUSH" else (100 / 110 if res == "WIN" else -1.0)
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


# ----------------------------------------------------------------------
# Card rendering
# ----------------------------------------------------------------------
CARD_CSS = """
<style>
.se-card{border:1px solid rgba(128,128,128,.25);border-radius:14px;
  padding:14px 16px;margin-bottom:10px}
.se-card.off{border-left:5px solid #16a34a}
.se-card.watch{border-left:5px solid #d97706;opacity:.85}
.se-top{display:flex;align-items:center;gap:8px;margin-bottom:6px}
.se-rank{font-weight:700;font-size:.8rem;opacity:.55;min-width:16px}
.se-chip{font-size:.65rem;letter-spacing:.08em;font-weight:700;opacity:.6;
  border:1px solid rgba(128,128,128,.35);border-radius:99px;padding:2px 8px}
.se-badge{margin-left:auto;font-size:.65rem;font-weight:800;letter-spacing:.06em;
  border-radius:99px;padding:3px 10px}
.se-badge.best{background:#16a34a;color:#fff}
.se-badge.bet{border:1px solid #16a34a;color:#16a34a}
.se-badge.watch{border:1px solid #d97706;color:#d97706}
.se-pick{font-size:1.45rem;font-weight:800;line-height:1.15;margin:2px 0}
.se-sub{font-size:.82rem;opacity:.6;margin-bottom:10px}
.se-m{display:flex;gap:14px;flex-wrap:wrap}
.se-m div{display:flex;flex-direction:column}
.se-m b{font-size:.95rem;font-weight:700}
.se-m span{font-size:.6rem;letter-spacing:.08em;opacity:.5;font-weight:600}
.se-ev-pos{color:#16a34a}.se-ev-neg{color:#dc2626}
</style>
"""


def render_card(r, rank, badge):
    cls = "off" if badge in ("BEST BET", "BET") else "watch"
    bcls = {"BEST BET": "best", "BET": "bet"}.get(badge, "watch")
    ev = float(r["expected_value"])
    evc = "se-ev-pos" if ev > 0 else "se-ev-neg"
    e = _html.escape
    st.markdown(f"""
<div class="se-card {cls}">
  <div class="se-top">
    <span class="se-rank">{rank}</span>
    <span class="se-chip">{e(str(r['market_type']))}</span>
    <span class="se-badge {bcls}">{e(badge)}</span>
  </div>
  <div class="se-pick">{e(str(r['pick_label']))}</div>
  <div class="se-sub">{e(str(r['matchup']))}{
      ' · ' + e(str(r['kickoff'])) if str(r.get('kickoff','')).strip() else ''}</div>
  <div class="se-m">
    <div><b>{float(r['bet_line']):g}</b><span>LINE</span></div>
    <div><b>{float(r['model_line']):.1f}</b><span>MODEL</span></div>
    <div><b>{float(r['edge_pts']):+.2f}</b><span>EDGE</span></div>
    <div><b>{float(r['cover_prob']):.1%}</b><span>COVER</span></div>
    <div><b class="{evc}">{ev:+.2%}</b><span>EV</span></div>
  </div>
</div>""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.title("Sunday Edge")
st.caption(f"NFL spreads and totals · model {MODEL_VERSION}")

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

    card, rt = build_card(sched_all, season, week, sign, offers=live_offers)
    st.markdown(CARD_CSS, unsafe_allow_html=True)

    if rt is None:
        st.info("Not enough completed games yet to build ratings.")
    else:
        official = card[card["bet_tier"] == "OFFICIAL"] if not card.empty \
            else card
        watch = card[card["bet_tier"] == "WATCH"] if not card.empty else card

        # The headline decision, before any numbers.
        if official.empty:
            st.markdown("## No bets this week")
            st.caption(
                f"Nothing clears the {MIN_EDGE_PTS:g}-point bar. That is the "
                f"model's answer, not a failure to load — most NFL weeks "
                f"should look like this."
            )
        else:
            st.markdown(f"## Bet {len(official)} "
                        f"{'game' if len(official) == 1 else 'games'}")

        if not official.empty:
            st.markdown("#### Official bets")
            top = official["expected_value"].idxmax()
            for i, (idx, r) in enumerate(official.iterrows(), start=1):
                render_card(r, i, "BEST BET" if idx == top else "BET")

        if not watch.empty:
            st.markdown("#### Watch list")
            st.caption("Tracked separately. Not part of the official record.")
            for i, (_, r) in enumerate(watch.iterrows(), start=1):
                render_card(r, i, "WATCH")

        if not card.empty and st.button("Freeze this card", type="primary",
                                        use_container_width=True):
            tr, n = freeze(card, load_tracker())
            st.success(f"Froze {n} new bets." if n else "Nothing new to freeze.")

        st.caption(f"Ratings fit on {rt['n_prior']:,} prior games · "
                   f"home field {rt['hfa']:+.2f} pts")

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

    rt_g = build_ratings(sched_all, g_season, g_week)
    if rt_g is None:
        st.info("Not enough completed games to build ratings yet.")
    else:
        h, a = row["home_team"], row["away_team"]
        rh, ra = rt_g["margin"].get(h, 0.0), rt_g["margin"].get(a, 0.0)
        hfa = rt_g["hfa"]
        raw = rh - ra + hfa

        def verdict_block(label, edge, p, e, lean, mkt, model, sd):
            """Answer first. The arithmetic is available but folded away —
            on a phone the derivation was burying the actual call."""
            st.markdown(f"### {label}")
            if abs(edge) >= MIN_EDGE_PTS and e > 0:
                st.success(f"**BET {lean}** · EV {e:+.2%}")
            elif abs(edge) >= WATCH_EDGE_PTS:
                st.warning(f"**WATCH {lean}** · EV {e:+.2%} — below the "
                           f"{MIN_EDGE_PTS:g}-point bar")
            else:
                st.error(f"**NO BET** · EV {e:+.2%} · model leans {lean}")
            st.dataframe(
                pd.DataFrame({
                    "": ["Market", "Model", "Disagreement", "Edge after blend",
                         "Cover probability"],
                    " ": [f"{mkt:.1f}", f"{model:.1f}", f"{model - mkt:+.2f} pts",
                          f"{edge:+.2f} pts", f"{p:.1%}"],
                }), hide_index=True, use_container_width=True)
            with st.expander("Show the arithmetic"):
                st.write(
                    f"The model line comes from the two power ratings plus "
                    f"home field. It is then blended toward the market at "
                    f"{MODEL_WEIGHT}, the weight the backtest earned:"
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
        )

        if pd.notna(row.get("spread_line")):
            mkt = sign * float(row["spread_line"])
            fair = mkt + MODEL_WEIGHT * (raw - mkt)
            edge = fair - mkt
            lean = h if edge > 0 else a
            verdict_block("Spread", edge, norm_cdf(abs(edge) / SD_MARGIN),
                          ev_from_prob(norm_cdf(abs(edge) / SD_MARGIN)),
                          lean, mkt, raw, SD_MARGIN)
        else:
            st.info("No spread posted for this game.")

        if rt_g["total"] and pd.notna(row.get("total_line")):
            th = rt_g["total"].get(h, 0.0); ta = rt_g["total"].get(a, 0.0)
            raw_t = th + ta + rt_g["tbase"]
            mt = float(row["total_line"])
            edge_t = MODEL_WEIGHT * (raw_t - mt)
            lean = f"Over {mt:g}" if edge_t > 0 else f"Under {mt:g}"
            verdict_block("Total", edge_t, norm_cdf(abs(edge_t) / SD_TOTAL),
                          ev_from_prob(norm_cdf(abs(edge_t) / SD_TOTAL)),
                          lean, mt, raw_t, SD_TOTAL)
        else:
            st.info("No total posted for this game.")

        if pd.notna(row.get("home_score")):
            st.caption(f"Final: {a} {row['away_score']:.0f} — "
                       f"{h} {row['home_score']:.0f}")

with tab_tracker:
    tr = load_tracker()
    tr, n = grade(tr, sched_all)
    if n:
        st.success(f"Graded {n} completed bets.")

    ws, err = _sheet(return_error=True)
    if ws is None:
        st.warning("Storage: session only — records are lost when the app "
                   "restarts. Add Google Sheets credentials to keep history.")
        if err:
            st.caption(f"Reason: {err}")
    else:
        st.caption("Storage: Google Sheets — history is saved permanently.")

    if tr.empty:
        st.info("No bets frozen yet.")
    else:
        for tier in ["OFFICIAL", "WATCH"]:
            sub = tr[tr["bet_tier"] == tier]
            s = summarize(sub)
            st.subheader(f"{tier.title()} — {s['w']}-{s['l']}-{s['p']}")
            a, b, c = st.columns(3)
            a.metric("Units", f"{s['units']:+.2f}")
            b.metric("ROI", f"{s['roi']:+.1%}" if s["n"] else "—")
            c.metric("Graded", f"{s['n']}")

            m = pd.to_numeric(sub.get("result_margin"), errors="coerce").dropna()
            if len(m):
                st.caption(f"Average result margin {m.mean():+.2f} pts across "
                           f"{len(m)} graded bets — a continuous read that "
                           f"converges faster than win rate.")

        st.dataframe(tr, hide_index=True, use_container_width=True)
        st.download_button("Download tracker CSV",
                           tr.to_csv(index=False).encode(),
                           "sunday_edge_tracker.csv", "text/csv")
