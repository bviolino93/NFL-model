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

# How far the model must sit from the line for a market to make the card,
# in points of raw disagreement (before the 0.099 blend).
#
# Set from the real distribution: median disagreement is 2.2 pts, the 75th
# percentile 3.9, the 90th 5.5. At 4 points roughly 24% of markets qualify,
# which is about seven plays on a full Sunday, and the chance of a week
# producing nothing is under 2%. At 5 it drops to four plays and one Sunday
# in ten comes up empty; at 3 it is eleven plays, most of the board.
MIN_GAP_PTS = 4.0


MODEL_VERSION_BASE = f"1.1.0-a{RIDGE_ALPHA}-w{MODEL_WEIGHT}"


def model_version():
    """Threshold is part of the version: change the bar and the record it
    produces is no longer comparable with what came before."""
    return f"{MODEL_VERSION_BASE}-ev{MIN_EV:g}"

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
                                mls.append((_t, float(pr)))
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
def fit_ratings(hist, teams, target, symmetric=False):
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
    if len(hist) < 40:
        return None, None
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

    m = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False).fit(X, y0)
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
                "expected_value": ev_from_prob(p), "odds": -110,
            })

        # TOTAL — live books first
        if live and live["totals"] and rt["total"]:
            raw_total = rt["total"].get(h, 0.0) + rt["total"].get(a, 0.0) + rt["tbase"]
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
    # The card ranks by how far the model is from the line. EV is still
    # computed and shown on every row, but it is no longer a gate: a card
    # that says "no bets" most weeks does not answer the question this app
    # exists to answer, which is whether these picks land on the right side
    # more than 52.4% of the time. That gets settled by the record, not by
    # a threshold.
    card["bet_tier"] = "OFFICIAL"
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
            "model_version": model_version(), "status": "FROZEN",
        })
        new.append(row)
    if not new:
        return tracker, 0
    out = pd.concat([tracker, pd.DataFrame(new)], ignore_index=True)
    return save_tracker(out), len(new)


def capture_closing(tracker, offers, sched=None):
    """
    Record the market number at kickoff, so CLV means something.

    The docstring at the top of this file claimed "real closing-line
    capture" as a lesson carried over from the college app. It was not
    implemented: closing_line, clv_points and closing_captured_at existed in
    TRACKER_COLS and nothing ever wrote them.

    That matters more here than for college. The NFL backtest measured
    +0.099 with t = 1.19 over 4,254 games, and at ~100 bets a season the
    win-loss record will never settle anything. Closing line value is the
    only thing that can answer the question inside one season.

    Two rules learned the hard way from Saturday Edge:

      * Capture at KICKOFF, not at grading. A number pulled whenever the app
        next happens to run is not a closing line, and treating it as one
        produced a statistically significant CLV that turned out to be an
        artifact of when the page was opened.

      * Stamp WHEN. Without the timestamp there is no way to tell a clean
        capture from a stale one, so the lag is recorded and the display
        buckets on it.

    Only rows whose kickoff has passed and that have no closing line yet are
    touched; a captured line is never rewritten.
    """
    if tracker is None or tracker.empty or not offers:
        return tracker, 0
    df = tracker.copy()
    now = datetime.now(timezone.utc)

    cl = pd.to_numeric(df.get("closing_line"), errors="coerce")
    todo = cl.isna()
    if not todo.any():
        return df, 0

    n = 0
    for idx in df[todo].index:
        kick = pd.to_datetime(df.at[idx, "kickoff"], errors="coerce", utc=True)
        if pd.isna(kick) or kick > now:
            continue
        off = lookup_offers(offers, str(df.at[idx, "away_team"]),
                            str(df.at[idx, "home_team"]))
        if not off:
            continue
        mt = str(df.at[idx, "market_type"]).upper()
        side = str(df.at[idx, "pick_side"]).upper()
        home = str(df.at[idx, "home_team"])
        # fetch_live_odds stores spreads as (team, point, price, book) and
        # totals as (OVER|UNDER, point, price, book). Consensus close is the
        # median point across books, stated from the HOME side for spreads so
        # it is on the same footing as the frozen bet_line.
        pts = []
        if mt == "TOTAL":
            for nm, pt, _pr, _bk in off.get("totals", []) or []:
                if math.isfinite(float(pt)):
                    pts.append(float(pt))
        else:
            # The book states a home favourite as -4.5; bet_line is stored in
            # nflverse convention, +4.5. Negate, exactly as build_card does,
            # so the close and the frozen line are on the same scale.
            for team, pt, _pr, _bk in off.get("spreads", []) or []:
                if str(team) == home and math.isfinite(float(pt)):
                    pts.append(-float(pt))
        if not pts:
            continue
        close = float(np.median(pts))

        try:
            bl = float(df.at[idx, "bet_line"])
        except (TypeError, ValueError):
            continue
        # Positive CLV means the number moved in the bet's favour. Both
        # numbers are in nflverse convention here: + means home favoured by
        # that much.
        #
        # HOME backer LAYS the number, so they want to have taken a smaller
        # one than the close: took home -3, it closed -4.25, that is +1.25.
        # AWAY backer RECEIVES the number, so the reverse: took +3 when it
        # closed +4.25 means they got less than they could have, -1.25.
        if mt == "TOTAL":
            clv = (close - bl) if side == "OVER" else (bl - close)
        elif side == "HOME":
            clv = close - bl
        else:
            clv = bl - close

        df.at[idx, "closing_line"] = round(close, 2)
        df.at[idx, "clv_points"] = round(float(clv), 2)
        df.at[idx, "closing_captured_at"] = now.isoformat(timespec="seconds")
        n += 1
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

    kick = pd.to_datetime(df.loc[ok, "kickoff"], errors="coerce", utc=True)
    cap = pd.to_datetime(df.loc[ok, "closing_captured_at"],
                         errors="coerce", utc=True)
    lag_h = (cap - kick).dt.total_seconds() / 3600.0
    clean = lag_h.notna() & (lag_h <= 3.0)
    out["n_clean"] = int(clean.sum())
    if out["n_clean"]:
        out["clean_mean"] = float(cc[clean.values].mean())
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
        raw = rt["margin"][h] - rt["margin"][a] + rt["hfa"]
        p_home = norm_cdf(raw / SD_MARGIN)
        for team, prob in ((h, p_home), (a, 1.0 - p_home)):
            price = None
            for side, pt, pr, bk in live.get("spreads", []):
                pass
            mls = live.get("moneylines") or []
            for nm, pr in mls:
                if nm == team:
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
[data-testid="stTabs"] [data-baseweb="tab-list"]{
  display:grid!important;grid-template-columns:repeat(3,minmax(0,1fr))!important;
  gap:4px!important;padding:4px!important;margin:2px 0 14px!important;
  border-radius:14px!important;background:#0A1628!important;
  border:1px solid var(--line)!important;
}
[data-testid="stTabs"] [data-baseweb="tab-list"]::before,
[data-testid="stTabs"] [data-baseweb="tab-highlight"],
[data-testid="stTabs"] [data-baseweb="tab-border"]{display:none!important}
[data-testid="stTabs"] button[data-baseweb="tab"]{
  width:100%!important;min-height:38px!important;padding:9px 2px!important;
  margin:0!important;border-radius:10px!important;background:transparent!important;
  display:flex!important;align-items:center!important;justify-content:center!important;
  transition:background .15s ease,color .15s ease;
}
[data-testid="stTabs"] button[data-baseweb="tab"] p{
  margin:0!important;font-size:.72rem!important;font-weight:850!important;
  letter-spacing:-.01em!important;color:var(--muted)!important;
}
[data-testid="stTabs"] button[aria-selected="true"]{
  background:linear-gradient(145deg,#174676,#10345a)!important;
  border:1px solid rgba(66,148,239,.26)!important;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.035)!important;
}
[data-testid="stTabs"] button[aria-selected="true"] p{color:#F7FBFF!important}

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
    {BACKTEST_N:,} past games.</div>
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

# Price at one book, since that is where the bets actually get placed.
_books = sorted({bk for o in (live_offers or {}).values()
                 for _, _, _, bk in (o.get("spreads", []) + o.get("totals", []))})
if _books:
    _book = st.selectbox("Your book", ["Best of all books"] + _books,
                         key="se_book")
    if _book != "Best of all books":
        live_offers = {
            k: {"spreads": [r for r in v.get("spreads", []) if r[3] == _book],
                "totals": [r for r in v.get("totals", []) if r[3] == _book],
                "moneylines": v.get("moneylines", []),
                "commence": v.get("commence")}
            for k, v in live_offers.items()
        }
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
            Everything where the model sits at least MIN_GAP_PTS from the
            line, ranked. Not a fixed top three — some weeks the board is
            full of disagreements and some weeks it is not, and the card
            should say so.
            """
            if df.empty:
                return [], 0
            d = df.assign(
                _gap=(pd.to_numeric(df["model_line"], errors="coerce")
                      - pd.to_numeric(df["bet_line"], errors="coerce")).abs())
            d = d[d["_gap"] >= MIN_GAP_PTS].sort_values("_gap",
                                                        ascending=False)
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
              f'{MIN_GAP_PTS:g}+ pts off the line</span></div>']

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
                _h.append(
                    f'<div class="sc-row"><div class="sc-rank">{i}</div>'
                    f'<div class="sc-main"><b>{_html.escape(str(r["pick_label"]))}</b>'
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

        _bets = card[card["bet_tier"] == "OFFICIAL"] if not card.empty else card
        with st.expander("Detail"):
            for _, r in (_sp + _to):
                render_row(r, "Bet" if float(r["expected_value"]) >= MIN_EV
                           else "Lean")

        if not _bets.empty and st.button("Freeze qualifying bets",
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
        raw = rh - ra + hfa

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
            if _gap >= MIN_GAP_PTS:
                st.success(f"**Bet {lean}.**")
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
        )

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
            raw_t = th + ta + rt_g["tbase"]
            mt = float(row["total_line"])
            edge_t = MODEL_WEIGHT * (raw_t - mt)
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
    # Capture BEFORE grading: a game can finish and be graded on the same
    # load, and the close has to be recorded at kickoff either way.
    tr, _ncap = capture_closing(tr, live_offers)
    tr, n = grade(tr, sched_all)
    if n:
        st.success(f"Graded {n} completed bets.")
    if _ncap:
        st.caption(f"Captured the closing number on {_ncap} bet(s).")

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
                f"{_clv['n_clean']} of {_clv['n']} closes were captured within "
                "3 hours of kickoff. Only those are a real measurement — a "
                "number pulled whenever the app happened to run is not a "
                "closing line. Nothing here counts as evidence until roughly "
                "100 clean captures, whatever the sign."
            )
        elif math.isfinite(_clv["t"]):
            st.caption(
                f"Signal strength {_clv['t']:+.2f} on {_clv['n_clean']} "
                "kickoff-captured bets — above +2.00 would be meaningful."
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
