"""
Headless picks: the Picks tab's engine without Streamlit, so it can run on
a schedule (GitHub Actions) and publish a page anyone can open.

    ODDS_API_KEY=... python picks.py --out site/index.html --json site/picks.json

Steps are the same as the app's Get bets:
  1. every book's spreads/totals/moneylines from The Odds API (us + eu,
     so Pinnacle is in the consensus)
  2. sharp-weighted, de-vigged fair lines (selector.consensus)
  3. nfl_combined's factors (QB changes, injuries, rest, division, late
     season) and the forecast wind rule move the fair line
  4. every offer priced; bets that clear the EV bar are sized and explained

The wind rule and stadium table are copied from sunday_edge_app.py (same
measured coefficients) so this file runs without importing the app.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

import ledger
import paper
import nfl_combined
import selector as sel
from log_lines import _rows
from nfl_data import _current_season, load_games

ODDS_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"

# From sunday_edge_app.py: measured on 3,551 outdoor games, discounted to
# ~40% for forecast error, applied above 8 mph.
WIND_PTS_PER_MPH = -0.196
WIND_FORECAST_DISCOUNT = 0.40
WIND_MIN_MPH = 8.0

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

BOOK_NAME = {
    "draftkings": "DraftKings", "fanduel": "FanDuel", "betmgm": "BetMGM",
    "williamhill_us": "Caesars", "espnbet": "ESPN BET", "betrivers":
    "BetRivers", "fanatics": "Fanatics", "hardrockbet": "Hard Rock",
    "ballybet": "Bally Bet", "betonlineag": "BetOnline", "bovada": "Bovada",
    "mybookieag": "MyBookie", "lowvig": "LowVig", "betus": "BetUS",
    "pinnacle": "Pinnacle",
}
FULL_NAME = {v: k for k, v in ODDS_TEAM.items()}
FULL_NAME.setdefault("LAR", FULL_NAME.get("LA"))


def fetch_rows(key: str, regions: str = "us,eu") -> pd.DataFrame:
    r = requests.get(ODDS_URL, params={
        "apiKey": key, "regions": regions, "markets": "spreads,totals,h2h",
        "oddsFormat": "american"}, timeout=30)
    if r.status_code == 401:
        raise SystemExit(
            f"The Odds API rejected the key (401). It is {len(key)} "
            "characters long; a key is normally 32. Re-save the "
            "ODDS_API_KEY secret with just the key, no quotes or spaces, "
            "and check the plan has credits left at the-odds-api.com.")
    r.raise_for_status()
    print(f"odds: {len(r.json())} events, credits left "
          f"{r.headers.get('x-requests-remaining')}")
    snap = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return pd.DataFrame([x for ev in r.json() for x in _rows(ev, snap)])


def forecast_wind(home_abbr: str, commence_utc: str):
    """Forecast mph at kickoff from Open-Meteo; 0 indoors; None on failure."""
    s = STADIUM.get(home_abbr)
    if not s:
        return None
    lat, lon, outdoor = s
    if not outdoor:
        return 0.0
    k = pd.to_datetime(commence_utc, utc=True, errors="coerce")
    if pd.isna(k):
        return None
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": lat, "longitude": lon, "hourly": "wind_speed_10m",
            "wind_speed_unit": "mph", "timezone": "UTC",
            "start_date": k.strftime("%Y-%m-%d"),
            "end_date": k.strftime("%Y-%m-%d")}, timeout=15)
        h = r.json().get("hourly", {})
        t = pd.to_datetime(pd.Series(h.get("time", [])), utc=True)
        if t.empty:
            return None
        return float(h["wind_speed_10m"][int((t - k).abs().idxmin())])
    except Exception:
        return None


def wind_points(mph) -> float:
    if mph is None or mph < WIND_MIN_MPH:
        return 0.0
    return WIND_PTS_PER_MPH * float(mph) * WIND_FORECAST_DISCOUNT


# Books a US bettor can normally use. The eu region is still pulled, because
# Pinnacle and the exchanges make the consensus sharper, but bets are only
# suggested at these unless MY_BOOKS says otherwise.
# Where Brian bets (2026-10-25). Every book still sets the fair line.
MY_BOOKS = "draftkings,fanduel"
US_BOOKS = ["draftkings", "fanduel", "betmgm", "williamhill_us", "espnbet",
            "betrivers", "fanatics", "hardrockbet", "ballybet", "betonlineag",
            "bovada", "mybookieag", "lowvig", "betus", "fliff", "betparx",
            "unibet_us", "superbook", "wynnbet"]

# Only totals use the combined model to move the line: its totals picks held
# up out of sample (55%) and its spread picks did not (50%). Spread factors
# are still computed and logged, as INFO, so they can earn their way back.
COMBINED_MARKETS = {"totals"}
# Spreads: the ratings-plus-injuries margin (paper.model_margins) moves the
# fair line by this share of its gap to the market. Brian's call
# (2026-10-25) to bet these; the opener backtest has them near break-even
# (52-54% at a 3+ point gap); a fifth of the gap, capped at a point,
# keeps the stated win chances in that range instead of trusting the gap.
RATINGS_SPREAD_WEIGHT = 0.2
RATINGS_SPREAD_CAP = 1.0      # points; keeps win chances near the backtest

# Every bet needs the model on its side (Brian, 2026-10-31: "bets that will
# win, not just better price"): the active adjustments (ratings, combined
# totals factors, wind, overrides) must move the fair line at least this
# many points toward the bet. A book being off the sharp consensus no longer
# carries a bet on its own. With the ratings capped at a point, a spread
# needs a 5+ point ratings gap.
MIN_LEAN = 1.0

# Underdog moneylines (Brian, 2026-10-28). moneyline_research.py: nothing
# in 2010-26 closing moneylines beats the close. Every dog loses ~2-6% (the
# vig), dogs past +400 ~17%, and dogs our ratings like went -13% to -20% in
# 2022-26. So these are priced off the moneyline market itself (de-vigged,
# sharp-weighted), moved only by the same capped ratings nudge the spreads
# get, and bet when DraftKings/FanDuel pay more than that is worth: mostly
# a line-shopping bet. Small, separate stakes, a few a week, nothing long.
DOG_MAX_PRICE = 300           # +300 or shorter; long shots lose the most
DOG_MAX_BETS = 3              # the best few each week
DOG_MAX_STAKE = 0.01          # of bankroll per dog ($10 on $1,000)


def _nick(team: str) -> str:
    return str(team).split()[-1]


def _last(name) -> str | None:
    """Surname, skipping suffixes: Michael Penix Jr. -> Penix."""
    if not isinstance(name, str) or not name.strip():
        return None
    w = [x for x in name.split()
         if x.rstrip(".").lower() not in {"jr", "sr", "ii", "iii", "iv"}]
    return w[-1] if w else None


def explain_spread(p, m: dict) -> str:
    """The model's case for a spread bet: what it thinks happens, and the
    matchup pieces pushing that way (each in points)."""
    side = 1 if p.pick == p.home else -1
    me, opp = _nick(p.pick), _nick(p.away if side > 0 else p.home)
    mine = side * m["margin"]                 # model margin for our side
    line = float(p.point)                     # e.g. +4.5 or -3
    room = mine + line
    if mine >= 0:
        call = f"The model has the {me} winning by {mine:.1f}"
    else:
        call = f"The model has the {me} losing by only {-mine:.1f}"
    if line > 0:
        lose = (f"more than {line:g}" if line == int(line)
                else f"{int(line) + 1} or more")
        call += (f", and they're getting {line:g}, so they cover unless "
                 f"they lose by {lose} ({room:.1f} pts of room).")
    else:
        call += (f", {room:.1f} more than the {-line:g} they're giving.")
    return " ".join([call] + _matchup(p, m))


def _matchup(p, m: dict) -> list[str]:
    """The ratings pieces behind the model's margin for the side picked:
    what pushes its way, and the biggest things pushing back."""
    side = 1 if p.pick == p.home else -1
    opp = _nick(p.away if side > 0 else p.home)
    parts = []
    if "pass_off" in m:
        qb_me = _last(m["home_qb"] if side > 0 else m["away_qb"])
        qb_opp = _last(m["away_qb"] if side > 0 else m["home_qb"])
        qb_txt = (f"{qb_me} over {qb_opp} at quarterback" if qb_me and qb_opp
                  else "their quarterback")
        names = {"qb": qb_txt,
                 "pass_off": "their passing game",
                 "run_off": "their running game",
                 "pass_def": "their pass defense",
                 "run_def": "their run defense",
                 "hfa": "home field" if side > 0 else None}
        for k, label in names.items():
            v = side * m[k]
            if label and v >= 0.5:
                parts.append((v, f"{label} (+{v:.1f})"))
    inj = side * m["injuries"]
    if inj >= 0.5:
        parts.append((inj, f"the injury report (+{inj:.1f})"))
    parts.sort(reverse=True)
    why = []
    if parts:
        why.append("Where the edge comes from: "
                   + ", ".join(t for _, t in parts[:4]) + ".")
    against = []
    if "pass_off" in m:
        qb_opp = _last(m["away_qb"] if side > 0 else m["home_qb"])
        qb_me = _last(m["home_qb"] if side > 0 else m["away_qb"])
        cons = {"qb": (f"{qb_opp} over {qb_me} at quarterback" if qb_opp and qb_me
                       else f"{qb_opp} at quarterback" if qb_opp else "quarterback"),
                "pass_off": "passing game", "run_off": "running game",
                "pass_def": "pass defense", "run_def": "run defense",
                "hfa": "home field"}
        whose = f"the {opp}' " if opp.endswith("s") else f"the {opp}'s "
        against = [(-side * m[k], (whose if k != "qb" or not qb_opp
                                   else "") + v)
                   for k, v in cons.items() if -side * m[k] >= 1.0]
    if -inj >= 1.0:
        against.append((-inj, "the injury report"))
    if against:
        against.sort(reverse=True)
        why.append("Working against it: " + ", ".join(
            f"{t} ({v:.1f})" for v, t in against[:2]) + ".")
    return why


def explain_dog(p, m: dict | None) -> str:
    """An underdog moneyline: what the price needs, what we think, and the
    football behind it when the ratings like this side."""
    me = _nick(p.pick)
    need = sel.american_to_prob(p.price)
    fair = p.fair_market
    dog_by = abs(fair)
    book = BOOK_NAME.get(p.book, p.book)
    why = [f"{book} pays {int(p.price):+d} on the {me}, which needs a "
           f"{need:.0%} win chance; we make it {p.win_chance:.0%}."]
    side = 1 if p.pick == p.home else -1
    if m and side * (m["margin"] - fair) > 0:
        mine = side * m["margin"]
        if mine >= 0:
            why.append(f"The market has them as {dog_by:.1f}-point underdogs, "
                       f"but the model has the {me} winning by {mine:.1f}.")
        else:
            why.append(f"The market has them as {dog_by:.1f}-point underdogs; "
                       f"the model has them losing by only {-mine:.1f}.")
        why += _matchup(p, m)
    if "market" in str(p.sources).split(","):
        # The sharp books' chance for this side, from the offer's EV at
        # the unadjusted market line.
        q = (p.ev_market + 1 - p.p_push) / (sel.payout(p.price) + 1)
        q /= max(1e-9, 1 - p.p_push)
        fair_odds = 100 * (1 - q) / q if q < 0.5 else -100 * q / (1 - q)
        why.append(f"{'Also, the' if len(why) > 1 else 'The'} sharp books' "
                   f"moneylines make the {me} worth about {fair_odds:+.0f}, "
                   f"so {book} is paying more than the market thinks they "
                   "are worth.")
    return " ".join(why)


def injuries_for(p, ctx: dict) -> list[dict]:
    """The injured players the model counts in this game (paper.model_margins),
    each with his team and the points it takes off (none for a QB swap,
    which the ratings' QB term prices). The totals model only uses the
    weekly injury report, so non-QBs on IR are left out there."""
    m = (ctx.get(p.home, {}).get("ratings") or {}).get("injured") or {}
    out = []
    for side, team in (("away", p.away), ("home", p.home)):
        for x in m.get(side, []):
            if p.market == "totals" and x["status"] == "IR" and x["pos"] != "QB":
                continue
            out.append({"team": _nick(team), **x})
    return out


def explain_pick(p, ctx: dict) -> str:
    """Why this bet: the model's own case where it has one, else the gap
    between this book and the sharp books."""
    why = []
    c = ctx.get(p.home, {})
    m = c.get("ratings")
    if p.market == "h2h":
        return explain_dog(p, m)
    side = 1 if p.pick == p.home else -1
    rated = (p.market == "spreads" and "ratings" in str(p.sources) and m
             and side * (m["margin"] - p.fair_market) > 0)
    if rated:
        why.append(explain_spread(p, m))
    elif "market" in str(p.sources).split(","):
        if p.market == "totals":
            fv = f"put the total at {p.fair_market:.1f}"
        else:
            fav = p.home if p.fair_market >= 0 else p.away
            fv = f"make {fav} a {abs(p.fair_market):.1f}-point favorite"
        why.append(f"{BOOK_NAME.get(p.book, p.book)} hasn't caught up with "
                   f"the sharp books, which {fv}.")
    if p.market == "totals":
        toward = -1 if str(p.pick).lower() == "under" else 1
        if "combined" in str(p.sources) and "why_total" in c:
            why.append(nfl_combined.explain(c["why_total"], c["x"], c["h"],
                                            c["a"], "totals", toward=toward))
        if "wind" in str(p.sources) and toward < 0 and c.get("wind"):
            why.append(c["wind"])
    for n in str(p.notes or "").split("; "):
        if n and "override" in str(p.sources) and n not in why:
            why.append(n)
    return " ".join(w for w in why if w)


def week_end(now: datetime) -> pd.Timestamp:
    """End of the current NFL week: the coming Tuesday 12:00 UTC, after
    Monday night's game. A run on Tuesday looks at the week ahead."""
    now = pd.Timestamp(now).tz_convert("UTC")
    days = (1 - now.weekday()) % 7 or 7
    return (now + pd.Timedelta(days=days)).normalize() + pd.Timedelta(hours=12)


def this_week(rows: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    """Only games that have not kicked off and start before the current NFL
    week ends. Books post next week's lines early; those move a lot on
    injury news, so the bot waits for that week's own runs. Games already
    under way carry live in-game lines, which the model can't price (a
    scheduled run that GitHub starts late would otherwise bet them)."""
    now = pd.Timestamp(now or datetime.now(timezone.utc)).tz_convert("UTC")
    kick = pd.to_datetime(rows["commence_time"], utc=True, errors="coerce")
    return rows[(kick > now) & (kick < week_end(now))]


def build_picks(rows: pd.DataFrame, my_books=None, bankroll: float = 1000.0,
                min_ev: float = 0.02, use_wind: bool = True, overrides=(),
                wind_fn=None, week_only: bool = True, ctx_out=None):
    """
    The whole Get bets pipeline. Returns (picks, fair, priced). `overrides`
    are selector.Adjustment rows from the user; `wind_fn(home_abbr,
    commence_utc)` returns forecast mph (defaults to Open-Meteo).
    `week_only` drops games after the current NFL week. `ctx_out`, a dict,
    gets each home team's model context (ratings, factors, wind note).
    """
    wind_fn = wind_fn or forecast_wind
    if week_only:
        rows = this_week(rows)
    games = load_games()
    mm, tm = sel.build_outcome_models(games)
    fair = sel.consensus(rows, mm, tm)
    adj = list(overrides)
    ctx = {}
    season = _current_season()
    left = games[(games["season"] == season) & games["home_score"].isna()]
    if len(left):
        week = int(left["week"].min())
        try:
            cw = nfl_combined.predict_week(season, week)
        except Exception as e:
            print(f"combined model unavailable: {e}")
            cw = pd.DataFrame()
        for r in cw.itertuples():
            hn = FULL_NAME.get(r.home)
            if not hn:
                continue
            ctx[hn] = {"why_total": r.why_total, "why_spread": r.why_spread,
                       "x": r.x, "h": r.home, "a": r.away}
            for mkt, v in (("spreads", r.adj_spread), ("totals", r.adj_total)):
                if abs(v) >= 0.05:
                    src = "combined" if mkt in COMBINED_MARKETS else \
                        "combined_info"
                    adj.append(sel.Adjustment(hn, mkt, float(v), src))
    try:
        mine = paper.model_margins(fair, ODDS_TEAM)
    except Exception as e:
        print(f"ratings unavailable: {e}")
        mine = {}
    for k, m in mine.items():
        f = fair[k]
        if np.isnan(f.market_margin):
            continue
        pts = float(np.clip(RATINGS_SPREAD_WEIGHT
                            * (m["margin"] - f.market_margin),
                            -RATINGS_SPREAD_CAP, RATINGS_SPREAD_CAP))
        ctx.setdefault(f.home_team, {})["ratings"] = m
        if abs(pts) >= 0.05:
            adj.append(sel.Adjustment(f.home_team, "spreads", pts, "ratings"))
    if use_wind:
        for f in fair.values():
            mph = wind_fn(ODDS_TEAM.get(f.home_team), f.commence)
            pts = wind_points(mph)
            if pts:
                note = (f"{mph:.0f} mph wind is forecast, which keeps scoring "
                        f"down ({abs(pts):.1f} pts).")
                ctx.setdefault(f.home_team, {})["wind"] = note
                adj.append(sel.Adjustment(f.home_team, "totals", pts, "wind"))
    sel.apply_adjustments(fair, adj)
    if ctx_out is not None:
        ctx_out.update(ctx)
    pol = sel.Policy(min_ev=min_ev)
    priced = sel.price_offers(rows, fair, mm, tm, pol)
    if priced.empty:
        return priced, fair, priced
    # Moneylines are priced from the spread distribution, which is not
    # validated for them; they stay visible in `priced` but are not picked.
    bettable = priced[priced["market"].isin(["spreads", "totals"])]
    bettable = bettable[bettable["book"].isin(my_books or US_BOOKS)]
    picks = sel.select(model_backed(bettable), bankroll, pol)
    dogs = underdogs(model_backed(priced[priced["book"].isin(my_books or US_BOOKS)]),
                     bankroll, pol)
    if len(dogs):
        picks = pd.concat([picks, dogs], ignore_index=True)
        # a moneyline has no point; None, not NaN, so picks.json stays JSON
        picks["point"] = picks["point"].astype(object).where(
            picks["market"] != "h2h", None)
    if len(picks):
        picks["win_chance"] = picks["p_win"] / (1 - picks["p_push"])
        picks["why"] = [explain_pick(p, ctx) for p in picks.itertuples()]
        picks["injuries"] = [injuries_for(p, ctx) for p in picks.itertuples()]
        picks["book"] = picks["book"].map(lambda b: BOOK_NAME.get(b, b))
    return picks, fair, priced


def lean(priced: pd.DataFrame) -> pd.Series:
    """Points the active adjustments move the fair line toward each
    offer's side (fair minus the sharp consensus, signed to the side)."""
    toward = np.where(priced["market"] == "totals",
                      np.where(priced["pick"].astype(str).str.lower() == "under", -1, 1),
                      np.where(priced["pick"] == priced["home"], 1, -1))
    return (priced["fair"] - priced["fair_market"]) * toward


def model_backed(priced: pd.DataFrame, min_lean: float = MIN_LEAN) -> pd.DataFrame:
    """Offers the model itself argues for by at least min_lean points."""
    if priced.empty:
        return priced
    return priced[lean(priced) >= min_lean - 1e-9]


def underdogs(priced: pd.DataFrame, bankroll: float,
              policy: sel.Policy) -> pd.DataFrame:
    """The best few underdog moneylines: a dog in the market (under 50% to
    win at the sharp books) at +100 to DOG_MAX_PRICE, whose price clears
    the EV bar. Quarter Kelly, at most DOG_MAX_STAKE each."""
    if priced.empty:
        return priced
    d = priced[(priced["market"] == "h2h") & (priced["price"] >= 100)
               & (priced["price"] <= DOG_MAX_PRICE)
               & (priced["ev"] >= policy.min_ev)].copy()
    if d.empty:
        return d
    fav_home = d["fair_market"] >= 0
    d = d[(d["pick"] == d["home"]) != fav_home]
    d = (d.sort_values("ev", ascending=False)
          .drop_duplicates(["event_id", "pick"]).head(DOG_MAX_BETS))
    d["stake"] = (np.minimum(d["kelly"] * policy.kelly_mult, DOG_MAX_STAKE)
                  * bankroll).round(2)
    d["point"] = None
    return d.reset_index(drop=True)


def _bet_label(p) -> str:
    if p.market == "h2h":
        return f"{p.pick} ML ({int(p.price):+d})"
    pt = f"{p.point:+g}" if p.market == "spreads" else f"{p.point:g}"
    return f"{p.pick} {pt} ({int(p.price):+d})"


def render_html(picks: pd.DataFrame, fair: dict, made: str, bankroll: float,
                history: list[dict], tests: list[dict] = ()) -> str:
    """The dashboard: dashboard.html with this run's data embedded, so the
    page works from GitHub Pages, the picks branch or a saved copy."""
    cols = ["away", "home", "commence", "market", "pick", "point", "price",
            "book", "stake", "win_chance", "ev", "why", "injuries"]
    data = {
        "updated": made, "bankroll": bankroll,
        "picks": picks[cols].to_dict("records") if len(picks) else [],
        "fair": [{"away": f.away_team, "home": f.home_team,
                  "market_margin": f.market_margin, "margin": f.margin,
                  "market_total": f.market_total, "total": f.total}
                 for f in fair.values()],
        "history": {"summary": ledger.summary(history), "bets": history},
        "paper": {"summary": paper.summary(list(tests)), "gap": paper.GAP,
                  "bets": list(tests)},
    }
    blob = json.dumps(data, default=str).replace("</", "<\\/")
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "dashboard.html")) as fh:
        return fh.read().replace("/*DATA*/", blob)


def render_alert(bets: list[dict]) -> str:
    """Short note on bets this run found that no earlier run had."""
    out = [f"**{len(bets)} new bet{'s' if len(bets) != 1 else ''}**\n"]
    for b in bets:
        p = pd.Series(b)
        kick = pd.to_datetime(p.commence, utc=True).tz_convert(
            "America/New_York").strftime("%a %-I:%M %p ET")
        out.append(f"- **{_bet_label(p)}** at {p.book}, ${p.stake:,.2f} "
                   f"({p.win_chance * 100:.0f}% to win) · {p.away} @ "
                   f"{p.home}, {kick}")
    out.append("\nDashboard: https://bviolino93.github.io/NFL-model/")
    return "\n".join(out) + "\n"


def render_dropped(bets: list[dict]) -> str:
    """Alerted bets a later run no longer makes: skip them if not bet yet.
    They stay in the record at the alerted price."""
    one = len(bets) == 1
    out = [f"**Skip {'this' if one else 'these'} if you haven't bet "
           f"{'it' if one else 'them'}: {len(bets)} bet{'' if one else 's'} "
           f"dropped off the board**\n"]
    for b in bets:
        p = pd.Series(b)
        kick = pd.to_datetime(p.commence, utc=True).tz_convert(
            "America/New_York").strftime("%a %-I:%M %p ET")
        out.append(f"- ~~{_bet_label(p)} at {p.book}~~ · {p.away} @ {p.home}, {kick}")
    out.append("\nThe line moved or the model no longer backs it. If you already "
               "bet it, it still counts in the record at the price you got alerted.")
    out.append("\nDashboard: https://bviolino93.github.io/NFL-model/")
    return "\n".join(out) + "\n"


def render_md(picks: pd.DataFrame, made: str) -> str:
    """Same content as the page, as Markdown: GitHub renders it on any
    phone with no hosting set up."""
    out = [f"# Sunday Edge Picks\n\n_Updated {made}_\n"]
    if picks.empty:
        out.append("**No bets right now.** Nothing beats the sharp market by "
                   "enough, which is the usual answer and the right one.\n")
    for p in picks.itertuples():
        kick = pd.to_datetime(p.commence, utc=True).tz_convert(
            "America/New_York").strftime("%a %-I:%M %p ET")
        out.append(f"### {_bet_label(p)} at {p.book}\n\n"
                   f"**Stake ${p.stake:,.2f}** · win chance "
                   f"{p.win_chance * 100:.1f}% · EV {p.ev * 100:.1f}%  \n"
                   f"{p.away} @ {p.home} · {kick}\n\n{p.why}\n")
    out.append("---\nQuarter-Kelly stakes on a $1,000 bankroll, capped at 2% "
               "per bet and 10% per slate (underdog moneylines: 1% each, "
               "best three). Win chances are model estimates, "
               "not guarantees. 21+ where legal; 1-800-GAMBLER.\n")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="site/index.html")
    ap.add_argument("--json", default="site/picks.json")
    ap.add_argument("--md", default="site/README.md")
    ap.add_argument("--history", default="site/history.json",
                    help="bet ledger; read, extended with this run, graded")
    ap.add_argument("--paper", default="site/paper.json",
                    help="no-money spread test log; read, extended, graded")
    ap.add_argument("--alert", default="alert.md",
                    help="written only when this run finds bets not seen "
                         "before; the workflow posts it as a notification")
    ap.add_argument("--bankroll", type=float, default=1000.0)
    ap.add_argument("--min-ev", type=float, default=0.02)
    ap.add_argument("--books",
                    default=os.environ.get("MY_BOOKS") or MY_BOOKS,
                    help="comma list of books you can bet at")
    ap.add_argument("--lines-csv", default="",
                    help="use a logged snapshot instead of the live feed")
    a = ap.parse_args()
    if a.lines_csv:
        rows = sel.latest_snapshot(pd.read_csv(a.lines_csv))
    else:
        # Pasted secrets often carry a newline, spaces or the quotes from a
        # TOML line (odds_api_key = "..."); none of those are part of a key.
        key = (os.environ.get("ODDS_API_KEY") or "").strip().strip('"\'')
        if "=" in key:
            key = key.split("=", 1)[1].strip().strip('"\'')
        if not key:
            raise SystemExit("set ODDS_API_KEY")
        rows = fetch_rows(key)
    books = [b.strip() for b in a.books.split(",") if b.strip()]
    picks, fair, _ = build_picks(rows, books, a.bankroll, a.min_ev)
    now = datetime.now(timezone.utc)
    made = now.strftime("%a %b %-d, %H:%M UTC")
    for path in (a.out, a.json, a.md, a.history):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cols = ["away", "home", "commence", "market", "pick", "point", "price",
            "book", "stake", "win_chance", "ev", "why", "injuries"]
    out = picks[cols].to_dict("records") if len(picks) else []
    out = json.loads(json.dumps(out, default=str))
    # Bets on later weeks are left over from before week_only; drop them.
    end = week_end(now)
    history = [b for b in ledger.load(a.history)
               if pd.Timestamp(b["commence"]) < end]
    seen = {ledger._key(b) for b in history}
    was = ledger.on_board(history)
    history = ledger.merge(history, out, now, now.isoformat())
    gone = ledger.dropped(was, history, now)
    try:
        history = ledger.grade(history, load_games(), ODDS_TEAM)
    except Exception as exc:  # scores can wait for the next run
        print(f"grading skipped: {exc}")
    ledger.save(a.history, history)
    tests = paper.load(a.paper)
    try:
        games = load_games()
        mm, tm = sel.build_outcome_models(games)
        every = sel.consensus(rows, mm, tm)     # next week's lines too
        tests = paper.log(tests, every, paper.model_margins(every, ODDS_TEAM),
                          now.isoformat())
        tests = paper.grade(tests, games, ODDS_TEAM)
        paper.save(a.paper, tests)
    except Exception as exc:  # the test must never block real picks
        print(f"spread test skipped: {exc}")
    with open(a.out, "w") as fh:
        fh.write(render_html(picks, fair, made, a.bankroll, history, tests))
    with open(a.md, "w") as fh:
        fh.write(render_md(picks, made))
    with open(a.json, "w") as fh:
        json.dump({"updated": made, "picks": out}, fh, indent=2, default=str)
    fresh = [p for p in out if ledger._key(p) not in seen]
    note = "\n".join(r(b) for b, r in ((fresh, render_alert), (gone, render_dropped)) if b)
    if note:
        with open(a.alert, "w") as fh:
            fh.write(note)
    print(f"{len(out)} bets, {len(fresh)} new, {len(gone)} dropped")
    for p in out:
        print(f"  {p['pick']} {p['point']} ({p['price']}) at {p['book']}: "
              f"${p['stake']} | {p['win_chance']:.1%} | {p['why']}")


if __name__ == "__main__":
    main()
