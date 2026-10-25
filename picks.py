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

import pandas as pd
import requests

import ledger
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
US_BOOKS = ["draftkings", "fanduel", "betmgm", "williamhill_us", "espnbet",
            "betrivers", "fanatics", "hardrockbet", "ballybet", "betonlineag",
            "bovada", "mybookieag", "lowvig", "betus", "fliff", "betparx",
            "unibet_us", "superbook", "wynnbet"]

# Only totals use the combined model to move the line: its totals picks held
# up out of sample (55%) and its spread picks did not (50%). Spread factors
# are still computed and logged, as INFO, so they can earn their way back.
COMBINED_MARKETS = {"totals"}


def explain_pick(p, ctx: dict) -> str:
    """Why this bet: the market gap, then only the factors pointing its way."""
    why = []
    if "market" in str(p.sources).split(","):
        if p.market == "totals":
            fv = f"put the total at {p.fair_market:.1f}"
        else:
            fav = p.home if p.fair_market >= 0 else p.away
            fv = f"make {fav} a {abs(p.fair_market):.1f}-point favorite"
        why.append(f"{BOOK_NAME.get(p.book, p.book)} hasn't caught up with "
                   f"the sharp books, which {fv}.")
    c = ctx.get(p.home, {})
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
    """Only games kicking off before the current NFL week ends. Books post
    next week's lines early; those move a lot on injury news, so the bot
    waits for that week's own runs."""
    end = week_end(now or datetime.now(timezone.utc))
    kick = pd.to_datetime(rows["commence_time"], utc=True, errors="coerce")
    return rows[kick < end]


def build_picks(rows: pd.DataFrame, my_books=None, bankroll: float = 1000.0,
                min_ev: float = 0.02, use_wind: bool = True, overrides=(),
                wind_fn=None, week_only: bool = True):
    """
    The whole Get bets pipeline. Returns (picks, fair, priced). `overrides`
    are selector.Adjustment rows from the user; `wind_fn(home_abbr,
    commence_utc)` returns forecast mph (defaults to Open-Meteo).
    `week_only` drops games after the current NFL week.
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
    pol = sel.Policy(min_ev=min_ev)
    priced = sel.price_offers(rows, fair, mm, tm, pol)
    if priced.empty:
        return priced, fair, priced
    # Moneylines are priced from the spread distribution, which is not
    # validated for them; they stay visible in `priced` but are not picked.
    bettable = priced[priced["market"].isin(["spreads", "totals"])]
    bettable = bettable[bettable["book"].isin(my_books or US_BOOKS)]
    picks = sel.select(bettable, bankroll, pol)
    if len(picks):
        picks["win_chance"] = picks["p_win"] / (1 - picks["p_push"])
        picks["why"] = [explain_pick(p, ctx) for p in picks.itertuples()]
        picks["book"] = picks["book"].map(lambda b: BOOK_NAME.get(b, b))
    return picks, fair, priced


def _bet_label(p) -> str:
    if p.market == "h2h":
        return f"{p.pick} ML ({int(p.price):+d})"
    pt = f"{p.point:+g}" if p.market == "spreads" else f"{p.point:g}"
    return f"{p.pick} {pt} ({int(p.price):+d})"


def render_html(picks: pd.DataFrame, fair: dict, made: str, bankroll: float,
                history: list[dict]) -> str:
    """The dashboard: dashboard.html with this run's data embedded, so the
    page works from GitHub Pages, the picks branch or a saved copy."""
    cols = ["away", "home", "commence", "market", "pick", "point", "price",
            "book", "stake", "win_chance", "ev", "why"]
    data = {
        "updated": made, "bankroll": bankroll,
        "picks": picks[cols].to_dict("records") if len(picks) else [],
        "fair": [{"away": f.away_team, "home": f.home_team,
                  "market_margin": f.market_margin, "margin": f.margin,
                  "market_total": f.market_total, "total": f.total}
                 for f in fair.values()],
        "history": {"summary": ledger.summary(history), "bets": history},
    }
    blob = json.dumps(data, default=str).replace("</", "<\\/")
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "dashboard.html")) as fh:
        return fh.read().replace("/*DATA*/", blob)


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
               "per bet and 10% per slate. Win chances are model estimates, "
               "not guarantees. 21+ where legal; 1-800-GAMBLER.\n")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="site/index.html")
    ap.add_argument("--json", default="site/picks.json")
    ap.add_argument("--md", default="site/README.md")
    ap.add_argument("--history", default="site/history.json",
                    help="bet ledger; read, extended with this run, graded")
    ap.add_argument("--bankroll", type=float, default=1000.0)
    ap.add_argument("--min-ev", type=float, default=0.02)
    ap.add_argument("--books", default=os.environ.get("MY_BOOKS", ""),
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
            "book", "stake", "win_chance", "ev", "why"]
    out = picks[cols].to_dict("records") if len(picks) else []
    out = json.loads(json.dumps(out, default=str))
    # Bets on later weeks are left over from before week_only; drop them.
    end = week_end(now)
    history = [b for b in ledger.load(a.history)
               if pd.Timestamp(b["commence"]) < end]
    history = ledger.merge(history, out, now, now.isoformat())
    try:
        history = ledger.grade(history, load_games(), ODDS_TEAM)
    except Exception as exc:  # scores can wait for the next run
        print(f"grading skipped: {exc}")
    ledger.save(a.history, history)
    with open(a.out, "w") as fh:
        fh.write(render_html(picks, fair, made, a.bankroll, history))
    with open(a.md, "w") as fh:
        fh.write(render_md(picks, made))
    with open(a.json, "w") as fh:
        json.dump({"updated": made, "picks": out}, fh, indent=2, default=str)
    print(f"{len(out)} bets")
    for p in out:
        print(f"  {p['pick']} {p['point']} ({p['price']}) at {p['book']}: "
              f"${p['stake']} | {p['win_chance']:.1%} | {p['why']}")


if __name__ == "__main__":
    main()
