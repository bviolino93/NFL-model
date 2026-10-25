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
import html
import json
import os
from datetime import datetime, timezone

import pandas as pd
import requests

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


def explain_pick(p) -> str:
    why = []
    if "market" in str(p.sources).split(","):
        if p.market == "totals":
            fv = f"put the total at {p.fair_market:.1f}"
        else:
            fav = p.home if p.fair_market >= 0 else p.away
            fv = f"make {fav} a {abs(p.fair_market):.1f}-point favorite"
        why.append(f"{p.book} hasn't caught up with the sharp books, which "
                   f"{fv}.")
    if p.notes:
        why.append(p.notes)
    return " ".join(why)


def build_picks(rows: pd.DataFrame, my_books=None, bankroll: float = 1000.0,
                min_ev: float = 0.02, use_wind: bool = True):
    games = load_games()
    mm, tm = sel.build_outcome_models(games)
    fair = sel.consensus(rows, mm, tm)
    adj = []
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
            for mkt, v, why in (("spreads", r.adj_spread, r.why_spread),
                                ("totals", r.adj_total, r.why_total)):
                if abs(v) >= 0.05:
                    adj.append(sel.Adjustment(
                        hn, mkt, float(v), "combined",
                        nfl_combined.explain(why, r.x, r.home, r.away, mkt)))
    if use_wind:
        for f in fair.values():
            mph = forecast_wind(ODDS_TEAM.get(f.home_team), f.commence)
            pts = wind_points(mph)
            if pts:
                adj.append(sel.Adjustment(
                    f.home_team, "totals", pts, "wind",
                    f"{mph:.0f} mph wind is forecast, which keeps scoring "
                    f"down ({abs(pts):.1f} pts)."))
    sel.apply_adjustments(fair, adj)
    pol = sel.Policy(min_ev=min_ev)
    priced = sel.price_offers(rows, fair, mm, tm, pol)
    bettable = priced[priced["book"].isin(my_books)] if my_books else priced
    picks = sel.select(bettable, bankroll, pol)
    if len(picks):
        picks["book"] = picks["book"].map(lambda b: BOOK_NAME.get(b, b))
        picks["win_chance"] = picks["p_win"] / (1 - picks["p_push"])
        picks["why"] = [explain_pick(p) for p in picks.itertuples()]
    return picks, fair


def _bet_label(p) -> str:
    if p.market == "h2h":
        return f"{p.pick} ML ({int(p.price):+d})"
    pt = f"{p.point:+g}" if p.market == "spreads" else f"{p.point:g}"
    return f"{p.pick} {pt} ({int(p.price):+d})"


def render_html(picks: pd.DataFrame, fair: dict, made: str) -> str:
    e = html.escape
    cards = []
    for p in picks.itertuples():
        kick = pd.to_datetime(p.commence, utc=True).tz_convert(
            "America/New_York").strftime("%a %-I:%M %p ET")
        cards.append(
            f'<article><h2>{e(_bet_label(p))}</h2>'
            f'<p class="meta">{e(p.book)} &middot; stake ${p.stake:,.2f} '
            f'&middot; win chance {p.win_chance * 100:.1f}% &middot; EV '
            f'{p.ev * 100:.1f}%</p>'
            f'<p class="game">{e(p.away)} @ {e(p.home)} &middot; {kick}</p>'
            f'<p>{e(p.why)}</p></article>')
    if not cards:
        cards.append('<article><h2>No bets right now</h2><p>Nothing beats '
                     'the sharp market by enough. That is the usual answer, '
                     'and the right one.</p></article>')
    lines = "".join(
        f"<tr><td>{e(f.away_team)} @ {e(f.home_team)}</td>"
        f"<td>{f.market_margin:+.1f}</td><td>{f.margin:+.1f}</td>"
        f"<td>{f.market_total:.1f}</td><td>{f.total:.1f}</td></tr>"
        for f in fair.values())
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sunday Edge Picks</title><style>
:root{{--bg:#0a1020;--card:#111a2e;--fg:#e8f0fa;--mute:#8ea4ba;--acc:#60a5fa}}
@media (prefers-color-scheme: light){{:root{{--bg:#f5f7fb;--card:#fff;--fg:#0a1020;--mute:#55657a}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}}
main{{max-width:720px;margin:0 auto;padding:24px 16px}}
h1{{font-size:1.4rem;margin:0}} .sub{{color:var(--mute);margin:4px 0 20px}}
article{{background:var(--card);border-radius:14px;padding:14px 16px;margin:0 0 12px}}
article h2{{font-size:1.15rem;margin:0 0 4px;color:var(--acc)}}
.meta{{font-weight:600;margin:0}} .game{{color:var(--mute);margin:0 0 6px}}
table{{width:100%;border-collapse:collapse;font-size:.85rem}}
td,th{{padding:6px 4px;border-bottom:1px solid rgba(128,128,128,.2);text-align:right}}
td:first-child,th:first-child{{text-align:left}}
details{{margin-top:20px}} footer{{color:var(--mute);font-size:.75rem;margin-top:24px}}
</style></head><body><main>
<h1>Sunday Edge Picks</h1><p class="sub">Updated {e(made)}</p>
{''.join(cards)}
<details><summary>Fair lines (home margin, total): market vs adjusted</summary>
<table><tr><th>Game</th><th>Mkt</th><th>Adj</th><th>Mkt tot</th><th>Adj tot</th></tr>
{lines}</table></details>
<footer>Stakes are quarter-Kelly on a $1,000 bankroll, capped at 2% per bet
and 10% per slate. Win chances are model estimates, not guarantees. 21+ where
legal; 1-800-GAMBLER.</footer></main></body></html>"""


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
    picks, fair = build_picks(rows, books, a.bankroll, a.min_ev)
    made = datetime.now(timezone.utc).strftime("%a %b %-d, %H:%M UTC")
    for path in (a.out, a.json, a.md):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        fh.write(render_html(picks, fair, made))
    with open(a.md, "w") as fh:
        fh.write(render_md(picks, made))
    cols = ["away", "home", "commence", "market", "pick", "point", "price",
            "book", "stake", "win_chance", "ev", "why"]
    out = picks[cols].to_dict("records") if len(picks) else []
    with open(a.json, "w") as fh:
        json.dump({"updated": made, "picks": out}, fh, indent=2, default=str)
    print(f"{len(out)} bets")
    for p in out:
        print(f"  {p['pick']} {p['point']} ({p['price']}) at {p['book']}: "
              f"${p['stake']} | {p['win_chance']:.1%} | {p['why']}")


if __name__ == "__main__":
    main()
