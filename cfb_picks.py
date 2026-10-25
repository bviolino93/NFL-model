"""
College football picks: picks.py's engine pointed at college games.

    ODDS_API_KEY=... python cfb_picks.py --out site/cfb/index.html

Same steps as the NFL picks:
  1. every book's spreads and totals from The Odds API (us + eu)
  2. sharp-weighted, de-vigged fair lines (selector.consensus), with the
     result distributions built from college games (bigger margins, fewer
     key numbers than the NFL)
  3. cfb_model's prediction of result-minus-line moves the fair line:
     spreads always; totals only on lines seen 2+ days before kickoff,
     where the backtest found the edge (they lose to the closing total)
  4. every DraftKings/FanDuel offer priced; bets that clear the EV bar are
     sized (quarter Kelly) and explained in football terms

The bet history, alerts and dashboard are the NFL ones, in site/cfb/.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

import cfb_data
import cfb_model
import ledger
import selector as sel
from log_lines import _rows

ODDS_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf/odds"
DASHBOARD_URL = "https://bviolino93.github.io/NFL-model/cfb/"
MY_BOOKS = "draftkings,fanduel"
BOOK_NAME = {"draftkings": "DraftKings", "fanduel": "FanDuel"}

# The model's lean is shrunk already; the caps stop a data glitch (a team
# with two games of play-by-play) from making a huge edge.
SPREAD_CAP = 3.0
TOTAL_CAP = 3.0
TOTAL_MIN_LEAD_DAYS = 2.0
# College picks go through at a 3% edge (NFL: 2%): below ~1.5 points of
# lean the backtest's picks ran about break-even.
MIN_EV = 0.03

# Odds API name -> schedule name where the school part differs.
ALIASES = {
    "appalachian state": "app state", "hawaii": "hawaii", "umass": "massachusetts",
    "southern mississippi": "southern miss", "louisiana monroe": "ul monroe",
    "ul monroe": "ul monroe", "sam houston state": "sam houston",
    "connecticut": "uconn", "fiu": "florida international",
    "north carolina state": "nc state", "san jose state": "san jose state",
    "texas san antonio": "utsa", "central florida": "ucf",
    "louisiana lafayette": "louisiana", "miami fl": "miami",
    "mississippi": "ole miss", "brigham young": "byu",
    "southern methodist": "smu", "texas christian": "tcu",
    "louisiana state": "lsu", "southern california": "usc",
}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"['’.]", "", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def _matches(odds_name: str, school: str) -> int:
    """Length of the school name if the Odds API name is that school plus
    a nickname ("Miami (OH) RedHawks" -> "Miami (OH)"), else 0."""
    o, sc = _norm(odds_name), _norm(school)
    for a, b in ALIASES.items():
        if o.startswith(a + " ") and b == sc:
            return len(sc) + 1
    return len(sc) if o == sc or o.startswith(sc + " ") else 0


def match_games(fair: dict, sched: pd.DataFrame) -> dict:
    """Odds event id -> schedule game_id. A game matches when both teams'
    names do, or one does and it's that team's only game within a day."""
    s = sched.assign(day=pd.to_datetime(sched.start_date, utc=True))
    out = {}
    for ev, f in fair.items():
        kick = pd.Timestamp(f.commence)
        near = s[(s.day - kick).abs() <= pd.Timedelta(hours=36)]
        best, score = None, 0
        for r in near.itertuples():
            h, a = _matches(f.home_team, r.home_team), _matches(f.away_team, r.away_team)
            if not (h or a):
                # neutral-site games are sometimes listed the other way round
                h, a = _matches(f.home_team, r.away_team), _matches(f.away_team, r.home_team)
                if h and a:
                    h = a = 0       # flipped home/away: skip rather than mis-sign
            sc = (h > 0) + (a > 0) + 0.001 * (h + a)
            if sc > score:
                best, score = r.game_id, sc
        if best is not None and score >= 1:
            out[ev] = int(best)
    return out


def outcome_models(g: pd.DataFrame):
    d = g.dropna(subset=["close_spread", "close_total", "margin"])
    d = d[d.season >= 2012]
    return (sel.OutcomeModel(-d.close_spread.to_numpy(float), d.margin.to_numpy(float)),
            sel.OutcomeModel(d.close_total.to_numpy(float), d.points.to_numpy(float)))


def fetch_rows(key: str, regions: str = "us,eu") -> pd.DataFrame:
    r = requests.get(ODDS_URL, params={
        "apiKey": key, "regions": regions, "markets": "spreads,totals",
        "oddsFormat": "american"}, timeout=30)
    r.raise_for_status()
    print(f"odds: {len(r.json())} college events, credits left "
          f"{r.headers.get('x-requests-remaining')}")
    snap = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return pd.DataFrame([x for ev in r.json() for x in _rows(ev, snap)])


def week_end(now) -> pd.Timestamp:
    """The next Sunday 12:00 UTC, after Saturday's late games: a run from
    Sunday evening on looks at the coming Saturday."""
    now = pd.Timestamp(now).tz_convert("UTC")
    end = now.normalize() + pd.Timedelta(days=(6 - now.weekday()) % 7, hours=12)
    return end if end > now else end + pd.Timedelta(days=7)


# ----------------------------------------------------------------------
# Plain-English reasons
# ----------------------------------------------------------------------
def _tier(p: float) -> str:
    if p >= 0.9:
        return "top 10%"
    if p >= 0.75:
        return "top quarter"
    if p <= 0.1:
        return "bottom 10%"
    if p <= 0.25:
        return "bottom quarter"
    return "middle of the pack"


class Ranks:
    """Where each team's opponent-adjusted unit sits among the teams
    playing this week (1 = best; defenses rank low EPA allowed as best)."""

    def __init__(self, rows: pd.DataFrame):
        t = []
        for side, team in (("h", "home_team"), ("a", "away_team")):
            c = {f"{u}_{s}": rows[f"{side}_{u}_{s}"].values for u in ("off", "def")
                 for s in ("epa", "pass_epa", "rush_epa", "plays")}
            t.append(pd.DataFrame({"team": rows[team].values, **c}))
        t = pd.concat(t).drop_duplicates("team").set_index("team")
        self.p = t.rank(pct=True)
        for c in t.columns:
            if c.startswith("def_"):
                self.p[c] = 1 - self.p[c] + 1 / len(t)

    def tier(self, team, col) -> str | None:
        try:
            v = self.p.loc[team, col]
        except KeyError:
            return None
        return None if pd.isna(v) else _tier(v)


UNIT = {"epa": ("offense", "defense", "per play"),
        "pass_epa": ("passing game", "pass defense", "per dropback"),
        "rush_epa": ("running game", "run defense", "per carry")}


def _matchup(me, opp, stat, ranks):
    """'Oregon's passing game (top 10%) against Purdue's pass defense
    (bottom quarter)', from whichever side of the ball favours `me` more."""
    o, d, _ = UNIT[stat]
    p = lambda t, c: ranks.p[c].get(t, 0.5)
    if p(me, "off_" + stat) - p(opp, "def_" + stat) >= p(me, "def_" + stat) - p(opp, "off_" + stat):
        return (f"{me}'s {o} ({ranks.tier(me, 'off_' + stat)}) against {opp}'s {d} "
                f"({ranks.tier(opp, 'def_' + stat)})")
    return (f"{me}'s {d} ({ranks.tier(me, 'def_' + stat)}) against {opp}'s {o} "
            f"({ranks.tier(opp, 'off_' + stat)})")


def _units(home, away, stat, ranks, over: bool) -> str:
    """The two units (of the four offenses and defenses) that most push a
    game over (or under): 'Oregon's passing game (top 10%) and Purdue's
    pass defense (bottom quarter)'."""
    o, d, _ = UNIT[stat]
    p = lambda t, c: ranks.p[c].get(t, 0.5)
    # how much each unit pushes scoring up: good offenses, bad defenses
    cand = [(p(t, "off_" + stat), f"{t}'s {o} ({ranks.tier(t, 'off_' + stat)})") for t in (home, away)]
    cand += [(1 - p(t, "def_" + stat), f"{t}'s {d} ({ranks.tier(t, 'def_' + stat)})") for t in (home, away)]
    cand = sorted((c for c in cand if (c[0] >= 0.75 if over else c[0] <= 0.25)),
                  reverse=over)
    return " and ".join(t for _, t in cand[:2])


def _fav(home, away, m):
    return f"{home if m >= 0 else away} by {abs(m):.1f}"


def explain(p, row: pd.Series, parts: dict, ranks: Ranks, adj: float,
            fair_market: float) -> str:
    """The model's case for a bet in football terms: what it expects, then
    each opinion behind it (cfb_model.LineModel) that pushes our way, with
    the matchup that drives the efficiency numbers, and the biggest one
    that pushes the other way. parts: opinion -> (push on the home margin
    or total, the opinion itself in points)."""
    home, away = row.home_team, row.away_team
    spread = p.market == "spreads"
    if spread:
        sign = 1 if p.pick == p.home else -1
        me, opp = (home, away) if sign > 0 else (away, home)
        mine = sign * (fair_market + adj)
        line = float(p.point)
        room = mine + line
        call = (f"The model has {me} winning by {mine:.1f}" if mine >= 0
                else f"The model has {me} losing by only {-mine:.1f}")
        call += (f", and they're getting {line:g} ({room:.1f} points of room)."
                 if line > 0 else f", {room:.1f} more than the {-line:g} they're giving.")
    else:
        over = str(p.pick).lower() != "under"
        sign = 1 if over else -1
        mine = fair_market + adj
        call = (f"The model puts this game at {mine:.1f} points, "
                f"{abs(mine - float(p.point)):.1f} {'over' if over else 'under'} "
                f"the {float(p.point):g} total.")

    def say(k, view, toward):
        if spread:
            t, u = (me, opp) if toward else (opp, me)
            if k == "eff":
                return (f"play-by-play efficiency, adjusted for opponents, has it "
                        f"{_fav(home, away, view)}: {_matchup(t, u, 'epa', ranks)}")
            if k == "elo":
                return f"Elo has it {_fav(home, away, view)}"
            # scores: fitted with a negative weight, so it pushes toward the
            # side the scoreboard rates BELOW the market
            return (f"{u}'s scores so far ({_fav(home, away, view)}) run ahead of "
                    f"what the line already gives them, and teams like that have "
                    f"tended to fall short of the spread")
        up = over == toward
        if k == "eff":
            # no number: the efficiency fit runs high on low totals and
            # low on high ones; its direction is what the backtest trusts
            units = _units(home, away, "epa", ranks, up)
            return (f"play-by-play efficiency points to {'more' if up else 'less'} "
                    f"scoring than the total" + (f": {units}" if units else ""))
        return f"scoring so far, adjusted for opponents, puts it at {view:.1f}"

    pro = sorted(((sign * v, k, view) for k, (v, view) in parts.items() if sign * v >= 0.2),
                 reverse=True)
    con = sorted(((-sign * v, k, view) for k, (v, view) in parts.items() if -sign * v >= 0.4),
                 reverse=True)
    why = [call]
    if pro:
        why.append("Why: " + "; ".join(say(k, view, True) for _, k, view in pro) + ".")
    if con:
        _, k, view = con[0]
        why.append("Working against it: " + say(k, view, False) + ".")
    return " ".join(w[0].upper() + w[1:] for w in why)


# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------
def build_picks(rows: pd.DataFrame, my_books, bankroll=1000.0, min_ev=MIN_EV,
                now=None, g: pd.DataFrame | None = None):
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    kick = pd.to_datetime(rows["commence_time"], utc=True, errors="coerce")
    rows = rows[(kick > now) & (kick < week_end(now))]
    g = cfb_model.load_features() if g is None else g
    mm, tm = outcome_models(g)
    fair = sel.consensus(rows, mm, tm)
    models = cfb_model.fit_models(g[g.margin.notna()])
    ids = match_games(fair, g[g.season == cfb_data.current_season(now)])
    print(f"matched {len(ids)} of {len(fair)} college games to the schedule")
    upcoming = g[g.game_id.isin(ids.values())].set_index("game_id")
    ranks = Ranks(upcoming) if len(upcoming) else None
    adj, ctx = [], {}
    for ev, gid in ids.items():
        f, r = fair[ev], upcoming.loc[gid]
        lead = (pd.Timestamp(f.commence) - now) / pd.Timedelta(days=1)
        c = ctx.setdefault(f.home_team, {"row": r})
        for kind, mkt, cap in (("spread", f.market_margin, SPREAD_CAP),
                               ("total", f.market_total, TOTAL_CAP)):
            m = models[kind]
            if np.isnan(mkt) or r[m.feats].isna().any():
                continue
            frame = r[m.feats].to_frame().T.astype(float)
            pts = float(np.clip(m.predict(frame, np.array([mkt]))[0], -cap, cap))
            c[kind] = (pts, m.parts(r, mkt))
            market = "spreads" if kind == "spread" else "totals"
            src = "combined" if kind == "spread" or lead >= TOTAL_MIN_LEAD_DAYS \
                else "combined_info"
            if abs(pts) >= 0.05:
                adj.append(sel.Adjustment(f.home_team, market, pts, src))
    sel.apply_adjustments(fair, adj)
    pol = sel.Policy(min_ev=min_ev)
    priced = sel.price_offers(rows, fair, mm, tm, pol)
    if priced.empty:
        return priced, fair, ids
    bettable = priced[priced.market.isin(["spreads", "totals"])
                      & priced.book.isin(my_books)]
    picks = sel.select(bettable, bankroll, pol)
    if len(picks):
        picks["win_chance"] = picks.p_win / (1 - picks.p_push)
        why = []
        for p in picks.itertuples():
            c = ctx.get(p.home, {})
            kind = "spread" if p.market == "spreads" else "total"
            if kind in c and "combined" in str(p.sources).split(","):
                pts, parts = c[kind]
                why.append(explain(p, c["row"], parts, ranks, pts, p.fair_market))
            else:
                fv = (f"put the total at {p.fair_market:.1f}" if p.market == "totals"
                      else f"make {p.home if p.fair_market >= 0 else p.away} a "
                           f"{abs(p.fair_market):.1f}-point favorite")
                why.append(f"{BOOK_NAME.get(p.book, p.book)} hasn't caught up "
                           f"with the sharp books, which {fv}.")
        picks["why"] = why
        picks["game_id"] = [ids.get(e) for e in picks.event_id]
        picks["book"] = picks.book.map(lambda b: BOOK_NAME.get(b, b))
    return picks, fair, ids


def grade(bets: list[dict], g: pd.DataFrame) -> list[dict]:
    done = g.dropna(subset=["margin"]).set_index("game_id")
    for b in bets:
        if b.get("result") or b.get("game_id") not in done.index:
            continue
        r = done.loc[b["game_id"]]
        b["result"] = ledger.grade_one(b, r.home_points, r.away_points)
        b["final"] = f"{int(r.away_points)}-{int(r.home_points)}"
        b["profit"] = round({"win": ledger.payout(b["stake"], b["price"]),
                             "loss": -b["stake"], "push": 0.0}[b["result"]], 2)
    return bets


def school_names(fair: dict, ids: dict, g: pd.DataFrame, history) -> dict:
    """Odds API name -> school name, so the page says "Ohio State", not
    "Buckeyes"."""
    s = g.set_index("game_id")
    out = {}
    for ev, gid in ids.items():
        out[fair[ev].home_team] = s.loc[gid, "home_team"]
        out[fair[ev].away_team] = s.loc[gid, "away_team"]
    for b in history:
        if b.get("game_id") in s.index:
            out.setdefault(b["home"], s.loc[b["game_id"], "home_team"])
            out.setdefault(b["away"], s.loc[b["game_id"], "away_team"])
    return out


COLS = ["away", "home", "commence", "market", "pick", "point", "price",
        "book", "stake", "win_chance", "ev", "why", "game_id"]


def render_html(picks, fair, made, bankroll, history, names) -> str:
    data = {
        "title": "College Edge", "names": names,
        "nav": {"href": "../", "text": "NFL picks →"},
        "updated": made, "bankroll": bankroll,
        "picks": picks[COLS].to_dict("records") if len(picks) else [],
        "fair": [{"away": f.away_team, "home": f.home_team,
                  "market_margin": f.market_margin, "margin": f.margin,
                  "market_total": f.market_total, "total": f.total}
                 for f in fair.values()
                 if not (np.isnan(f.market_margin) or np.isnan(f.market_total))],
        "history": {"summary": ledger.summary(history), "bets": history},
        "paper": None,
    }
    blob = json.dumps(data, default=str).replace("</", "<\\/")
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "dashboard.html")) as fh:
        return fh.read().replace("/*DATA*/", blob)


def render_alert(bets: list[dict], names: dict) -> str:
    n = lambda t: names.get(t, t)
    out = [f"**{len(bets)} new college bet{'s' if len(bets) != 1 else ''}**\n"]
    for b in bets:
        kick = pd.to_datetime(b["commence"], utc=True).tz_convert(
            "America/New_York").strftime("%a %-I:%M %p ET")
        pick = n(b["pick"]) if b["market"] == "spreads" else b["pick"]
        pt = f"{b['point']:+g}" if b["market"] == "spreads" else f"{b['point']:g}"
        out.append(f"- **{pick} {pt} ({int(b['price']):+d})** at {b['book']}, "
                   f"${b['stake']:,.2f} ({b['win_chance'] * 100:.0f}% to win) · "
                   f"{n(b['away'])} @ {n(b['home'])}, {kick}")
    out.append(f"\nDashboard: {DASHBOARD_URL}")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="site/cfb/index.html")
    ap.add_argument("--json", default="site/cfb/picks.json")
    ap.add_argument("--history", default="site/cfb/history.json")
    ap.add_argument("--alert", default="alert.md",
                    help="appended to when this run finds new bets")
    ap.add_argument("--bankroll", type=float, default=1000.0)
    ap.add_argument("--books", default=os.environ.get("MY_BOOKS") or MY_BOOKS)
    ap.add_argument("--lines-csv", default="")
    a = ap.parse_args()
    if a.lines_csv:
        rows = sel.latest_snapshot(pd.read_csv(a.lines_csv))
    else:
        key = (os.environ.get("ODDS_API_KEY") or "").strip().strip('"\'')
        if "=" in key:
            key = key.split("=", 1)[1].strip().strip('"\'')
        if not key:
            raise SystemExit("set ODDS_API_KEY")
        rows = fetch_rows(key)
    books = [b.strip() for b in a.books.split(",") if b.strip()]
    now = datetime.now(timezone.utc)
    g = cfb_model.load_features()
    picks, fair, ids = build_picks(rows, books, a.bankroll, now=now, g=g)
    made = now.strftime("%a %b %-d, %H:%M UTC")
    for path in (a.out, a.json, a.history):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out = picks[COLS].to_dict("records") if len(picks) else []
    out = json.loads(json.dumps(out, default=str))
    history = ledger.load(a.history)
    seen = {ledger._key(b) for b in history}
    history = ledger.merge(history, out, now, now.isoformat())
    history = grade(history, g)
    ledger.save(a.history, history)
    names = school_names(fair, ids, g, history)
    with open(a.out, "w") as fh:
        fh.write(render_html(picks, fair, made, a.bankroll, history, names))
    with open(a.json, "w") as fh:
        json.dump({"updated": made, "picks": out}, fh, indent=2, default=str)
    fresh = [p for p in out if ledger._key(p) not in seen]
    if fresh:
        gap = "\n" if os.path.exists(a.alert) else ""   # after the NFL alert
        with open(a.alert, "a") as fh:
            fh.write(gap + render_alert(fresh, names))
    print(f"{len(out)} college bets, {len(fresh)} new")
    for p in out:
        print(f"  {p['pick']} {p['point']} ({p['price']}) at {p['book']}: "
              f"${p['stake']} | {p['win_chance']:.1%} | {p['why']}")


if __name__ == "__main__":
    main()
