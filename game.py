"""
One game through the model, on demand: what it makes the spread and total,
the best price at your books for each side, and whether that's a bet.

    ODDS_API_KEY=... python game.py Chiefs
    python game.py "KC SEA" --lines-csv lines.csv

Same pipeline as picks.py (build_picks), so the numbers match what a picks
run would say about this game; it just reports every side, not only the
ones that clear the EV bar. Run it from the "Check one game" workflow.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

import pandas as pd

import nfl_combined
import picks as pk
import selector as sel

MARKETS = ("spreads", "totals")


def _words(q: str) -> list[str]:
    return [w for w in q.lower().replace("@", " ").replace("/", " ")
            .replace(",", " ").split() if w not in {"at", "vs", "v"}]


def _team_matches(word: str, team: str) -> bool:
    abbr = pk.ODDS_TEAM.get(team, "").lower()
    return word == abbr or (len(word) >= 3 and word in team.lower())


def find_event(rows: pd.DataFrame, query: str):
    """event_id of the soonest game where every word names one of its teams
    ("Chiefs", "kc sea", "Chiefs at Seahawks"), or None."""
    words = _words(query)
    if not words or rows.empty:
        return None
    games = (rows.drop_duplicates("event_id")
                 .sort_values("commence_time"))
    for g in games.itertuples():
        if all(_team_matches(w, g.home_team) or _team_matches(w, g.away_team)
               for w in words):
            return g.event_id
    return None


def _fav(home: str, away: str, margin: float) -> str:
    team = home if margin >= 0 else away
    return f"{pk._nick(team)} by {abs(margin):.1f}"


def _label(r) -> str:
    if r.market == "spreads":
        return f"{pk._nick(r.pick)} {r.point:+g}"
    return f"{r.pick} {r.point:g}"


def render(f, priced: pd.DataFrame, bets: pd.DataFrame, ctx: dict,
           books, min_ev: float, made: str) -> str:
    """The game report, as Markdown."""
    kick = pd.to_datetime(f.commence, utc=True).tz_convert(
        "America/New_York").strftime("%a %b %-d, %-I:%M %p ET")
    c = ctx.get(f.home_team, {})
    out = [f"# {f.away_team} @ {f.home_team}\n",
           f"_{kick} · checked {made}_\n"]

    out.append("## What the model makes it\n")
    out.append("| | Sharp books | Model |\n|---|---|---|")
    if pd.notna(f.market_margin):
        out.append(f"| Spread | {_fav(f.home_team, f.away_team, f.market_margin)}"
                   f" | {_fav(f.home_team, f.away_team, f.margin)} |")
    if pd.notna(f.market_total):
        out.append(f"| Total | {f.market_total:.1f} | {f.total:.1f} |")
    out.append("")
    if c.get("ratings"):
        out.append(f"Team ratings plus the injury report have it "
                   f"{_fav(f.home_team, f.away_team, c['ratings']['margin'])}"
                   f"; the model moves the sharp line a fifth of the way "
                   f"there (at most 1 point).\n")

    mine = priced[priced["market"].isin(MARKETS)
                  & priced["book"].isin(books)]
    best = (mine.sort_values("ev", ascending=False)
                .drop_duplicates(["market", "pick"]))
    bet_keys = {(b.market, b.pick) for b in bets.itertuples()} \
        if len(bets) else set()

    out.append("## Best price at your books\n")
    if best.empty:
        out.append("None of your books has this game up yet.\n")
    else:
        out.append("| Side | Price | Book | Win chance | EV |\n"
                   "|---|---|---|---|---|")
        for m in MARKETS:
            for r in best[best["market"] == m].sort_values(
                    "ev", ascending=False).itertuples():
                wc = r.p_win / (1 - r.p_push) if r.p_push < 1 else 0
                mark = " ✅" if (r.market, r.pick) in bet_keys else ""
                out.append(f"| {_label(r)}{mark} | {int(r.price):+d} | "
                           f"{pk.BOOK_NAME.get(r.book, r.book)} | "
                           f"{wc * 100:.1f}% | {r.ev * 100:+.1f}% |")
        out.append("")

    out.append("## Verdict\n")
    for m in MARKETS:
        side = best[best["market"] == m]
        name = "Spread" if m == "spreads" else "Total"
        if side.empty:
            continue
        top = side.iloc[0]
        b = bets[(bets["market"] == m)] if len(bets) else bets
        if len(b):
            p = b.iloc[0]
            out.append(f"**{name}: bet {_label(top)} ({int(top.price):+d}) at "
                       f"{pk.BOOK_NAME.get(top.book, top.book)}, "
                       f"${p.stake:,.2f}.** {p.why}\n")
            continue
        verdict = ("pass" if top.ev < min_ev / 2
                   else f"lean {_label(top)}, but under the bar")
        line = (f"**{name}: {verdict}.** The best side is {_label(top)} "
                f"at {top.ev * 100:+.1f}% EV; a bet needs {min_ev * 100:.0f}%.")
        if m == "totals" and "why_total" in c:
            toward = -1 if str(top.pick).lower() == "under" else 1
            line += " " + nfl_combined.explain(c["why_total"], c["x"], c["h"],
                                               c["a"], "totals", toward=toward)
        if m == "totals" and c.get("wind"):
            line += " " + c["wind"]
        out.append(line.strip() + "\n")

    out.append("---\nWin chances are model estimates, not guarantees. The "
               "model hasn't yet shown an edge over closing lines. 21+ "
               "where legal; 1-800-GAMBLER.\n")
    return "\n".join(out)


def run(rows: pd.DataFrame, query: str, books, bankroll: float = 1000.0,
        min_ev: float = 0.02, wind_fn=None) -> str:
    now = datetime.now(timezone.utc)
    ev = find_event(pk.this_week(rows, now) if len(rows) else rows, query)
    if ev is None:
        later = find_event(rows, query)
        upcoming = pk.this_week(rows, now).drop_duplicates("event_id")
        lst = "\n".join(f"- {g.away_team} @ {g.home_team}"
                        for g in upcoming.itertuples()) or "- none"
        why = ("That game has kicked off or isn't this week, and the model "
               "only prices games before kickoff in the current week."
               if later is not None else f"No game matches \"{query}\".")
        return f"# No game found\n\n{why}\n\nThis week's games:\n\n{lst}\n"
    ctx = {}
    bets, fair, priced = pk.build_picks(rows[rows["event_id"] == ev], books,
                                        bankroll, min_ev, wind_fn=wind_fn,
                                        ctx_out=ctx)
    if priced.empty:
        f = next(iter(fair.values()), None)
        title = f"{f.away_team} @ {f.home_team}" if f else query
        return (f"# {title}\n\nNot enough books have lines up to price it "
                f"(need {sel.Policy().min_books}).\n")
    return render(fair[ev], priced, bets, ctx, books, min_ev,
                  now.strftime("%a %b %-d, %H:%M UTC"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("game", nargs="+", help='team name(s), e.g. Chiefs')
    ap.add_argument("--md", default="game.md")
    ap.add_argument("--bankroll", type=float, default=1000.0)
    ap.add_argument("--min-ev", type=float, default=0.02)
    ap.add_argument("--books",
                    default=os.environ.get("MY_BOOKS") or pk.MY_BOOKS)
    ap.add_argument("--lines-csv", default="",
                    help="use a logged snapshot instead of the live feed")
    a = ap.parse_args()
    if a.lines_csv:
        rows = sel.latest_snapshot(pd.read_csv(a.lines_csv))
    else:
        key = (os.environ.get("ODDS_API_KEY") or "").strip().strip('"\'')
        if "=" in key:
            key = key.split("=", 1)[1].strip().strip('"\'')
        if not key:
            raise SystemExit("set ODDS_API_KEY")
        rows = pk.fetch_rows(key)
    books = [b.strip() for b in a.books.split(",") if b.strip()]
    md = run(rows, " ".join(a.game), books, a.bankroll, a.min_ev)
    with open(a.md, "w") as fh:
        fh.write(md)
    sys.stdout.write(md)


if __name__ == "__main__":
    main()
