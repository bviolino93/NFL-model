"""
The handicapper: a last look at each NFL bet the way a sharp human bettor
would, ported from the college model (Brian, 2026-11-01: "I want it to pick
up what the college model was picking up").

Two jobs each picks run:

  * every model bet goes to Claude with the matchup: the model's margin
    split into QB, passing, running, defense and home field, each team's
    season profile with league ranks, this season's results with how each
    game went against its closing spread and total, ATS and over/under
    records, common opponents, the injuries the model counts and the wind.
    It keeps the bet only with an affirmative matchup case ("bet"), else
    "pass" drops it, as college's MATCHUP_ALL does
  * the model's biggest leans that fall short of a bet (up to MAX_LEANS
    games) get the same brief; Claude may take either side ("bet" or
    "other") at LEAN_STAKE, $5 on $1,000, or pass (most should)

No web searches: nflverse has the injury report, rosters and closing lines,
so a review is about two cents. Reviews are cached in handicap.json on the
picks branch by a hash of the brief, so a game is reviewed again only when
its line, price or news changes. Spend stops for the month at
MONTHLY_BUDGET (college stops at $40; Brian's API ceiling is $50).

Unbacktested: nobody has shown judgment calls like this beat the close, so
the weekly closing-line review is the judge. HANDICAP = False turns it off;
without the ANTHROPIC_API_KEY secret bets go out unreviewed, as before.
"""

from __future__ import annotations

import hashlib
import json
import os
import re

import numpy as np
import pandas as pd

HANDICAP = True
MODEL = "claude-sonnet-5-5"
EFFORT = "low"
MONTHLY_BUDGET = 8.0
# $ per million tokens (Sonnet 5.5), for the running total
PRICE = {"input": 2.0, "output": 10.0}

LEAN_STAKE = 0.005          # of bankroll ($5 on $1,000) for a handicapper's own pick
MAX_LEANS = 8               # games reviewed per run beyond the bets, biggest lean first
MIN_SPREAD_LEAN = 1.5       # ratings-vs-market gap, points, for a spread lean
MIN_TOTAL_LEAN = 0.3        # model-vs-market gap, points, for a totals lean
LEAN_HOURS = 72             # only games kicking off within this window
SHARED = [{"LA", "LAC"}, {"NYG", "NYJ"}]   # teams that share a home stadium

BET_SYSTEM = """You are a sharp NFL handicapper making the final call on a \
bet a model has flagged. Brian wants every bet to rest on analysis of this \
game, not on a model number alone. Break down the matchup: how each offense \
(quarterback, passing, rushing) lines up against the other defense, both \
teams' scoring and points allowed, recent form and who it came against, how \
their games have gone against the closing spread and total, injuries and \
the situation (rest, travel, division game, weather). The betting market is \
efficient, so bet only when that analysis makes a clear case for this side \
and nothing the model can't see cuts against it; pass when it is unclear or \
points the other way. Don't invent facts that aren't in what you're given.

Answer with one ```json block and nothing after it:
{"decision": "bet|pass", "reason": "<two or three sentences on the matchup case, plain English>"}"""

LEAN_SYSTEM = """You are a sharp NFL handicapper. The model only leans to the \
side shown (not enough for a bet on its numbers alone), so this is your call \
on the football, and you may take either side: the matchup (quarterback, \
passing, rushing, each defense), recent form and who it came against, how \
each team's games have gone against the closing spread and total, common \
opponents, injuries and the situation. The model's numbers can be wrong, so \
if the football clearly points to the other side, take it. Answer "bet" for \
the side shown, "other" for the other side, or "pass". Bet only when the \
football case is clearly strong and nothing cuts against it; most of these \
should be passes. Don't invent facts that aren't in what you're given.

Answer with one ```json block and nothing after it:
{"decision": "bet|other|pass", "reason": "<two or three sentences on the football case for the side you pick, plain English>"}"""


# ------------------------------------------------------------------ state

def load(path: str) -> dict:
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def save(path: str, state: dict) -> None:
    rev = state.get("_reviews", {})
    if len(rev) > 300:                      # keep the cache to the last 300
        for k in sorted(rev, key=lambda k: rev[k].get("at", ""))[:len(rev) - 300]:
            rev.pop(k)
    with open(path, "w") as fh:
        json.dump(state, fh, indent=2, default=str)


# ------------------------------------------------------------------ brief

def _results(games: pd.DataFrame, team: str, before) -> list[dict]:
    """This season's finished games for `team` (nflverse abbreviation),
    oldest first, with each result against the closing spread and total."""
    g = games[((games.home_team == team) | (games.away_team == team))
              & games.home_score.notna()]
    g = g[pd.to_datetime(g.gameday) < pd.Timestamp(before).tz_localize(None).normalize()]
    if not len(g):
        return []
    g = g[g.season == g.season.max()].sort_values(["week"])
    out = []
    for r in g.itertuples():
        home = r.home_team == team
        pf, pa = (r.home_score, r.away_score) if home else (r.away_score, r.home_score)
        line = r.spread_line if home else -r.spread_line     # team's expected margin
        out.append({"week": int(r.week), "opp": r.away_team if home else r.home_team,
                    "where": "vs" if home else "at", "pf": int(pf), "pa": int(pa),
                    "ats": None if pd.isna(line) else float(pf - pa - line),
                    "ou": None if pd.isna(r.total_line) else float(pf + pa - r.total_line)})
    return out


def _record(xs, key) -> str:
    v = [x[key] for x in xs if x[key] is not None]
    w, lo = sum(a > 0 for a in v), sum(a < 0 for a in v)
    return f"{w}-{lo}" + (f"-{len(v) - w - lo}" if len(v) - w - lo else "")


def form_text(games, team: str, name: str, before) -> str:
    res = _results(games, team, before)
    if not res:
        return f"{name}: no results on file this season."
    w = sum(x["pf"] > x["pa"] for x in res)
    t = sum(x["pf"] == x["pa"] for x in res)
    pf = np.mean([x["pf"] for x in res])
    pa = np.mean([x["pa"] for x in res])
    last = "; ".join(
        f"wk{x['week']} {x['where']} {x['opp']} {'W' if x['pf'] > x['pa'] else 'L' if x['pf'] < x['pa'] else 'T'} "
        f"{x['pf']}-{x['pa']}"
        + (f" ({'covered' if x['ats'] > 0 else 'failed to cover' if x['ats'] < 0 else 'push'} by "
           f"{abs(x['ats']):g}" if x["ats"] is not None else " (")
        + (f", {'over' if x['ou'] > 0 else 'under' if x['ou'] < 0 else 'push'} by {abs(x['ou']):g})"
           if x["ou"] is not None else ")")
        for x in res[-5:])
    return (f"{name} is {w}-{len(res) - w - t}{f'-{t}' if t else ''}, scoring {pf:.1f} and allowing "
            f"{pa:.1f} a game; {_record(res, 'ats')} against the spread, over/under "
            f"{_record(res, 'ou')}. Latest (vs closing lines): {last}.")


def common_text(games, a: str, b: str, before) -> str:
    ra, rb = {}, {}
    for x in _results(games, a, before):
        ra.setdefault(x["opp"], []).append(x["pf"] - x["pa"])
    for x in _results(games, b, before):
        rb.setdefault(x["opp"], []).append(x["pf"] - x["pa"])
    both = sorted((set(ra) & set(rb)) - {a, b})
    if not both:
        return "No common opponents this season."
    return "Common opponents (margins): " + "; ".join(
        f"{o}: {a} {', '.join(f'{m:+d}' for m in ra[o])}, {b} {', '.join(f'{m:+d}' for m in rb[o])}"
        for o in both) + "."


def profile_text(table: pd.DataFrame | None, a: str, b: str) -> str:
    """Season power ratings (points a game vs average, opponent-adjusted,
    recent games weighted more) with league ranks."""
    if table is None or not len(table):
        return ""
    t = table.set_index("team")
    n = len(t)
    out = []
    for team in (a, b):
        if team not in t.index:
            continue
        r = t.loc[team]
        rk = {c: int((t[c] > r[c]).sum() + 1) for c in ("off_pts", "def_pts", "net_pts")}
        out.append(f"{team}: offense {r.off_pts:+.1f} (#{rk['off_pts']} of {n}), defense "
                   f"{r.def_pts:+.1f} (#{rk['def_pts']}), overall {r.net_pts:+.1f} (#{rk['net_pts']})")
    return "; ".join(out) + "." if out else ""


def _side(market, pick, point) -> str:
    if market == "h2h":
        return f"{pick} moneyline"
    return f"{pick} {point:+g}" if market == "spreads" else f"{pick} {point:g}"


def brief(p, ctx: dict, games: pd.DataFrame | None, table, abbr: dict,
          injuries: list | None = None) -> str:
    c = ctx.get(p.home, {})
    m = c.get("ratings") or {}
    lines = [f"Game: {p.away} at {p.home}, kickoff {pd.Timestamp(p.commence):%a %b %-d %H:%M UTC}",
             f"Bet: {_side(p.market, p.pick, p.point)} at {int(p.price):+d}, {p.book}",
             f"Model: win chance {p.win_chance:.1%}, expected value {p.ev:+.1%}",
             f"Sharp market: home margin {p.fair_market:+.1f}" if p.market != "totals"
             else f"Sharp market: total {p.fair_market:.1f}"]
    if getattr(p, "why", None):
        lines.append(f"Model's reasoning: {p.why}")
    if m:
        lines.append(
            f"Ratings (points, home team's view): home by {m['margin']:+.1f} = ratings "
            f"{m['ratings']:+.1f} + injuries {m['injuries']:+.1f}; pieces: home field "
            f"{m.get('hfa', 0):+.1f}, QB {m.get('qb', 0):+.1f}, passing {m.get('pass_off', 0):+.1f}, "
            f"rushing {m.get('run_off', 0):+.1f}, pass defense {m.get('pass_def', 0):+.1f}, run defense "
            f"{m.get('run_def', 0):+.1f}. Starting QBs: {m.get('away_qb') or 'not confirmed'} (away), "
            f"{m.get('home_qb') or 'not confirmed'} (home).")
    if c.get("why_total") is not None and p.market == "totals":
        lines.append("Totals model factors: " + json.dumps(c.get("why_total"), default=str)[:600])
    if c.get("wind"):
        lines.append(c["wind"])
    ha, aa = abbr.get(p.home), abbr.get(p.away)
    if {ha, aa} in SHARED:
        lines.append("Both teams play home games in the same stadium, so home field is worth "
                     "little here (the ratings above still count it).")
    if games is not None and ha and aa:
        lines += [form_text(games, aa, p.away, p.commence),
                  form_text(games, ha, p.home, p.commence),
                  common_text(games, aa, ha, p.commence)]
        prof = profile_text(table, aa, ha)
        if prof:
            lines.append("Season power ratings (pts/game vs average): " + prof)
    lines.append("Injuries the model counts: " + (json.dumps(injuries) if injuries else "none"))
    return "\n".join(lines)


# ------------------------------------------------------------------ calls

def _parse(text: str, default: str) -> dict:
    m = re.findall(r"```json\s*(\{.*?\})\s*```", text, re.S)
    try:
        d = json.loads(m[-1]) if m else {}
    except json.JSONDecodeError:
        d = {}
    dec = str(d.get("decision", "")).lower()
    return {"decision": dec if dec in ("bet", "pass", "other") else default,
            "reason": str(d.get("reason", "")).strip()}


def _ask(client, text: str, system: str) -> tuple[dict, float]:
    resp = client.beta.messages.create(
        model=MODEL, max_tokens=2000, system=system,
        messages=[{"role": "user", "content": text}],
        output_config={"effort": EFFORT},
        betas=["server-side-fallback-2026-07-01"], fallbacks="default")
    u = getattr(resp, "usage", None)
    usd = 0.0
    if u is not None:
        usd = (getattr(u, "input_tokens", 0) * PRICE["input"]
               + getattr(u, "output_tokens", 0) * PRICE["output"]) / 1e6
    if resp.stop_reason == "refusal":
        return {"decision": "pass", "reason": ""}, usd
    return _parse("".join(b.text for b in resp.content if b.type == "text"), "pass"), usd


def review(picks: pd.DataFrame, ctx: dict, state: dict, games=None, table=None,
           abbr: dict | None = None, now=None, client=None, leans: bool = False,
           injuries=None) -> pd.DataFrame:
    """Bets: keep only those with the handicapper's matchup case, reason
    added to `why`. Leans (leans=True): keep only the handicapper's own
    picks, flipped to the other side on "other". Without the API key or
    past the monthly budget, bets pass through unreviewed and leans are
    dropped. Never fails the run."""
    if not len(picks):
        return picks
    off = picks.iloc[:0] if leans else picks
    if not HANDICAP:
        return off
    now = pd.Timestamp(now or pd.Timestamp.now(tz="UTC"))
    month = f"{now:%Y-%m}"
    b = state.get("_budget", {})
    spent = b.get("usd", 0.0) if b.get("month") == month else 0.0
    done = state.setdefault("_reviews", {})
    keep, whys, flips = [], [], []
    for p in picks.itertuples():
        inj = injuries(p) if injuries else None
        text = brief(p, ctx, games, table, abbr or {}, inj)
        if leans:
            text = "LEAN ONLY (the model is short of a bet).\n" + text
            if pd.notna(getattr(p, "opp_pick", None)):
                text += (f"\nOther side: {_side(p.market, p.opp_pick, p.opp_point)} at "
                         f"{int(p.opp_price):+d}, {p.opp_book}")
        else:
            text = "MODEL BET: bet only if the matchup analysis supports it.\n" + text
        key = hashlib.sha1(text.encode()).hexdigest()[:16]
        got = done.get(key)
        if got is None:
            if spent >= MONTHLY_BUDGET:
                print("handicap: monthly budget reached; bets go out unreviewed")
                return off
            if client is None:
                if not os.environ.get("ANTHROPIC_API_KEY"):
                    print("handicap: no ANTHROPIC_API_KEY, skipping")
                    return off
                try:
                    import anthropic
                    client = anthropic.Anthropic()
                except Exception as e:
                    print(f"handicap: no client ({e})")
                    return off
            try:
                got, usd = _ask(client, text, LEAN_SYSTEM if leans else BET_SYSTEM)
                spent += usd
                done[key] = {**got, "at": now.isoformat()}
            except Exception as e:          # a review is a check, never a reason to fail
                print(f"handicap: review failed ({type(e).__name__}: {e})")
                got = None
        dec = (got or {}).get("decision", "pass" if leans else "bet")
        reason = (got or {}).get("reason", "")
        print(f"handicap{' (lean)' if leans else ''}: {_side(p.market, p.pick, p.point)} "
              f"-> {dec}: {reason}")
        if leans:
            other = dec == "other" and pd.notna(getattr(p, "opp_pick", None))
            keep.append(dec == "bet" or other)
            flips.append(other)
            whys.append(f"Handicapper's pick: {reason} (Against the model, which leaned "
                        f"{_side(p.market, p.pick, p.point)}; small stake.)" if other else
                        f"Handicapper's pick: {reason} (The model leans this way too, by "
                        f"less than a bet needs; small stake.)")
        else:
            flips.append(False)
            keep.append(got is None or dec != "pass")
            whys.append(f"{p.why} Handicapper: {reason}" if reason else p.why)
    state["_budget"] = {"month": month, "usd": round(spent, 4)}
    out = picks.assign(why=whys)
    flip = np.array(flips, dtype=bool)
    if flip.any():
        for c in ("pick", "point", "price", "book", "ev", "win_chance"):
            out[c] = out[c].astype(object)
            out.loc[flip, c] = out.loc[flip, "opp_" + c].values
    return out[np.array(keep, dtype=bool)].reset_index(drop=True)


# ------------------------------------------------------------------ leans

def lean_candidates(priced: pd.DataFrame, ctx: dict, my_books, picks: pd.DataFrame,
                    now, bankroll: float, lean_fn) -> pd.DataFrame:
    """Games the model leans on but doesn't bet: the best price at my books
    for the side it leans to (spreads: the ratings-vs-market gap;
    totals: the model's adjustment), not already a pick, kicking off
    within LEAN_HOURS. The biggest MAX_LEANS leans, one per game, with
    the best offer on the other side too."""
    if priced is None or priced.empty:
        return pd.DataFrame()
    p = priced[priced.book.isin(my_books) & priced.market.isin(["spreads", "totals"])].copy()
    if p.empty:
        return p
    raw = []
    for r in p.itertuples():
        m = (ctx.get(r.home, {}) or {}).get("ratings")
        if r.market == "spreads" and m and pd.notna(r.fair_market):
            side = 1 if r.pick == r.home else -1
            raw.append(side * (m["margin"] - r.fair_market))
        else:
            raw.append(np.nan)
    p["lean_pts"] = np.where(p.market == "spreads", raw, lean_fn(p))
    every = p.sort_values("ev", ascending=False).drop_duplicates(["event_id", "market", "pick"])
    kick = pd.to_datetime(p.commence, utc=True)
    now = pd.Timestamp(now)
    need = np.where(p.market == "spreads", MIN_SPREAD_LEAN, MIN_TOTAL_LEAN)
    p = p[(kick > now) & (kick <= now + pd.Timedelta(hours=LEAN_HOURS))
          & (p.lean_pts >= need) & (p.ev >= -0.04)]
    if len(picks) and "event_id" in picks:
        taken = set(picks.event_id)
        p = p[~p.event_id.isin(taken)]
    if p.empty:
        return p
    p = (p.sort_values("ev", ascending=False).drop_duplicates(["event_id", "market"])
          .sort_values("lean_pts", ascending=False).drop_duplicates("event_id").head(MAX_LEANS))
    opp = {}
    for r in every.itertuples():
        opp.setdefault((r.event_id, r.market), []).append(r)
    for c in ("pick", "point", "price", "book", "ev", "win_chance"):
        vals = []
        for r in p.itertuples():
            o = next((x for x in opp.get((r.event_id, r.market), []) if x.pick != r.pick), None)
            vals.append(None if o is None else
                        o.p_win / (1 - o.p_push) if c == "win_chance" else getattr(o, c))
        p["opp_" + c] = vals
    p["stake"] = round(bankroll * LEAN_STAKE, 2)
    p["win_chance"] = p.p_win / (1 - p.p_push)
    p["why"] = [f"The model leans {r.pick} by {r.lean_pts:.1f} points, short of a bet."
                for r in p.itertuples()]
    return p.reset_index(drop=True)


def power_table(season: int, week: int):
    """League power ratings going into `week` (nfl_power), for the briefs."""
    from backtest import home_field_before
    from nfl_data import graded_games, load_team_games
    from nfl_power import PowerConfig, fit_power
    from nfl_data import load_games
    g = load_games()
    tg = load_team_games(range(season - 1, season + 1))
    pr = fit_power(tg, season, week, home_field_before(graded_games(g), season, week),
                   PowerConfig())
    return pr.table()
