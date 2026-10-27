"""
Weekly review: how the bets are doing, and a refit of the totals model.

Runs every Tuesday (.github/workflows/review.yml), after Monday night's
game is graded. Two jobs:

1. Refit. combined_model.json holds the weights the picks use for totals
   (nfl_combined.export_model). They are refitted on every graded game
   through last week, and the new weights replace the old ones only if
   the recipe still holds up on games it never trained on:

     * walk-forward 2022-now (each season fitted on earlier seasons only),
       totals the model calls at 52%+ win at least 52.4%, the break-even
       at -110 (the report also shows whether its predictions sit closer
       to the result than the closing total on average: they don't quite,
       since most games' small adjustments are noise and only the
       confident calls carry the edge);
     * and the new weights move a typical game by under half a point from
       the current ones. One week of games barely shifts a ridge fitted on
       3,000+; a bigger jump means bad data, so the old weights stay and
       the report says why.

   Betting rules (thresholds, stakes, which markets) are never touched:
   a few dozen bets of wins and losses are mostly luck.

2. Report, posted to the "Picks alerts" issue: record, units and
   closing-line value (CLV) per market, NFL and college, plus the no-money
   spread test, and the biggest misses with a known cause (a QB the bet
   didn't know about, a line that ran away from it). CLV is the early
   read: points the closing line moved toward the bet. A market whose CLV
   is negative over 30+ bets gets flagged.

    python review.py --history site/history.json --paper site/paper.json \\
        --cfb-history cfb/history.json --cfb-closes cfb/closes.json \\
        --refit --out review.md
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

UNIT = 10.0            # 1 unit = 1% of the $1,000 bankroll the picks assume
FLAG_MIN_BETS = 30     # CLV below zero over this many bets gets flagged
BREAK_EVEN = 110 / 210
MAX_SHIFT = 0.5        # points: mean change in a game's total from new weights
MISS_PTS = 10.0        # a loss by this much against the number is a big miss
LINE_MOVE = 1.5        # CLV this far below zero counts as the line running away


# ----------------------------------------------------------------------
# Closing-line value
# ----------------------------------------------------------------------
def clv(bet: dict, close_spread_home: float | None,
        close_total: float | None) -> float | None:
    """Points the close moved toward the bet, from the line it was alerted
    at. Spread bets: our side's point minus its closing point (took +3,
    closed +2 = +1). Totals: over at 44 closing 45 = +1."""
    point = bet.get("first_point", bet["point"])
    if bet["market"] == "totals":
        if close_total is None or pd.isna(close_total):
            return None
        d = close_total - point
        return round(float(d if bet["pick"] == "Over" else -d), 2)
    if close_spread_home is None or pd.isna(close_spread_home):
        return None
    close = close_spread_home if bet["pick"] == bet["home"] \
        else -close_spread_home
    return round(float(point - close), 2)


def nfl_closes(bets: list[dict], games: pd.DataFrame, abbr: dict) -> list[dict]:
    """Adds close (home spread, total) and clv from nflverse games.csv,
    whose spread_line (home margin) and total_line are the closing lines."""
    g = games.assign(day=pd.to_datetime(games["gameday"]).dt.normalize())
    for b in bets:
        day = (pd.Timestamp(b["commence"]).tz_convert("America/New_York")
               .tz_localize(None).normalize())
        m = g[(g.home_team == abbr.get(b["home"]))
              & (g.away_team == abbr.get(b["away"]))
              & ((g.day - day).abs() <= pd.Timedelta(days=1))]
        if m.empty:
            continue
        m = m.iloc[0]
        b["game_id"] = m.game_id
        sp = -float(m.spread_line) if pd.notna(m.spread_line) else None
        tot = float(m.total_line) if pd.notna(m.total_line) else None
        b["close_spread"], b["close_total"] = sp, tot
        b["clv"] = clv(b, sp, tot)
    return bets


def cfb_closes(bets: list[dict], closes: dict) -> list[dict]:
    """College closes are the last consensus line the picks runs saw
    before kickoff (closes.json on the CFB-model picks branch)."""
    for b in bets:
        c = closes.get(f"{b['away']}|{b['home']}|{str(b['commence'])[:10]}")
        if not c:
            continue
        b["close_spread"], b["close_total"] = c.get("spread"), c.get("total")
        b["clv"] = clv(b, b["close_spread"], b["close_total"])
    return bets


# ----------------------------------------------------------------------
# Record by market
# ----------------------------------------------------------------------
def market_stats(bets: list[dict]) -> dict:
    done = [b for b in bets if b.get("result")]
    w = sum(b["result"] == "win" for b in done)
    lo = sum(b["result"] == "loss" for b in done)
    staked = sum(b.get("stake", 0) for b in done)
    profit = sum(b.get("profit") or 0 for b in done)
    c = [b["clv"] for b in bets if b.get("clv") is not None
         and pd.Timestamp(b["commence"]) <= pd.Timestamp.now(tz="UTC")]
    return {"bets": len(done), "wins": w, "losses": lo,
            "pushes": len(done) - w - lo, "units": round(profit / UNIT, 2),
            "roi": round(profit / staked, 4) if staked else None,
            "clv_n": len(c),
            "clv": round(float(np.mean(c)), 2) if c else None,
            "beat_close": round(float(np.mean([x > 0 for x in c])), 3)
            if c else None}


def flagged(s: dict) -> bool:
    return s["clv_n"] >= FLAG_MIN_BETS and s["clv"] is not None \
        and s["clv"] < 0


# ----------------------------------------------------------------------
# Big misses
# ----------------------------------------------------------------------
def _last(name) -> str:
    return str(name).split()[-1].lower() if isinstance(name, str) else ""


def qb_surprise(b: dict, games: pd.DataFrame, abbr: dict) -> str | None:
    """A team in the bet started a QB other than last game's, and the bet
    never mentions him: news after the bet (or a late scratch)."""
    if "game_id" not in b or "home_qb_name" not in games:
        return None
    g = games.set_index("game_id")
    if b["game_id"] not in g.index:
        return None
    row = g.loc[b["game_id"]]
    said = (b.get("why", "") + " " + json.dumps(b.get("injuries", []))).lower()
    teams = [b["home"], b["away"]] if b["market"] == "totals" else [b["pick"]]
    # For a spread bet, the other side's QB matters too.
    if b["market"] != "totals":
        teams.append(b["away"] if b["pick"] == b["home"] else b["home"])
    played = games[games.home_score.notna()].sort_values(["season", "week"])
    for team in teams:
        t = abbr.get(team)
        side = "home" if t == row.home_team else "away"
        qb = row.get(f"{side}_qb_name")
        prev = played[((played.home_team == t) | (played.away_team == t))
                      & ((played.season < row.season)
                         | ((played.season == row.season)
                            & (played.week < row.week)))].tail(1)
        if prev.empty or not isinstance(qb, str):
            continue
        p = prev.iloc[0]
        was = p.home_qb_name if p.home_team == t else p.away_qb_name
        if isinstance(was, str) and _last(was) != _last(qb) \
                and _last(qb) not in said:
            return f"{qb} started for {t} instead of {was}, which the bet didn't account for"
    return None


def big_misses(bets: list[dict], games: pd.DataFrame | None = None,
               abbr: dict | None = None) -> list[str]:
    out = []
    for b in bets:
        if b.get("result") != "loss" or not b.get("final"):
            continue
        a, h = (int(x) for x in b["final"].split("-"))
        if b["market"] == "totals":
            by = abs(a + h - b["point"])
        else:
            mine, theirs = (h, a) if b["pick"] == b["home"] else (a, h)
            by = theirs - mine - (b["point"] or 0)
        why = []
        if games is not None:
            q = qb_surprise(b, games, abbr or {})
            if q:
                why.append(q)
        if b.get("clv") is not None and b["clv"] <= -LINE_MOVE:
            why.append(f"the line moved {-b['clv']:.1f} pts against it "
                       "before kickoff, so the market knew something")
        if why or by >= MISS_PTS:
            out.append(f"{_label(b)} lost by {by:.1f} vs the number "
                       f"({b['final']}): "
                       + ("; ".join(why) if why else "no known cause, "
                          "just a bad beat"))
    return out


def _label(b: dict) -> str:
    game = f"{b['away']} @ {b['home']}"
    if b["market"] == "totals":
        return f"{game} {b['pick']} {b['point']}"
    pt = b["point"]
    return f"{game}, {b['pick']} {'+' if pt > 0 else ''}{pt}"


# ----------------------------------------------------------------------
# Refit
# ----------------------------------------------------------------------
def refit_check(fr: pd.DataFrame, current: dict, split: int = 2022) -> dict:
    """Walk-forward check of the totals recipe through the latest graded
    week, and how far candidate weights fitted on everything move a game
    from the current ones. Returns the candidate and the verdict."""
    import nfl_combined as nc
    from nfl_data import load_games
    from selector import build_outcome_models
    wf, _ = nc.walk_forward(fr)
    mm, tm = build_outcome_models(load_games())
    wf = nc.win_probs(wf, mm, tm)
    held = wf[wf["season"] >= split]
    rec = nc.records(held, cuts=(0.52,))
    tot = rec[rec["market"] == "total"].iloc[0]
    r = held["t_res"].to_numpy(float)
    p = held["pred_t"].to_numpy(float)
    skill = float(np.mean(r ** 2) - np.mean((r - p) ** 2))

    ws = nc.fit_weights(fr, nc.SPREAD_FEATS, "s_res")
    wt = nc.fit_weights(fr, nc.TOTAL_FEATS, "t_res")
    old = pd.Series(current["total"]).reindex(nc.TOTAL_FEATS).fillna(0.0)
    X = held[nc.TOTAL_FEATS].to_numpy(float)
    shift = float(np.mean(np.abs(X @ (wt.values - old.values))))
    finite = bool(np.isfinite(wt.values).all() and np.isfinite(ws.values).all())

    why = []
    if not finite:
        why.append("the new weights aren't all numbers")
    if tot["win%"] < BREAK_EVEN:
        why.append(f"held-out totals at 52%+ went {tot['win%']:.1%}, "
                   "under break-even")
    if shift > MAX_SHIFT:
        why.append(f"the new weights would move a typical total {shift:.2f} "
                   f"pts (limit {MAX_SHIFT}), too much for one week of games")
    e = fr["model_edge"].to_numpy(float)
    b, a0 = np.polyfit(fr["epa_sum"], fr["points"], 1)
    cand = {"spread": ws.round(5).to_dict(), "total": wt.round(5).to_dict(),
            "k_margin": float(e @ (fr["margin"] - fr["hfa"]).to_numpy(float)
                              / (e @ e)),
            "total_a": float(a0), "total_b": float(b),
            "through_season": int(fr["season"].max()),
            "n_games": int(len(fr)),
            "fit_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    return {"ok": not why, "why": why, "candidate": cand,
            "held_bets": int(tot["bets"]), "held_win": float(tot["win%"]),
            "skill": round(skill, 3), "shift": round(shift, 3),
            "new_games": int(len(fr)) - int(current.get("n_games", 0)),
            "changes": {k: (round(float(old[k]), 3), round(float(wt[k]), 3))
                        for k in nc.TOTAL_FEATS
                        if abs(float(wt[k]) - float(old[k])) >= 0.005}}


def refit_text(c: dict) -> list[str]:
    head = (f"**Totals model refit**: {c['new_games']} new graded games. "
            f"Check on 2022-now games the model never trained on: "
            f"{c['held_bets']} bets at 52%+ went {c['held_win']:.1%} "
            f"(break-even {BREAK_EVEN:.1%}). "
            + ("Across every game its adjustments also land closer to the "
               "result than the closing total. " if c["skill"] > 0 else
               "Across every game, not just bets, its adjustments land no "
               "closer to the result than the closing total, so the edge "
               "is only in the confident calls. ")
            + f"New weights move a typical total "
            f"{c['shift']:.2f} pts.")
    if c["ok"]:
        out = [head, "", "New weights are in for the next picks run."]
        if c["changes"]:
            out.append("Weights that moved (pts per unit, old → new): " + ", ".join(
                f"{k} {a:+.3f} → {b:+.3f}" for k, (a, b) in c["changes"].items()))
        return out
    return [head, "", "**Kept the old weights**: " + "; ".join(c["why"]) + "."]


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------
def _row(name: str, s: dict) -> str:
    rec = f"{s['wins']}-{s['losses']}" + (f"-{s['pushes']}" if s["pushes"] else "")
    roi = f"{s['roi']:+.1%}" if s["roi"] is not None else "–"
    c = (f"{s['clv']:+.2f} pts ({s['beat_close']:.0%} beat it, n={s['clv_n']})"
         if s["clv"] is not None else "–")
    flag = " ⚠️" if flagged(s) else ""
    return f"| {name}{flag} | {rec} | {s['units']:+.2f}u | {roi} | {c} |"


def report(sections: dict[str, list[dict]], misses: list[str],
           paper: dict | None, refit: list[str] | None, owner: str = "") -> str:
    week = datetime.now(timezone.utc).strftime("%b %-d")
    out = [f"## Weekly review, {week}", "",
           "Record and units (1u = $10) by market, and closing-line value: "
           "points the close moved toward each bet after it went out. "
           "CLV is the early read on whether the model has an edge; "
           "a few dozen wins and losses are mostly luck.", "",
           "| Market | Record | Units | ROI | Avg CLV |",
           "|---|---|---|---|---|"]
    stats = {k: market_stats(v) for k, v in sections.items()}
    for k, s in stats.items():
        out.append(_row(k, s))
    if paper and paper.get("bets"):
        p = paper
        rec = f"{p['wins']}-{p['losses']} ({p['bets']} logged)"
        c = f"{p['avg_clv']:+.2f} pts" if p.get("avg_clv") is not None else "–"
        out.append(f"| NFL spread test (no money) | {rec} | – | – | {c} |")
    bad = [k for k, s in stats.items() if flagged(s)]
    if bad:
        out += ["", "**⚠️ Negative CLV over a meaningful sample**: "
                + ", ".join(f"{k} ({stats[k]['clv']:+.2f} pts over "
                            f"{stats[k]['clv_n']} bets)" for k in bad)
                + ". The closing line is moving away from these bets, "
                "which says the market disagrees with the model more often "
                "than it agrees. Worth deciding whether to keep betting "
                "them."]
    out += ["", "**Big misses**"]
    out += [f"- {m}" for m in misses] or ["- None this week."]
    if refit:
        out += [""] + refit
    out += ["", "Nothing in the betting rules changes on its own; this is "
            "for your read."]
    if owner:
        out += ["", f"@{owner}"]
    return "\n".join(out) + "\n"


def split(bets: list[dict], league: str) -> dict[str, list[dict]]:
    return {f"{league} spreads": [b for b in bets if b["market"] == "spreads"],
            f"{league} totals": [b for b in bets if b["market"] == "totals"]}


def _load(path: str, default):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default="site/history.json")
    ap.add_argument("--paper", default="site/paper.json")
    ap.add_argument("--cfb-history", default="cfb/history.json")
    ap.add_argument("--cfb-closes", default="cfb/closes.json")
    ap.add_argument("--model", default="combined_model.json")
    ap.add_argument("--refit", action="store_true",
                    help="refit the totals weights; write --model if they pass")
    ap.add_argument("--out", default="review.md")
    ap.add_argument("--owner", default="")
    a = ap.parse_args()

    from nfl_data import load_games
    from picks import ODDS_TEAM
    games = load_games()
    now = pd.Timestamp.now(tz="UTC")
    nfl = [b for b in _load(a.history, {}).get("bets", [])
           if pd.Timestamp(b["commence"]) <= now]
    # Monday night's game isn't graded in history.json until Thursday's
    # picks run; grade a copy here so the week is complete.
    import ledger
    nfl = ledger.grade(nfl, games, ODDS_TEAM)
    nfl = nfl_closes(nfl, games, ODDS_TEAM)
    cfb = [b for b in _load(a.cfb_history, {}).get("bets", [])
           if pd.Timestamp(b["commence"]) <= now]
    cfb = cfb_closes(cfb, _load(a.cfb_closes, {}))
    week_ago = now - pd.Timedelta(days=8)
    recent = lambda bs: [b for b in bs if pd.Timestamp(b["commence"]) > week_ago]
    misses = big_misses(recent(nfl), games, ODDS_TEAM) + big_misses(recent(cfb))

    refit = None
    if a.refit:
        import nfl_combined as nc
        current = _load(a.model, {})
        c = refit_check(nc.build_frame(), current)
        if c["ok"]:
            with open(a.model, "w") as fh:
                json.dump(c["candidate"], fh, indent=2)
        refit = refit_text(c)

    text = report({**split(nfl, "NFL"), **split(cfb, "College")}, misses,
                  _load(a.paper, {}).get("summary"), refit, a.owner)
    with open(a.out, "w") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
