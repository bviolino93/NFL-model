"""
Bet history for the picks dashboard: every bet the scheduled runs suggested,
graded against final scores.

Each run's picks are merged into the previous history.json (read back from
the `picks` branch). A bet is keyed by game, market and side; a later run
before kickoff replaces it (the line or stake may have moved), and once the
game starts the last version stands. Final scores come from nflverse
games.csv, so a bet is graded the first run after the score is posted.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

KEY = ("away", "home", "commence", "market", "pick")


def _key(b: dict) -> str:
    # Only the kickoff date: books shift commence times to the actual
    # kickoff (1:00 -> 1:03 PM), which must not make a bet look new.
    return "|".join(str(b[k])[:10] if k == "commence" else str(b[k])
                    for k in KEY)


def load(path: str) -> list[dict]:
    try:
        with open(path) as fh:
            return json.load(fh).get("bets", [])
    except (OSError, ValueError):
        return []


def merge(history: list[dict], picks: list[dict], now: datetime,
          run: str) -> list[dict]:
    """A bet a later run no longer suggests before kickoff (the line moved
    or a book caught up) is dropped: only bets still on the board at the
    last run before the game are kept and graded. Old files may hold
    bets marked pulled; those are dropped too."""
    current = {_key(p) for p in picks}
    # Bets first seen after kickoff came from live in-game lines (a late
    # run before the in-play filter); they were never real pre-game bets.
    def pregame(b):
        seen = pd.to_datetime(b.get("first_seen"), utc=True, errors="coerce")
        return pd.isna(seen) or seen < pd.Timestamp(b["commence"])

    out = {_key(b): b for b in history
           if not b.get("pulled") and pregame(b)
           and (_key(b) in current
                or pd.Timestamp(b["commence"]) <= pd.Timestamp(now))}
    for p in picks:
        k = _key(p)
        old = out.get(k)
        if old and pd.Timestamp(old["commence"]) <= pd.Timestamp(now):
            continue
        # The line at first sight is the one the alert went out with, so
        # it is the one closing-line value is measured from.
        first = old or p
        out[k] = {**p, "first_seen": (old or {}).get("first_seen", run),
                  "first_point": first.get("first_point", first["point"]),
                  "first_price": first.get("first_price", first["price"]),
                  "last_seen": run, "result": None, "profit": None,
                  "pulled": False}
    return sorted(out.values(), key=lambda b: (b["commence"], b["home"]))


def payout(stake: float, price: float) -> float:
    """Profit on a winning American-odds bet."""
    return stake * (price / 100 if price > 0 else 100 / -price)


def grade_one(b: dict, home_score: float, away_score: float) -> str:
    if b["market"] == "totals":
        diff = home_score + away_score - b["point"]
        if b["pick"] == "Under":
            diff = -diff
    else:
        mine, theirs = ((home_score, away_score) if b["pick"] == b["home"]
                        else (away_score, home_score))
        diff = mine - theirs + (0 if b["market"] == "h2h" else b["point"] or 0)
    return "win" if diff > 0 else "loss" if diff < 0 else "push"


def grade(bets: list[dict], games: pd.DataFrame,
          abbr: dict[str, str]) -> list[dict]:
    done = games.dropna(subset=["home_score", "away_score"])
    done = done.assign(day=pd.to_datetime(done["gameday"]).dt.normalize())
    for b in bets:
        if b.get("result") or b.get("pulled"):
            continue
        day = (pd.Timestamp(b["commence"]).tz_convert("America/New_York")
               .tz_localize(None).normalize())
        g = done[(done.home_team == abbr.get(b["home"]))
                 & (done.away_team == abbr.get(b["away"]))
                 & ((done.day - day).abs() <= pd.Timedelta(days=1))]
        if g.empty:
            continue
        g = g.iloc[0]
        b["result"] = grade_one(b, g.home_score, g.away_score)
        b["final"] = f"{int(g.away_score)}-{int(g.home_score)}"
        b["profit"] = round({"win": payout(b["stake"], b["price"]),
                             "loss": -b["stake"], "push": 0.0}[b["result"]], 2)
    return bets


def summary(bets: list[dict]) -> dict:
    graded = [b for b in bets if b.get("result") and not b.get("pulled")]
    staked = sum(b["stake"] for b in graded)
    profit = sum(b["profit"] for b in graded)
    return {
        "wins": sum(b["result"] == "win" for b in graded),
        "losses": sum(b["result"] == "loss" for b in graded),
        "pushes": sum(b["result"] == "push" for b in graded),
        "staked": round(staked, 2), "profit": round(profit, 2),
        "roi": round(profit / staked, 4) if staked else None,
    }


def save(path: str, bets: list[dict]) -> None:
    with open(path, "w") as fh:
        json.dump({"updated": datetime.now(timezone.utc).isoformat(),
                   "summary": summary(bets), "bets": bets}, fh, indent=2,
                  default=str)
