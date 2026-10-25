"""
Bet selector: start from the sharp market, add only what has earned a vote,
and size what is left.

The pipeline
------------
1. FAIR LINE FROM THE MARKET. Every book's two-way price is de-vigged, then
   turned into an implied fair home margin (or fair total). Those are
   combined as a weighted median, with sharp books weighted up. This is the
   best single estimate available, and it is the baseline everything else
   is measured against.

2. ADJUSTMENTS. Each one is a tagged nudge to the fair margin/total: wind,
   injury news after the line was set, a model edge, or a human override
   ("-1.5, starting LT out"). Each has a SOURCE, and a source is either
   ACTIVE (it moves the pick) or INFO (it is shown and logged but does not).
   A source becomes active only once it has shown positive CLV or beaten the
   close in a backtest. Human overrides are active by default: they are the
   point of having a human, and the CLV ledger says whether to keep them.

3. PRICE EVERY OFFER. Each book's actual number and price is priced against
   the adjusted fair line with EMPIRICAL margin distributions (games with a
   similar closing line), so pushes on 3, 7, 10 and the like are counted
   rather than smoothed over by a normal curve. That alone flags stale
   numbers at soft books, the most reliable edge a non-syndicate bettor has.

4. SIZE. Fractional Kelly on the EV, capped per bet and per week.

5. GRADE. clv() prices each bet at the CLOSING fair line, so every source
   accumulates a record you can judge in a few hundred bets rather than a
   few thousand.

Inputs are rows in log_lines.py's format, so the same code runs on a live
pull and on logged history.

    python selector.py data/lines/games_2026.csv [--overrides mine.csv]
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Weight in the consensus. Offshore and market-making books move first and
# take sharp action; the big US retail books mostly copy. Unknown books get 1.
BOOK_WEIGHT = {
    "pinnacle": 4.0, "circasports": 3.0, "betonlineag": 2.0, "lowvig": 2.0,
    "bookmaker": 2.0, "betcris": 2.0, "matchbook": 2.0, "betfair_ex_us": 2.0,
    "novig": 2.0, "prophetx": 1.5,
}

# Sources that move picks. Everything else is INFO until it proves itself.
ACTIVE_SOURCES = {"override", "wind"}


# ----------------------------------------------------------------------
# Odds arithmetic
# ----------------------------------------------------------------------
def american_to_prob(o: float) -> float:
    o = float(o)
    return 100.0 / (o + 100.0) if o > 0 else -o / (-o + 100.0)


def payout(o: float) -> float:
    """Profit per unit staked on a win."""
    o = float(o)
    return 100.0 / -o if o < 0 else o / 100.0


def devig(p_a: float, p_b: float) -> tuple[float, float]:
    """Multiplicative de-vig of a two-way market."""
    s = p_a + p_b
    return p_a / s, p_b / s


# ----------------------------------------------------------------------
# Empirical result distributions
# ----------------------------------------------------------------------
class OutcomeModel:
    """
    P(result > line), P(result == line) for a fair centre `mu`, from
    historical games whose closing number was near `mu`. The results are NOT
    shifted, so the integer structure (3s and 7s for margins) survives.
    The window widens until it holds `min_games`.
    """

    def __init__(self, centres: np.ndarray, results: np.ndarray,
                 min_games: int = 300):
        o = np.argsort(centres)
        self.c = np.asarray(centres, float)[o]
        self.r = np.asarray(results, float)[o]
        self.min_games = min_games
        self._cache: dict = {}

    def _sample(self, mu: float) -> np.ndarray:
        key = round(mu * 4) / 4
        if key in self._cache:
            return self._cache[key]
        w = 0.5
        while True:
            lo = np.searchsorted(self.c, key - w)
            hi = np.searchsorted(self.c, key + w, side="right")
            if hi - lo >= self.min_games or w > 20:
                break
            w += 0.5
        s = self.r[lo:hi]
        # Re-centre only the part the window's own spread introduces, by an
        # integer, so key numbers stay where they belong.
        shift = round(key - float(self.c[lo:hi].mean()))
        s = s + shift
        self._cache[key] = s
        return s

    def probs(self, mu: float, line: float) -> tuple[float, float]:
        s = self._sample(mu)
        return float((s > line).mean()), float((s == line).mean())

    def fair_centre(self, line: float, p_over_fair: float) -> float:
        """Invert: the mu at which P(over line | no push) = p_over_fair."""
        grid = np.arange(line - 20, line + 20.01, 0.25)
        best, err = line, 9.0
        for mu in grid:
            pw, pp = self.probs(mu, line)
            q = pw / max(1e-9, 1 - pp)
            if abs(q - p_over_fair) < err:
                best, err = mu, abs(q - p_over_fair)
        return float(best)


def build_outcome_models(games: pd.DataFrame, since: int = 2010):
    """Margin model keyed on closing spread, total model on closing total."""
    g = games[(games["season"] >= since)].dropna(
        subset=["spread_line", "total_line", "home_score", "away_score"])
    margin = (g["home_score"] - g["away_score"]).to_numpy(float)
    points = (g["home_score"] + g["away_score"]).to_numpy(float)
    return (OutcomeModel(g["spread_line"].to_numpy(float), margin),
            OutcomeModel(g["total_line"].to_numpy(float), points))


# ----------------------------------------------------------------------
# Adjustments
# ----------------------------------------------------------------------
@dataclass
class Adjustment:
    """
    A nudge to the fair line for one game. `points` is added to the fair
    HOME MARGIN for spreads/moneylines, or to the fair TOTAL for totals.
    `home_team` matches the feed's home team name.
    """
    home_team: str
    market: str          # "spreads" or "totals"
    points: float
    source: str          # "override", "wind", "injury", "model", ...
    note: str = ""

    @property
    def active(self) -> bool:
        return self.source in ACTIVE_SOURCES


@dataclass
class Fair:
    home_team: str
    away_team: str
    commence: str
    market_margin: float = float("nan")
    market_total: float = float("nan")
    margin: float = float("nan")         # after active adjustments
    total: float = float("nan")
    adjustments: list = field(default_factory=list)


# ----------------------------------------------------------------------
# Consensus
# ----------------------------------------------------------------------
def _wmedian(values, weights):
    v = np.asarray(values, float)
    w = np.asarray(weights, float)
    o = np.argsort(v)
    v, w = v[o], w[o]
    c = np.cumsum(w)
    return float(v[np.searchsorted(c, c[-1] / 2.0)])


def _pairs(snap: pd.DataFrame, market: str):
    """Yield (book, side_a_row, side_b_row) for two-way markets."""
    m = snap[snap["market"] == market]
    for (ev, book, pt_key), grp in m.groupby(
            ["event_id", "book", m["point"].abs() if market == "spreads"
             else m["point"]]):
        if len(grp) != 2:
            continue
        yield ev, book, grp


def consensus(snap: pd.DataFrame, margin_model: OutcomeModel,
              total_model: OutcomeModel) -> dict:
    """Fair home margin and total per event from every book's prices."""
    out = {}
    meta = snap.drop_duplicates("event_id").set_index("event_id")
    est = {}
    for ev, book, grp in _pairs(snap, "spreads"):
        home = meta.loc[ev, "home_team"]
        h = grp[grp["outcome"] == home]
        a = grp[grp["outcome"] != home]
        if h.empty or a.empty:
            continue
        p_h, _ = devig(american_to_prob(h["price"].iloc[0]),
                       american_to_prob(a["price"].iloc[0]))
        # Home -3 covers when margin > 3; stated home point is -3.
        line = -float(h["point"].iloc[0])
        mu = margin_model.fair_centre(line, p_h)
        est.setdefault((ev, "m"), []).append((mu, BOOK_WEIGHT.get(book, 1.0)))
    for ev, book, grp in _pairs(snap, "totals"):
        o = grp[grp["outcome"].str.lower() == "over"]
        u = grp[grp["outcome"].str.lower() == "under"]
        if o.empty or u.empty:
            continue
        p_o, _ = devig(american_to_prob(o["price"].iloc[0]),
                       american_to_prob(u["price"].iloc[0]))
        mu = total_model.fair_centre(float(o["point"].iloc[0]), p_o)
        est.setdefault((ev, "t"), []).append((mu, BOOK_WEIGHT.get(book, 1.0)))
    for ev, r in meta.iterrows():
        f = Fair(r["home_team"], r["away_team"], r["commence_time"])
        if (ev, "m") in est:
            v, w = zip(*est[(ev, "m")])
            f.market_margin = f.margin = _wmedian(v, w)
        if (ev, "t") in est:
            v, w = zip(*est[(ev, "t")])
            f.market_total = f.total = _wmedian(v, w)
        out[ev] = f
    return out


def apply_adjustments(fair: dict, adjustments) -> dict:
    by_home = {f.home_team: f for f in fair.values()}
    for a in adjustments or []:
        f = by_home.get(a.home_team)
        if f is None:
            continue
        f.adjustments.append(a)
        if not a.active:
            continue
        if a.market == "spreads":
            f.margin += a.points
        elif a.market == "totals":
            f.total += a.points
    return fair


# ----------------------------------------------------------------------
# Pricing and sizing
# ----------------------------------------------------------------------
def kelly(p_win: float, p_push: float, price: float) -> float:
    """Full-Kelly fraction with pushes (stake returned) accounted for."""
    b = payout(price)
    q = 1.0 - p_win - p_push
    f = (b * p_win - q) / b
    return max(0.0, f)


@dataclass
class Policy:
    min_ev: float = 0.02          # 2% of stake
    kelly_mult: float = 0.25      # quarter Kelly
    max_bet: float = 0.02         # of bankroll
    max_week: float = 0.10
    min_books: int = 3            # need a real consensus to price against


def price_offers(snap: pd.DataFrame, fair: dict, margin_model, total_model,
                 policy: Policy | None = None) -> pd.DataFrame:
    """Every offer priced against the adjusted fair line."""
    policy = policy or Policy()
    rows = []
    nbooks = snap.groupby("event_id")["book"].nunique()
    for r in snap.itertuples():
        f = fair.get(r.event_id)
        if f is None or nbooks.get(r.event_id, 0) < policy.min_books:
            continue
        if r.market == "spreads" and math.isfinite(f.margin):
            # Stated from the home margin: home -3 covers above 3, away +3
            # covers below 3.
            home = r.outcome == f.home_team
            line = -float(r.point) if home else float(r.point)
            pw, pp = margin_model.probs(f.margin, line)
            if not home:
                pw = 1.0 - pw - pp
            src_fair = f.margin
        elif r.market == "totals" and math.isfinite(f.total):
            pw, pp = total_model.probs(f.total, float(r.point))
            if str(r.outcome).lower() == "under":
                pw = 1.0 - pw - pp
            src_fair = f.total
        elif r.market == "h2h" and math.isfinite(f.margin):
            p_home, p_tie = margin_model.probs(f.margin, 0.0)
            # A tie is rare; split it, as most books grade it as a push.
            pw = p_home if r.outcome == f.home_team else 1 - p_home - p_tie
            pp = p_tie
            src_fair = f.margin
        else:
            continue
        b = payout(r.price)
        ev = pw * b - (1.0 - pw - pp)
        moved = [a for a in f.adjustments if a.active and
                 (a.market == r.market or
                  (r.market == "h2h" and a.market == "spreads"))]
        rows.append({
            "event_id": r.event_id, "home": f.home_team, "away": f.away_team,
            "commence": f.commence, "market": r.market, "pick": r.outcome,
            "point": r.point, "price": r.price, "book": r.book,
            "fair": src_fair, "p_win": pw, "p_push": pp, "ev": ev,
            "kelly": kelly(pw, pp, r.price),
            "sources": ",".join(sorted({a.source for a in moved})) or "market",
            "notes": "; ".join(a.note for a in f.adjustments if a.note
                               and a.market in (r.market, "spreads")),
        })
    return pd.DataFrame(rows)


def select(priced: pd.DataFrame, bankroll: float = 1.0,
           policy: Policy | None = None) -> pd.DataFrame:
    """Best offer per side, filtered by EV, sized, capped."""
    policy = policy or Policy()
    if priced.empty:
        return priced
    best = (priced.sort_values("ev", ascending=False)
                  .drop_duplicates(["event_id", "market", "pick"]))
    # One side per market per game: the better of the two.
    best = best.drop_duplicates(["event_id", "market"])
    pick = best[best["ev"] >= policy.min_ev].copy()
    pick["stake"] = np.minimum(pick["kelly"] * policy.kelly_mult,
                               policy.max_bet)
    tot = pick["stake"].sum()
    if tot > policy.max_week:
        pick["stake"] *= policy.max_week / tot
    pick["stake"] = (pick["stake"] * bankroll).round(2)
    return pick.sort_values("ev", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------
# Grading
# ----------------------------------------------------------------------
def clv(bet: dict, close_fair: Fair, margin_model, total_model) -> float:
    """
    EV of the bet as placed, priced at the CLOSING fair line. Positive CLV
    across a few hundred bets is the earliest trustworthy sign of an edge.
    """
    m, pick, price = bet["market"], bet["pick"], bet["price"]
    if m == "spreads":
        home = pick == close_fair.home_team
        line = -float(bet["point"]) if home else float(bet["point"])
        pw, pp = margin_model.probs(close_fair.market_margin, line)
        if not home:
            pw = 1 - pw - pp
    elif m == "totals":
        pw, pp = total_model.probs(close_fair.market_total,
                                   float(bet["point"]))
        if str(pick).lower() == "under":
            pw = 1 - pw - pp
    else:
        p_home, pp = margin_model.probs(close_fair.market_margin, 0.0)
        pw = p_home if pick == close_fair.home_team else 1 - p_home - pp
    return pw * payout(price) - (1 - pw - pp)


def load_overrides(path: str) -> list:
    """
    Your judgment, one row per call:

        home_team,market,points,source,note
        Baltimore Ravens,spreads,-1.5,override,Stanley out (beat reporter)
        Buffalo Bills,totals,-2,override,snow squall forecast

    `points` moves the fair HOME margin (spreads) or the fair total.
    """
    d = pd.read_csv(path)
    if "source" not in d:
        d["source"] = "override"
    return [Adjustment(r.home_team, r.market, float(r.points),
                       str(r.source), str(getattr(r, "note", "") or ""))
            for r in d.itertuples()]


def latest_snapshot(lines: pd.DataFrame) -> pd.DataFrame:
    last = lines["snapshot_utc"].max()
    return lines[lines["snapshot_utc"] == last]


def main():
    from nfl_data import load_games
    ap = argparse.ArgumentParser()
    ap.add_argument("lines_csv")
    ap.add_argument("--bankroll", type=float, default=1000.0)
    ap.add_argument("--min-ev", type=float, default=Policy.min_ev)
    ap.add_argument("--overrides", default="",
                    help="CSV of your adjustments (see load_overrides)")
    a = ap.parse_args()
    lines = pd.read_csv(a.lines_csv)
    snap = latest_snapshot(lines)
    mm, tm = build_outcome_models(load_games())
    fair = consensus(snap, mm, tm)
    if a.overrides:
        apply_adjustments(fair, load_overrides(a.overrides))
    pol = Policy(min_ev=a.min_ev)
    picks = select(price_offers(snap, fair, mm, tm, pol), a.bankroll, pol)
    pd.set_option("display.width", 200)
    cols = ["away", "home", "market", "pick", "point", "price", "book",
            "fair", "p_win", "ev", "stake", "sources"]
    print(picks[cols].round(3).to_string(index=False) if len(picks)
          else "no bets clear the threshold")


if __name__ == "__main__":
    main()
