"""Sanity checks for selector.py on synthetic snapshots (no network)."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import selector as S  # noqa: E402


def _models():
    # Synthetic history: margins ~ N(spread, 13) rounded, with extra mass
    # on 3, like the real thing; totals ~ N(total, 13).
    rng = np.random.default_rng(0)
    sp = rng.choice(np.arange(-14, 14.5, 0.5), 20000)
    m = np.round(sp + rng.normal(0, 13, sp.size))
    m[rng.random(sp.size) < 0.06] = 3
    tl = rng.choice(np.arange(37, 56, 0.5), 20000)
    pts = np.round(tl + rng.normal(0, 13, tl.size))
    return S.OutcomeModel(sp, m), S.OutcomeModel(tl, pts)


def _snap(rows):
    base = {"snapshot_utc": "t", "event_id": "e1", "commence_time": "c",
            "home_team": "Home", "away_team": "Away", "player": "",
            "last_update": "u"}
    return pd.DataFrame([{**base, **r} for r in rows])


def _game(book, home_pt, hp=-110, ap=-110, tot=45.5, op=-110, up=-110):
    return [
        {"book": book, "market": "spreads", "outcome": "Home",
         "point": home_pt, "price": hp},
        {"book": book, "market": "spreads", "outcome": "Away",
         "point": -home_pt, "price": ap},
        {"book": book, "market": "totals", "outcome": "Over",
         "point": tot, "price": op},
        {"book": book, "market": "totals", "outcome": "Under",
         "point": tot, "price": up},
    ]


def test_devig_and_kelly():
    a, b = S.devig(S.american_to_prob(-110), S.american_to_prob(-110))
    assert abs(a - 0.5) < 1e-12 and abs(b - 0.5) < 1e-12
    assert S.kelly(0.5, 0.0, -110) == 0.0
    assert S.kelly(0.55, 0.0, 100) > 0


def test_consensus_and_stale_line():
    mm, tm = _models()
    rows = []
    for bk in ("pinnacle", "b1", "b2", "b3"):
        rows += _game(bk, -3.0)
    rows += _game("soft", -1.5)        # stale: home only -1.5
    snap = _snap(rows)
    fair = S.consensus(snap, mm, tm)
    f = fair["e1"]
    assert 2.0 < f.margin < 4.0, f.margin
    assert 44.0 < f.total < 47.0, f.total
    priced = S.price_offers(snap, fair, mm, tm)
    picks = S.select(priced, bankroll=1000)
    sp = picks[picks["market"] == "spreads"]
    assert len(sp) == 1
    assert sp.iloc[0]["book"] == "soft" and sp.iloc[0]["pick"] == "Home"
    # Fair -3 games at -110 everywhere: nothing else should clear 2% EV.
    assert (picks["market"] == "totals").sum() == 0


def test_away_side_priced_from_home_margin():
    mm, tm = _models()
    rows = []
    for bk in ("pinnacle", "b1", "b2", "b3"):
        rows += _game(bk, -3.0)
    rows += _game("soft", -4.5)        # away +4.5 is the stale side
    snap = _snap(rows)
    fair = S.consensus(snap, mm, tm)
    picks = S.select(S.price_offers(snap, fair, mm, tm))
    sp = picks[picks["market"] == "spreads"]
    assert len(sp) == 1 and sp.iloc[0]["pick"] == "Away"


def test_override_moves_pick_and_info_does_not():
    mm, tm = _models()
    rows = []
    for bk in ("pinnacle", "b1", "b2", "b3"):
        rows += _game(bk, -3.0)
    snap = _snap(rows)
    fair = S.consensus(snap, mm, tm)
    S.apply_adjustments(fair, [
        S.Adjustment("Home", "totals", -10.0, "model", "info only"),
        S.Adjustment("Home", "spreads", -4.0, "override", "LT out"),
    ])
    picks = S.select(S.price_offers(snap, fair, mm, tm))
    assert set(picks["market"]) == {"spreads"}
    assert picks.iloc[0]["pick"] == "Away"
    assert picks.iloc[0]["sources"] == "override"


def test_clv_sign():
    mm, tm = _models()
    rows = []
    for bk in ("pinnacle", "b1", "b2", "b3"):
        rows += _game(bk, -3.0)
    close = S.consensus(_snap(rows), mm, tm)["e1"]
    good = {"market": "spreads", "pick": "Home", "point": -1.5, "price": -110}
    bad = {"market": "spreads", "pick": "Home", "point": -4.5, "price": -110}
    assert S.clv(good, close, mm, tm) > 0 > S.clv(bad, close, mm, tm)
