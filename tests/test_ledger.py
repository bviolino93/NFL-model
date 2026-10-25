from datetime import datetime, timezone

import pandas as pd

import ledger

BET = {"away": "San Francisco 49ers", "home": "Atlanta Falcons",
       "commence": "2026-10-25T17:00:00Z", "market": "totals", "pick": "Over",
       "point": 50.5, "price": -105, "book": "DraftKings", "stake": 21.0,
       "win_chance": 0.57, "ev": 0.11, "why": "x"}
ABBR = {"San Francisco 49ers": "SF", "Atlanta Falcons": "ATL"}


def _games(away, home):
    return pd.DataFrame([{"home_team": "ATL", "away_team": "SF",
                          "gameday": "2026-10-25", "home_score": home,
                          "away_score": away}])


def test_merge_updates_before_kickoff_then_freezes():
    early = datetime(2026, 10, 24, tzinfo=timezone.utc)
    late = datetime(2026, 10, 25, 18, tzinfo=timezone.utc)
    h = ledger.merge([], [BET], early, "r1")
    h = ledger.merge(h, [{**BET, "price": -110}], early, "r2")
    assert len(h) == 1 and h[0]["price"] == -110 and h[0]["first_seen"] == "r1"
    h = ledger.merge(h, [{**BET, "price": +100}], late, "r3")
    assert h[0]["price"] == -110


def test_grade_totals_and_spreads():
    h = ledger.grade([dict(BET)], _games(28, 24), ABBR)
    assert h[0]["result"] == "win" and h[0]["profit"] == 20.0
    sp = {**BET, "market": "spreads", "pick": "Atlanta Falcons", "point": 3.5}
    assert ledger.grade([sp], _games(28, 24), ABBR)[0]["result"] == "loss"
    sp = {**BET, "market": "spreads", "pick": "Atlanta Falcons", "point": 4.0}
    assert ledger.grade([sp], _games(28, 24), ABBR)[0]["result"] == "push"
    s = ledger.summary(h)
    assert (s["wins"], s["losses"], s["profit"]) == (1, 0, 20.0)


def test_ungraded_until_score_posted():
    h = ledger.grade([dict(BET)], _games(None, None), ABBR)
    assert h[0].get("result") is None


def test_this_week_ends_after_monday_night():
    import picks
    rows = pd.DataFrame({"commence_time": ["2026-10-27T00:15:00Z",
                                           "2026-11-01T17:00:00Z"]})
    now = datetime(2026, 10, 25, 13, tzinfo=timezone.utc)
    assert len(picks.this_week(rows, now)) == 1
    now = datetime(2026, 10, 27, 15, tzinfo=timezone.utc)
    assert len(picks.this_week(rows, now)) == 2


def test_dropped_bet_is_pulled_and_not_counted():
    early = datetime(2026, 10, 25, 13, tzinfo=timezone.utc)
    h = ledger.merge([], [BET], early, "r1")
    h = ledger.merge(h, [], early, "r2")
    assert h[0]["pulled"]
    h = ledger.grade(h, _games(28, 24), ABBR)
    assert h[0]["result"] is None and ledger.summary(h)["wins"] == 0
    h = ledger.merge(h, [BET], early, "r3")
    assert not h[0]["pulled"]
