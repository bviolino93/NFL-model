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
    now = datetime(2026, 10, 26, 15, tzinfo=timezone.utc)
    assert len(picks.this_week(rows, now)) == 1
    now = datetime(2026, 10, 27, 15, tzinfo=timezone.utc)
    assert len(picks.this_week(rows, now)) == 1   # Monday game under way


def test_games_under_way_are_not_priced():
    import picks
    rows = pd.DataFrame({"commence_time": ["2026-10-25T17:02:00Z",
                                           "2026-10-25T20:05:00Z"]})
    now = datetime(2026, 10, 25, 18, 13, tzinfo=timezone.utc)
    assert list(picks.this_week(rows, now).commence_time) == \
        ["2026-10-25T20:05:00Z"]


def test_bets_first_seen_after_kickoff_are_dropped():
    late = datetime(2026, 10, 25, 18, 13, tzinfo=timezone.utc)
    live = {**BET, "commence": "2026-10-25T17:02:00Z",
            "first_seen": late.isoformat(), "last_seen": late.isoformat()}
    assert ledger.merge([live], [], late, late.isoformat()) == []


def test_kickoff_time_shift_is_the_same_bet():
    early = datetime(2026, 10, 25, 13, tzinfo=timezone.utc)
    h = ledger.merge([], [BET], early, early.isoformat())
    moved = {**BET, "commence": "2026-10-25T17:03:00Z"}
    assert ledger._key(moved) == ledger._key(h[0])


def test_dropped_bet_is_removed_before_kickoff_kept_after():
    early = datetime(2026, 10, 25, 13, tzinfo=timezone.utc)
    late = datetime(2026, 10, 25, 18, tzinfo=timezone.utc)
    h = ledger.merge([], [BET], early, "r1")
    assert ledger.merge(h, [], early, "r2") == []
    assert len(ledger.merge(h, [], late, "r2")) == 1


def test_merge_keeps_the_alerted_line():
    early = datetime(2026, 10, 24, tzinfo=timezone.utc)
    h = ledger.merge([], [BET], early, "r1")
    h = ledger.merge(h, [{**BET, "point": 51.5, "price": -110}], early, "r2")
    assert (h[0]["point"], h[0]["first_point"], h[0]["first_price"]) == \
        (51.5, 50.5, -105)


def test_grade_moneyline():
    b = {"market": "h2h", "pick": "A", "home": "H", "away": "A", "point": None}
    assert ledger.grade_one(b, 20, 24) == "win"
    assert ledger.grade_one(b, 24, 20) == "loss"
    assert ledger.grade_one({**b, "point": float("nan")}, 17, 17) == "push"
