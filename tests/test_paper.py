import pandas as pd

import paper
from selector import Fair

ABBR = {"Green Bay Packers": "GB", "Carolina Panthers": "CAR"}


def _fair(margin):
    f = Fair("Green Bay Packers", "Carolina Panthers", "2026-11-01T17:00:00Z")
    f.market_margin = margin
    return {"e1": f}


def test_logs_once_when_gap_is_big_and_grades_clv():
    mine = {"e1": {"ratings": 7.0, "injuries": -0.5, "margin": 6.5}}
    h = paper.log([], _fair(3.0), mine, "2026-10-26T20:00:00+00:00")
    assert len(h) == 1 and h[0]["side"] == "Green Bay Packers"
    assert h[0]["point"] == -3.0
    h = paper.log(h, _fair(1.0), mine, "2026-10-27T20:00:00+00:00")        # already seen: no repeat
    assert len(h) == 1
    games = pd.DataFrame([{"home_team": "GB", "away_team": "CAR",
                           "gameday": "2026-11-01", "spread_line": 4.5,
                           "home_score": 27, "away_score": 20}])
    h = paper.grade(h, games, ABBR)
    assert h[0]["clv"] == 1.5 and h[0]["result"] == "win"
    assert paper.summary(h)["avg_clv"] == 1.5


def test_small_gap_is_skipped():
    mine = {"e1": {"ratings": 4.0, "injuries": 0.0, "margin": 4.0}}
    assert paper.log([], _fair(3.0), mine, "2026-10-26T20:00:00+00:00") == []


def test_late_lines_are_not_early():
    mine = {"e1": {"ratings": 9.0, "injuries": 0.0, "margin": 9.0}}
    assert paper.log([], _fair(3.0), mine, "2026-10-31T20:00:00+00:00") == []
