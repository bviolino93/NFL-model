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


def test_fix_starters_replaces_qb_on_ir(monkeypatch):
    import nfl_data
    games = pd.DataFrame([
        {"season": 2026, "week": 5, "home_team": "JAX", "away_team": "PHI",
         "home_qb_id": "jax1", "away_qb_id": "hurts", "home_score": 19,
         "away_score": 17},
        {"season": 2026, "week": 6, "home_team": "PHI", "away_team": "CAR",
         "home_qb_id": "mckee", "away_qb_id": "car1", "home_score": 14,
         "away_score": 25},
        {"season": 2026, "week": 8, "home_team": "WAS", "away_team": "PHI",
         "home_qb_id": "was1", "away_qb_id": "hurts", "home_score": None,
         "away_score": None}])
    ros = pd.DataFrame({"gsis_id": ["hurts", "mckee", "was1"],
                        "status": ["RES", "ACT", "ACT"], "week": [7, 7, 7]})
    monkeypatch.setattr(nfl_data, "load_rosters", lambda *a, **k: ros)
    out = nfl_data.fix_starters(games)
    assert out.loc[2, "away_qb_id"] == "mckee"
    assert out.loc[2, "home_qb_id"] == "was1"


def test_qb_swap_names_the_regular_starter_when_someone_else_starts():
    import pandas as pd
    import paper
    g = pd.DataFrame({
        "season": [2026] * 3, "margin": [3, -7, None],
        "home_team": ["PHI", "NYG", "PHI"], "away_team": ["DAL", "PHI", "DAL"],
        "home_qb_name": ["Jalen Hurts", "X", "Tanner McKee"],
        "away_qb_name": ["Dak Prescott", "Jalen Hurts", "Dak Prescott"]})
    s = paper.qb_swap(g, {"Jalen Hurts": "RES"}, 2026, "PHI", "Tanner McKee")
    assert s == {"name": "Jalen Hurts", "pos": "QB", "status": "IR",
                 "pts": None, "note": "McKee starting"}
    assert paper.qb_swap(g, {}, 2026, "DAL", "Dak Prescott") is None
