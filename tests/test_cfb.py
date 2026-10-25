import pandas as pd

import cfb_picks
from selector import Fair


def _sched():
    return pd.DataFrame([
        {"game_id": 1, "home_team": "Miami", "away_team": "Florida State",
         "start_date": "2026-10-31T19:30:00.000Z"},
        {"game_id": 2, "home_team": "Miami (OH)", "away_team": "Ohio",
         "start_date": "2026-10-31T19:30:00.000Z"},
        {"game_id": 3, "home_team": "San José State", "away_team": "Hawai'i",
         "start_date": "2026-11-01T03:00:00.000Z"},
        {"game_id": 4, "home_team": "App State", "away_team": "UL Monroe",
         "start_date": "2026-10-31T17:00:00.000Z"},
    ])


def test_odds_names_match_schedule_schools():
    fair = {
        "a": Fair("Miami Hurricanes", "Florida State Seminoles", "2026-10-31T19:30:00Z"),
        "b": Fair("Miami (OH) RedHawks", "Ohio Bobcats", "2026-10-31T19:30:00Z"),
        "c": Fair("San Jose State Spartans", "Hawaii Rainbow Warriors", "2026-11-01T03:00:00Z"),
        "d": Fair("Appalachian State Mountaineers", "Louisiana Monroe Warhawks",
                  "2026-10-31T17:00:00Z"),
        "e": Fair("Nowhere Tech Owls", "Elsewhere A&M Ducks", "2026-10-31T17:00:00Z"),
    }
    assert cfb_picks.match_games(fair, _sched()) == {"a": 1, "b": 2, "c": 3, "d": 4}


def test_week_runs_to_sunday_after_saturday():
    end = cfb_picks.week_end(pd.Timestamp("2026-10-25T22:00Z"))      # Sunday evening
    assert end == pd.Timestamp("2026-11-01T12:00Z")
    assert cfb_picks.week_end(pd.Timestamp("2026-10-31T15:00Z")) == end   # Saturday


def test_grade_by_game_id():
    g = pd.DataFrame([{"game_id": 7, "margin": 10.0, "home_points": 31.0,
                       "away_points": 21.0}])
    bets = [{"game_id": 7, "home": "H", "away": "A", "pick": "H", "point": -7.5,
             "market": "spreads", "price": -110, "stake": 11.0, "result": None},
            {"game_id": 7, "home": "H", "away": "A", "pick": "Over", "point": 55.5,
             "market": "totals", "price": -110, "stake": 11.0, "result": None}]
    out = cfb_picks.grade(bets, g)
    assert [b["result"] for b in out] == ["win", "loss"]
    assert out[0]["profit"] == 10.0 and out[0]["final"] == "21-31"
