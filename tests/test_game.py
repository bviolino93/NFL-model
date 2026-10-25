import pandas as pd

import game


def _rows():
    g = [("e1", "Seattle Seahawks", "Kansas City Chiefs", "2026-10-26T00:20Z"),
         ("e2", "Philadelphia Eagles", "Dallas Cowboys", "2026-10-27T00:15Z"),
         ("e3", "Arizona Cardinals", "Kansas City Chiefs", "2026-11-01T18:00Z")]
    return pd.DataFrame([{"event_id": e, "home_team": h, "away_team": a,
                          "commence_time": t} for e, h, a, t in g])


def test_finds_game_by_nickname_abbreviation_or_matchup():
    r = _rows()
    assert game.find_event(r, "Seahawks") == "e1"
    assert game.find_event(r, "kc sea") == "e1"
    assert game.find_event(r, "Cowboys at Eagles") == "e2"
    assert game.find_event(r, "DAL @ PHI") == "e2"


def test_picks_the_soonest_game_and_requires_every_word_to_match():
    r = _rows()
    assert game.find_event(r, "Chiefs") == "e1"
    assert game.find_event(r, "Chiefs Cardinals") == "e3"
    assert game.find_event(r, "Chiefs Eagles") is None
    assert game.find_event(r, "") is None
