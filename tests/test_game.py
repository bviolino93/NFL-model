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


def _check(title="Chiefs @ Seahawks", when="2026-10-27T03:00:00+00:00"):
    return {"league": "nfl", "found": True, "query": "Chiefs", "checked": when,
            "title": title, "away": "Kansas City Chiefs", "home": "Seattle Seahawks",
            "kickoff": "2026-11-01T21:25:00Z",
            "lines": [{"market": "Spread", "sharp": "Seahawks by 2.5",
                       "model": "Seahawks by 2.9"}],
            "offers": [{"market": "spreads", "side": "Seahawks -2.5", "price": -110,
                        "book": "DraftKings", "win": 0.53, "ev": 0.012, "bet": False}],
            "verdicts": [{"market": "Spread", "bet": False,
                          "head": "Lean Seahawks -2.5, but under the bar",
                          "text": "The best side is Seahawks -2.5 at +1.2% EV."}]}


def test_check_markdown():
    md = game.to_md(_check())
    assert md.startswith("# Chiefs @ Seahawks")
    assert "| Spread | Seahawks by 2.5 | Seahawks by 2.9 |" in md
    assert "| Seahawks -2.5 | -110 | DraftKings | 53.0% | +1.2% |" in md
    assert "**Spread: Lean Seahawks -2.5, but under the bar.**" in md
    miss = game.to_md({"found": False, "title": "No game found",
                       "checked": "2026-10-27T03:00:00+00:00", "message": "Nope."})
    assert "Nope." in miss


def test_checks_keep_newest_one_per_game(tmp_path):
    p = tmp_path / "checks.json"
    game.save_check(str(p), _check(when="2026-10-27T01:00:00+00:00"))
    game.save_check(str(p), _check("Bills @ Jets", "2026-10-27T02:00:00+00:00"))
    game.save_check(str(p), _check(when="2026-10-27T03:00:00+00:00"))
    import json
    c = json.loads(p.read_text())["checks"]
    assert [x["title"] for x in c] == ["Chiefs @ Seahawks", "Bills @ Jets"]
    assert c[0]["checked"].startswith("2026-10-27T03")
    for i in range(20):
        game.save_check(str(p), _check(f"G{i}"))
    assert len(json.loads(p.read_text())["checks"]) == game.CHECKS_KEPT
