import pandas as pd

import review

BET = {"away": "Cincinnati Bengals", "home": "Baltimore Ravens",
       "commence": "2026-10-25T17:00:00Z", "market": "spreads",
       "pick": "Baltimore Ravens", "point": 2.5, "price": -110, "stake": 10.0,
       "why": "Burrow at quarterback", "injuries": []}
ABBR = {"Cincinnati Bengals": "CIN", "Baltimore Ravens": "BAL"}


def test_clv_spreads_and_totals():
    # took Ravens +2.5, closed +3 (home spread +3): half a point worse
    assert review.clv(BET, 3.0, None) == -0.5
    assert review.clv({**BET, "pick": "Cincinnati Bengals", "point": -2.5},
                      3.0, None) == 0.5
    # alerted line wins over a later version of the bet
    assert review.clv({**BET, "first_point": 3.5}, 3.0, None) == 0.5
    over = {**BET, "market": "totals", "pick": "Over", "point": 44.0}
    assert review.clv(over, None, 45.0) == 1.0
    assert review.clv({**over, "pick": "Under"}, None, 45.0) == -1.0
    assert review.clv(over, None, None) is None


def test_nfl_close_from_nflverse_spread_line():
    g = pd.DataFrame([{"game_id": "x", "home_team": "BAL", "away_team": "CIN",
                       "gameday": "2026-10-25", "spread_line": -3.0,
                       "total_line": 46.5}])
    b = review.nfl_closes([dict(BET)], g, ABBR)[0]
    assert (b["close_spread"], b["close_total"], b["clv"]) == (3.0, 46.5, -0.5)


def test_cfb_close_from_snapshots():
    closes = {"Cincinnati Bengals|Baltimore Ravens|2026-10-25":
              {"spread": 1.5, "total": 50.0}}
    assert review.cfb_closes([dict(BET)], closes)[0]["clv"] == 1.0


def test_flag_needs_a_meaningful_sample():
    bets = [{**BET, "result": "win", "profit": 9.09, "clv": -0.3}] * 29
    assert not review.flagged(review.market_stats(bets))
    s = review.market_stats(bets + [{**BET, "result": "loss",
                                      "profit": -10.0, "clv": -0.3}])
    assert review.flagged(s) and s["units"] == round((29 * 9.09 - 10) / 10, 2)


def test_qb_surprise_named_as_cause_of_a_big_miss():
    g = pd.DataFrame([
        {"game_id": "a", "season": 2026, "week": 6, "home_team": "BAL",
         "away_team": "PIT", "home_qb_name": "Lamar Jackson",
         "away_qb_name": "A", "home_score": 20, "away_score": 17},
        {"game_id": "b", "season": 2026, "week": 7, "home_team": "BAL",
         "away_team": "CIN", "home_qb_name": "Tyler Huntley",
         "away_qb_name": "Joe Burrow", "home_score": 13, "away_score": 35}])
    b = {**BET, "game_id": "b", "result": "loss", "final": "35-13", "clv": 0.0}
    m = review.big_misses([b], g, ABBR)
    assert len(m) == 1 and "Tyler Huntley started for BAL instead of Lamar Jackson" in m[0]
    # known before the bet: not a surprise, still a big miss by the number
    m = review.big_misses([{**b, "why": "Huntley at quarterback"}], g, ABBR)
    assert "bad beat" in m[0]
    # a narrow loss with no known cause isn't listed
    assert review.big_misses([{**b, "final": "17-13", "why": "Huntley"}], g, ABBR) == []


def test_moneyline_clv_from_closing_prices():
    dog = {**BET, "market": "h2h", "pick": "Cincinnati Bengals", "point": None,
           "price": 170, "first_price": 170}
    g = pd.DataFrame([{"game_id": "x", "home_team": "BAL", "away_team": "CIN",
                       "gameday": "2026-10-25", "spread_line": 3.0,
                       "total_line": 46.5, "home_moneyline": -160,
                       "away_moneyline": 140}])
    b = review.nfl_closes([dog], g, ABBR)[0]
    # closed at a fair ~40.0%; +170 needed 37.0%
    assert 2.5 < b["clv"] < 3.5
    s = review.split([b, dict(BET)], "NFL")
    assert list(s) == ["NFL spreads", "NFL totals", "NFL underdog moneylines"]
    assert "% win chance" in review._row("NFL underdog moneylines",
                                         review.market_stats([b]))


def test_cfb_moneyline_clv_from_recorded_close():
    dog = {**BET, "market": "h2h", "pick": "Cincinnati Bengals", "point": None,
           "price": 170}
    k = f"{BET['away']}|{BET['home']}|2026-10-25"
    b = review.cfb_closes([dog], {k: {"spread": -3.0, "p_home": 0.6}})[0]
    assert 2.9 < b["clv"] < 3.1
